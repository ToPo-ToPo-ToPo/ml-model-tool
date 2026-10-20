"""The two documents an agent (or a person) writes: the data spec and the model spec.

Every key is declared once (``keys.Key``): data keys with their source (``sources/<source>.py``), model
keys here, per model type (``MODEL_TYPES``). The command layer rejects unknown key names before running
(the shapes below); here the defaults are filled in and the values are checked. A wrong value raises
``keys.SpecError`` with the key (``where``), the value and what to write instead. ``spec data`` /
``spec model`` show the same keys (``describe_data`` / ``describe_model``).

Data spec (``data --spec data.json``)::

    {"source": "simp-topopt" | "optopus-overhang" | "simp-fea" | "fea-elastic" | "topodiff", ...keys of that source...}

Model spec (``train --spec model.json --data <dataset>``); ``model.type`` picks the kind (default fno)::

    {"model":  {"type": "fno" | "diffusion" | "flow" | "vae" | "gan", ...keys of that type...},
     "inputs": ["volfrac", "fix_x", ...],      # subset of the dataset's x channels (default: all)
     "loss":   "bce" | "mse" | "rel_l2",       # FNO only; default: bce for density targets, rel_l2 for fields
     "train":  {"epochs": ..., "batch": ..., "lr": ..., ...},
     "sample": {"samples": 4, ...}}            # generative types only: defaults of evaluate / predict
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agk.lang import tr

from .keys import (Bool, Choice, Int, Ints, Key, Names, Num, Section, SpecError, check_keys, defaults, example,
                   names)
from .sources import SOURCES

DEVICES = ("auto", "cpu", "cuda", "mps")
LOSSES = ("bce", "mse", "rel_l2")
SCHEDULES = ("cosine", "linear")
MAX_SAMPLES = 64
ATTN_MIN_CHANNELS = 64          # PhysicsNeMo's U-Net block: one attention head per 64 channels


# ---------------------------------------------------------------------------
# model types
# ---------------------------------------------------------------------------
FNO_KEYS = [
    Key("latent_channels", 64, Int(1), ("FNO の内部のチャネル数", "channels inside the FNO")),
    Key("num_fno_layers", 4, Int(1), ("FNO の層の数", "FNO layers")),
    Key("num_fno_modes", 16, Int(1), ("残すフーリエモードの数（格子の半分以下が目安）", "Fourier modes kept (at most half the grid)")),
    Key("padding", 8, Int(0), ("周期でない境界のための詰め物（セル数）", "padding for non-periodic boundaries (cells)")),
    Key("decoder_layers", 1, Int(1), ("出力側の全結合層の数", "layers of the output MLP")),
    Key("decoder_layer_size", 64, Int(1), ("出力側の全結合層の幅", "width of the output MLP")),
    Key("coord_features", True, Bool(), ("座標 2 チャネルを入力に足す（PhysicsNeMo の既定）", "add the 2 coordinate channels (PhysicsNeMo default)")),
]
UNET_KEYS = [
    Key("model_channels", 64, Int(4), ("U-Net の基本のチャネル数（4 の倍数）", "base channels of the U-Net (a multiple of 4)")),
    Key("channel_mult", [1, 2, 2, 2], Ints(1, 16, 1, 6),
        ("段ごとのチャネル倍率。段の数 = 長さ。1 段下るごとに格子が半分になる",
         "channel multiplier per level; levels = its length; each level halves the grid")),
    Key("num_blocks", 2, Int(1, 8), ("段ごとの残差ブロック数", "residual blocks per level")),
    Key("attn_levels", [2], Ints(0, 5, 0, 6, unique=True),
        ("自己注意を入れる段（0 = 元の解像度、1 = 1/2、…）。[] で無し。最下段は常に入る",
         "levels with self-attention (0 = full resolution, 1 = half, ...); [] = none; the bottom always has it")),
    Key("dropout", 0.1, Num(0, 1, hi_open=True), ("ドロップアウトの割合", "dropout rate")),
]
LATENT_KEY = ("潜在ベクトルの次元", "latent vector size")


@dataclass(frozen=True)
class ModelType:
    name: str
    about: tuple[str, str]
    keys: list[Key]
    generative: bool = False
    sample: list[Key] = field(default_factory=list)
    one_pass: bool = False                       # makes a design in one pass (no sampling steps)


def _sample_keys(steps: int = 0, eta: bool = False) -> list[Key]:
    out = [Key("samples", 4, Int(1, MAX_SAMPLES), ("1 条件あたりの案の数", "designs per condition"))]
    if steps:
        out.append(Key("steps", steps, Int(1, 1000), ("生成の段数（少ないほど速いが粗い）", "sampling steps (fewer = faster, coarser)")))
    if eta:
        out.append(Key("eta", 0.0, Num(0, 1), ("0 で DDIM（ノイズを足さない）、1 で DDPM に近い", "0 = DDIM (no added noise), 1 = DDPM-like")))
    out.append(Key("seed", 0, Int(0), ("ノイズの種（同じ種なら同じ案）", "noise seed (the same seed gives the same designs)")))
    return out


MODEL_TYPES = {t.name: t for t in [
    ModelType("fno", ("PhysicsNeMo の FNO。1 条件に 1 つの答え（回帰・サロゲート）",
                      "PhysicsNeMo FNO: one answer per condition (regression / surrogate)"), FNO_KEYS),
    ModelType("diffusion", ("条件付き拡散モデル（PhysicsNeMo の TopoDiff の U-Net。DDPM で学習、DDIM で生成）。1 条件から何通りもの案",
                            "conditional diffusion (PhysicsNeMo TopoDiff U-Net; DDPM training, DDIM sampling): many designs per condition"),
              [*UNET_KEYS,
               Key("timesteps", 1000, Int(1, 10000), ("拡散の段数（学習）", "diffusion steps (training)")),
               Key("schedule", "cosine", Choice(SCHEDULES), ("ノイズの計画: cosine（Nichol & Dhariwal）・linear（Ho ら）",
                                                             "noise schedule: cosine (Nichol & Dhariwal), linear (Ho et al.)"))],
              generative=True, sample=_sample_keys(50, eta=True)),
    ModelType("flow", ("Flow matching（同じ U-Net。ノイズから形へのまっすぐな流れを学び、Euler 法で生成）。拡散より少ない段数で済む",
                       "flow matching (the same U-Net; a straight flow from noise to design, Euler steps): fewer steps than diffusion"),
              list(UNET_KEYS), generative=True, sample=_sample_keys(20)),
    ModelType("vae", ("条件付き VAE（潜在ベクトルで動く U-Net。1 回の計算で生成）", "conditional VAE (a U-Net driven by a latent vector; one pass)"),
              [*UNET_KEYS, Key("latent_dim", 32, Int(1, 4096), LATENT_KEY),
               Key("kl_weight", 0.01, Num(0), ("KL の重み。大きいほど案を引きやすいが形がぼやける",
                                               "KL weight; larger = easier to sample, blurrier designs"))],
              generative=True, sample=_sample_keys(), one_pass=True),
    ModelType("gan", ("条件付き GAN（U-Net の生成器と PatchGAN の識別器。1 回の計算で生成）",
                      "conditional GAN (U-Net generator, PatchGAN discriminator; one pass)"),
              [*UNET_KEYS, Key("latent_dim", 64, Int(1, 4096), LATENT_KEY),
               Key("l1_weight", 10.0, Num(0), ("正解との L1 誤差の重み。大きいほど正解に近いが、同じ形になりやすい",
                                               "weight of the L1 error to the truth; larger = closer, but less varied")),
               Key("diversity_weight", 1.0, Num(0), ("2 つの潜在ベクトルから違う形を出させる項の重み（0 で無し）",
                                                     "weight of the term that makes two latent vectors give different designs (0 = off)")),
               Key("disc_channels", 64, Int(1), ("識別器の最初の層のチャネル数", "channels of the discriminator's first layer"))],
              generative=True, sample=_sample_keys(), one_pass=True),
]}
GENERATIVE = tuple(k for k, t in MODEL_TYPES.items() if t.generative)
TYPE_KEY = Key("type", "fno", Choice(tuple(MODEL_TYPES)), ("モデルの種類", "kind of model"))


def model_keys(kind: str) -> list[Key]:
    return [TYPE_KEY, *MODEL_TYPES[kind].keys]


def sample_keys(kind: str) -> list[Key]:
    return MODEL_TYPES[kind].sample


def train_keys(kind: str) -> list[Key]:
    gen = MODEL_TYPES[kind].generative
    return [
        Key("epochs", 200 if gen else 40, Int(1, 100000), ("学習の回数（データ全体を何周するか）", "passes over the training data")),
        Key("batch", 32, Int(1, 4096), ("1 回の更新に使う件数", "samples per update")),
        Key("lr", 2e-4 if gen else 1e-3, Num(0, 1, lo_open=True), ("学習率（OneCycle の最大値）", "learning rate (OneCycle peak)")),
        Key("weight_decay", 0.0 if gen else 1e-4, Num(0), ("重みの減衰（AdamW）", "weight decay (AdamW)")),
        Key("n_train", 0, Int(0), ("学習に使う件数（0 = 全部）。データ量の効果を確かめるとき", "training samples to use (0 = all)")),
        Key("seed", 0, Int(0), ("乱数の種", "random seed")),
        Key("device", "auto", Choice(DEVICES), ("計算する装置。auto は CUDA → Apple の GPU（MPS）→ CPU の順",
                                                "where to compute; auto = CUDA, else Apple's GPU (MPS), else CPU")),
        Key("ema", 0.999 if gen else 0.0, Num(0, 1, hi_open=True),
            ("重みの移動平均（0 = 無し）。検証と保存は平均した重みで行う", "moving average of the weights (0 = off); validation and saving use it")),
        Key("physics", 0.0, Num(0), ("FEM のつり合いの式の重み（0 = 無し）。FNO と節点の変位のデータ（simp-fea の target: displacement）だけ",
                                     "weight of the FEM equilibrium term (0 = off); FNO on nodal displacements only "
                                     "(simp-fea with target: displacement)")),
    ]


TOP_KEYS = [
    Key("model", {}, Section(), ("モデルの種類と大きさ（model.type とその鍵）", "kind and size of the model (model.type and its keys)"),
        ("{}（FNO の既定）", "{} (FNO defaults)")),
    Key("inputs", [], Names(), ("データの入力チャネルのうち使うもの。生成モデルでは生成の条件になる",
                                "input channels of the dataset to use (the condition, for generative models)"),
        ("全部", "all channels")),
    Key("loss", "", Choice(LOSSES), ("損失（FNO だけ）: bce（密度）・rel_l2（場の既定。件ごとの相対 L2）・mse",
                                     "loss (FNO only): bce (density), rel_l2 (fields default; relative L2 per case), mse"),
        ("密度は bce、場は rel_l2", "bce for density, rel_l2 for fields")),
    Key("train", {}, Section(), ("学習の設定", "training settings"), ("{}", "{}")),
    Key("sample", {}, Section(), ("生成モデルだけ: evaluate / predict の既定の案の数など", "generative models only: defaults of evaluate / predict"),
        ("{}", "{}")),
]

# key names for the shape check of the command layer; model keys of every type, checked per type below
DATA_SHAPE_CASES = {name: ["source", *names(src.KEYS)] for name, src in SOURCES.items()}
MODEL_KEYS = list(dict.fromkeys(k.name for kind in MODEL_TYPES for k in model_keys(kind)))
SAMPLE_KEYS = list(dict.fromkeys(k.name for kind in GENERATIVE for k in sample_keys(kind)))
MODEL_SHAPE = {"model": MODEL_KEYS, "inputs": None, "loss": None, "train": names(train_keys("fno")),
               "sample": SAMPLE_KEYS}
MODEL_DEFAULTS = {kind: defaults(model_keys(kind)) | {"type": kind} for kind in MODEL_TYPES}


# ---------------------------------------------------------------------------
# data spec
# ---------------------------------------------------------------------------
def data_spec(doc: dict) -> dict:
    """Fill defaults and check a data spec. Raises SpecError with what to fix."""
    name = doc.get("source", "")
    src = SOURCES.get(name)
    if src is None:
        opts = " / ".join(SOURCES)
        raise SpecError("source", tr(f"{name!r} は知らない。{opts} のどれか（spec data で一覧）",
                                     f"{name!r} is unknown; one of {opts} (list: spec data)"), opts)
    spec = {"source": name, **check_keys({k: v for k, v in doc.items() if k != "source"}, src.KEYS)}
    if "n" in spec and spec["n_test"] + spec["n_val"] >= spec["n"]:
        raise SpecError("n_test", tr(f"n_test + n_val（{spec['n_test'] + spec['n_val']}）が n（{spec['n']}）以上。"
                                     "n を増やすか n_test / n_val を減らす",
                                     f"n_test + n_val ({spec['n_test'] + spec['n_val']}) is not below n ({spec['n']}); "
                                     "raise n or lower n_test / n_val"))
    if hasattr(src, "check"):
        src.check(spec)
    return spec


def describe_data(name: str = "") -> dict:
    """What ``spec data [--source NAME]`` answers."""
    if not name:
        return {"document": tr("データ仕様（data --spec <ファイル>）", "data spec (data --spec <file>)"),
                "sources": [{"source": n, "about": tr(*s.ABOUT), "x_channels": s.X_CHANNELS, "y_channels": s.Y_CHANNELS}
                            for n, s in SOURCES.items()],
                "next": tr("spec data --source <source> で、その source の鍵と例", "spec data --source <source> for its keys and an example")}
    if name not in SOURCES:
        opts = " / ".join(SOURCES)
        raise SpecError("source", tr(f"{name!r} は知らない。{opts} のどれか", f"{name!r} is unknown; one of {opts}"), opts)
    s = SOURCES[name]
    return {"document": tr("データ仕様（data --spec <ファイル>）", "data spec (data --spec <file>)"),
            "source": name, "about": tr(*s.ABOUT), "x_channels": s.X_CHANNELS, "y_channels": s.Y_CHANNELS,
            "keys": [k.row() for k in s.KEYS],
            "example": {"source": name, **example(s.KEYS)},
            "next": tr("example を JSON のファイルに書き、変えたいキーだけ直して data --spec <ファイル>（省いたキーは既定）",
                       "write example to a JSON file, change only what you need, run data --spec <file> "
                       "(left-out keys take the default)")}


# ---------------------------------------------------------------------------
# model spec
# ---------------------------------------------------------------------------
def unet_resolutions(m: dict, grid: tuple[int, int]) -> list[int]:
    """The long side (in cells, after padding) at each level of the U-Net (PhysicsNeMo names levels by this)."""
    from .generative import pad_multiple

    mult = pad_multiple(len(m["channel_mult"]))
    long = max(-(-int(grid[0]) // mult) * mult, -(-int(grid[1]) // mult) * mult)
    return [long >> level for level in range(len(m["channel_mult"]))]


def _check_unet(m: dict, grid) -> None:
    """Checks across the U-Net keys (the generative types)."""
    if m["model_channels"] % 4:
        raise SpecError("model.model_channels", tr(f"{m['model_channels']} は 4 の倍数でない（group normalization）。例: 32・64",
                                                   f"{m['model_channels']} is not a multiple of 4 (group normalization); e.g. 32, 64"))
    levels = len(m["channel_mult"])
    bad = [lv for lv in m["attn_levels"] if lv >= levels]
    if bad:
        raise SpecError("model.attn_levels", tr(f"{bad} は U-Net に無い段。0〜{levels - 1}（0 = 元の解像度、1 = 1/2、…）",
                                                f"{bad} are not U-Net levels; use 0-{levels - 1} (0 = full resolution, 1 = half, ...)"))
    bottom = m["model_channels"] * m["channel_mult"][-1]
    if bottom < ATTN_MIN_CHANNELS:      # the U-Nets always attend at their bottom level
        raise SpecError("model.model_channels", tr(
            f"U-Net の最下段は常に自己注意を持ち、{ATTN_MIN_CHANNELS} チャネル以上が要る（model_channels × channel_mult[-1] = {bottom}）",
            f"the bottom level of the U-Net always has attention and needs at least {ATTN_MIN_CHANNELS} channels "
            f"(model_channels x channel_mult[-1] = {bottom})"))
    for lv in m["attn_levels"]:
        ch = m["model_channels"] * m["channel_mult"][lv]
        if ch < ATTN_MIN_CHANNELS:
            raise SpecError("model.attn_levels", tr(
                f"段 {lv} の自己注意には {ATTN_MIN_CHANNELS} チャネル以上が要る（model_channels × channel_mult = {ch}）。"
                f"チャネルを増やすか {lv} を外す",
                f"attention at level {lv} needs at least {ATTN_MIN_CHANNELS} channels there "
                f"(model_channels x channel_mult = {ch}); raise them or drop {lv}"))
    if grid is not None and min(-(-int(g) // (2 ** (levels - 1))) for g in grid) < 2:
        raise SpecError("model.channel_mult", tr(f"{levels} 段では格子 {list(grid)} が 2 セル未満まで縮む。段を減らす",
                                                 f"{levels} levels halve the grid {list(grid)} below 2 cells; use fewer levels"))


def model_spec(doc: dict, x_channels: list[str], y_kind: str, grid=None) -> dict:
    """Fill defaults and check a model spec against the dataset it will train on (``grid`` = (H, W))."""
    given = dict(doc.get("model") or {})
    TYPE_KEY.check(given.get("type", "fno"), "model.type")
    kind = given.get("type", "fno")
    t = MODEL_TYPES[kind]
    own = names(model_keys(kind))
    foreign = [k for k in given if k not in own]
    if foreign:
        raise SpecError("model", tr(f"{', '.join(foreign)} は type {kind} では使わない（{kind} の鍵: {', '.join(own)}。spec model --type {kind}）",
                                    f"{', '.join(foreign)} not used by type {kind} (keys of {kind}: {', '.join(own)}; "
                                    f"spec model --type {kind})"), ", ".join(own))
    model = check_keys(given, model_keys(kind), "model.")
    train = check_keys(doc.get("train") or {}, train_keys(kind), "train.")
    if t.generative:
        _check_unet(model, grid)
    inputs = doc.get("inputs") or list(x_channels)
    unknown = [c for c in inputs if c not in x_channels] if isinstance(inputs, list) else inputs
    if unknown or len(set(inputs)) != len(inputs):
        raise SpecError("inputs", tr(f"知らないか重複したチャネル {unknown}（このデータのチャネル: {', '.join(x_channels)}）",
                                     f"unknown or repeated channels {unknown} (the dataset has {', '.join(x_channels)})"),
                        ", ".join(x_channels))
    if t.generative:
        if doc.get("loss") is not None:
            raise SpecError("loss", tr(f"{kind} は自分の損失を持つ。loss は書かない", f"a {kind} model has its own loss; leave loss out"))
        loss = kind
    else:
        loss = doc.get("loss") or ("bce" if y_kind == "density" else "rel_l2")
        Key("loss", "", Choice(LOSSES), ("", "")).check(loss, "loss")
        if loss == "bce" and y_kind != "density":
            raise SpecError("loss", tr("bce は [0, 1] の密度の正解だけ。場には rel_l2 か mse",
                                       "bce needs density targets in [0, 1]; use rel_l2 or mse for fields"))
    if train["physics"] and kind != "fno":
        raise SpecError("train.physics", tr("つり合いの項は model.type fno だけ（変位を予測するモデル）",
                                            "the equilibrium term is for model.type fno (it predicts the displacement)"))
    out = {"model": model, "inputs": inputs, "loss": loss, "train": train}
    if t.generative:
        out["sample"] = sample_spec(doc.get("sample") or {}, kind, model.get("timesteps"))
    elif doc.get("sample"):
        raise SpecError("sample", tr(f"生成モデル（model.type {' / '.join(GENERATIVE)}）だけ",
                                     f"only for generative models (model.type {' / '.join(GENERATIVE)})"))
    return out


def sample_spec(doc: dict, kind: str, timesteps=None) -> dict:
    """How many designs per condition, and for diffusion / flow how many steps (``sample`` of the model spec)."""
    keys = sample_keys(kind)
    own = names(keys)
    foreign = [k for k in doc if k not in own]
    if foreign:
        raise SpecError("sample", tr(f"{', '.join(foreign)} は type {kind} では使わない（鍵: {', '.join(own)}）",
                                     f"{', '.join(foreign)} not used by type {kind} (keys: {', '.join(own)})"), ", ".join(own))
    s = check_keys(doc, keys, "sample.")
    if timesteps and s.get("steps", 0) > timesteps:
        raise SpecError("sample.steps", tr(f"{s['steps']} は model.timesteps（{timesteps}）を超える",
                                           f"{s['steps']} is more than model.timesteps ({timesteps})"))
    return s


def describe_model(kind: str = "") -> dict:
    """What ``spec model [--type KIND]`` answers."""
    doc = tr("モデル仕様（train --spec <ファイル> --data <データ>）", "model spec (train --spec <file> --data <dataset>)")
    if not kind:
        return {"document": doc,
                "types": [{"type": n, "about": tr(*t.about), "generative": t.generative} for n, t in MODEL_TYPES.items()],
                "spec": [k.row() for k in TOP_KEYS],
                "next": tr("spec model --type <type> で、その種類の鍵と例", "spec model --type <type> for its keys and an example")}
    TYPE_KEY.check(kind, "type")
    t = MODEL_TYPES[kind]
    out = {"document": doc, "type": kind, "about": tr(*t.about), "generative": t.generative,
           "spec": [k.row() for k in TOP_KEYS if t.generative or k.name != "sample"],
           "model": [k.row() for k in model_keys(kind)],
           "train": [k.row() for k in train_keys(kind)]}
    if t.generative:
        out["sample"] = [k.row() for k in t.sample]
        out["notes"] = tr("loss は書かない（種類ごとの損失を持つ）。evaluate / predict の --samples・--steps で sample を上書きできる",
                          "leave loss out (each type has its own); --samples / --steps of evaluate / predict override sample")
    ex = {"model": {**example(model_keys(kind)), "type": kind}, "train": example(train_keys(kind))}
    if t.generative:
        ex["sample"] = example(t.sample)
    out["example"] = ex
    out["next"] = tr("example を JSON のファイルに書き、変えたいキーだけ直して train --spec <ファイル> --data <データ>"
                     "（省いたキーは既定。inputs で入力チャネルを絞れる）",
                     "write example to a JSON file, change only what you need, run train --spec <file> --data <dataset> "
                     "(left-out keys take the default; inputs selects input channels)")
    return out
