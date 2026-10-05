"""The commands of ``mmt`` — the window's input box, the CLI and the MCP tool ``run`` are all built
from this one declaration (agk.connect). People and brains type the same lines::

    data --spec data.json                                   make a dataset (SIMP / FEM / import)
    train --spec model.json --data jobs/<data job>          train a model on it
    evaluate --model jobs/<train job> --data jobs/<data job> [--split test]
    predict --model jobs/<train job> --data <dataset> [--split all]
    help [<command>] / job list / job status --job-dir jobs/<…> / job stop --job-dir jobs/<…>

How to write the two JSON documents is in the skill (skills/ml-model/SKILL.md) and docs/specs.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agk.connect import By, Commands, Doc, File, Obj, Value, rejected, tr
from agk.connect import InputError

from ml_model_tool import spec as specs
from ml_model_tool.sources import SOURCES
from ml_model_tool_cli import jobs

APP = "ml-model-tool"
APP_ROOT = Path(__file__).resolve().parents[2]

DATA_SHAPE = By("source", {name: Obj(keys) for name, keys in specs.DATA_SHAPE_CASES.items()})
MODEL_SHAPE = Obj({"model": Obj(specs.MODEL_SHAPE["model"]), "inputs": None, "loss": None,
                   "train": Obj(specs.MODEL_SHAPE["train"])})


def read_json(path: str | Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError("spec", tr(f"JSON として読めない: {exc}", f"cannot read as JSON: {exc}")) from exc
    if not isinstance(doc, dict):
        raise InputError("spec", tr("JSON オブジェクト（{...}）で書く", "write a JSON object ({...})"))
    return doc


def load_data_spec(path: str | Path) -> dict:
    """Data spec with paths (``root`` / ``topodiff_repo``) made absolute from the spec's folder."""
    doc = read_json(path)
    for key in ("root", "topodiff_repo"):
        if isinstance(doc.get(key), str) and doc[key]:
            doc[key] = str((Path(path).parent / Path(doc[key]).expanduser()).resolve())
    return doc


def _check_data_spec(path: Path) -> None:
    try:
        specs.data_spec(load_data_spec(path))
    except ValueError as exc:
        raise InputError("spec", str(exc)) from exc


def _dataset_file(p: Path) -> Path:
    q = p / "dataset.npz" if p.is_dir() else p
    if q.suffix != ".npz" or not q.is_file() or not q.with_suffix(".json").is_file():
        raise ValueError(tr(f"データセットではない: {p}（data ジョブのフォルダか dataset.npz / level_<k>.npz を渡す）",
                            f"not a dataset: {p} (pass a data job folder or dataset.npz / level_<k>.npz)"))
    return q


def _model_folder(p: Path) -> Path:
    q = p if p.is_dir() else p.parent
    if not (q / "model.mdlus").is_file() or not (q / "model.json").is_file():
        raise ValueError(tr(f"学習済みモデルのフォルダではない: {p}（train ジョブのフォルダを渡す）",
                            f"not a trained model folder: {p} (pass a train job folder)"))
    return q


def _check_train(call) -> None:
    meta = json.loads(Path(call.args["data"]).with_suffix(".json").read_text(encoding="utf-8"))
    if "y" not in (meta.get("shape") or {}) or not (meta.get("shape") or {}).get("y"):
        raise InputError("data", tr("このデータセットには正解（y）が無い（評価専用。train には data ジョブの dataset.npz を渡す）",
                                    "this dataset has no targets (y) (evaluation only; pass dataset.npz of a data job to train)"))
    try:
        specs.model_spec(read_json(call.args["spec"]), meta["x_channels"], meta["y_kind"])
    except ValueError as exc:
        raise InputError("spec", str(exc)) from exc


def _artifacts(st: dict, job_dir: str, base: str) -> list[dict]:
    notes = {"summary": tr("要約（まずこれを読む）", "summary (read this first)"),
             "dataset": tr("データセット（train / evaluate の --data に渡す）", "dataset (pass to --data of train / evaluate)"),
             "model": tr("学習済みモデル（このフォルダを --model に渡す）", "trained model (pass this folder to --model)"),
             "compare": tr("正解と予測の比較図", "truth vs prediction figure"),
             "loss_plot": tr("損失の推移", "loss history"), "preview": tr("見本の図", "preview figure"),
             "metrics": tr("主要数値", "key numbers")}
    out = []
    for key, rel in (st.get("artifacts") or {}).items():
        p = Path(job_dir) / rel
        try:
            path = str(p.resolve().relative_to(Path(base).resolve()))
        except ValueError:
            path = str(p)
        out.append({"path": path, "format": p.suffix.lstrip(".").lower() or "folder",
                    "description": notes.get(key, key)})
    return out


def _finish(result: dict, call, options: dict) -> dict:
    """The answer to the brain: state, summary, artifacts (paths relative to the work folder)."""
    if COMMANDS.get(call.name).local is not None:
        return result
    base = options["aios_base"]
    job_dir = str(result.get("job_dir") or options.get("aios_out") or "")
    status_file = Path(job_dir) / jobs.agk_jobs.STATUS_FILENAME
    if not job_dir or not status_file.exists():
        err = result.get("error") or {}
        return rejected(err if isinstance(err, dict) and err else
                        {"stage": "start", "message": str(err or tr("ジョブが始まらなかった", "the job did not start"))})
    st = jobs.agk_jobs.read_status(job_dir)
    state = str(st.get("state") or "unknown")
    try:
        rel_dir = str(Path(job_dir).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        rel_dir = job_dir
    out = {"ok": state == "done", "state": state, "job_dir": rel_dir, "command": st.get("command"),
           "summary": st.get("summary") or {}, "error": st.get("error")}
    if state == "running":
        out["progress"] = (st.get("progress") or {}).get("message", "")
    arts = _artifacts(st, job_dir, base)
    if arts:
        out["artifacts"] = arts
    return out


COMMANDS = Commands(prog="mmt", app=APP, module="ml_model_tool_cli.cli", root=APP_ROOT, finish=_finish)

COMMANDS.add(
    "data", help=("データセットを作る（SIMP・FEM で生成、または取り込み）", "make a dataset (generate by SIMP / FEM, or import)"),
    args=[Doc("spec", "ml-model-tool-data/1", validate=_check_data_spec, shape=DATA_SHAPE, stage="spec",
              help=("データ仕様（JSON）。source: simp-topopt / fea-elastic / topodiff と、その鍵",
                    "data spec (JSON): source (simp-topopt / fea-elastic / topodiff) and its keys"))],
    description=("""学習用のデータセットをジョブフォルダに作る（`mmt data`）。

source ごとの鍵と既定値は docs/specs.md と手順書 ml-model。出力は dataset.npz（x / y / split / aux_*）と
dataset.json（チャネル名・形）、preview.png（先頭の見本）、summary.md。SIMP は 1 件数秒かかる（600 件で数十分）。""",
                 """Make a training dataset in a job folder (`mmt data`).

Keys and defaults per source are in docs/specs.md and the skill ml-model. Writes dataset.npz (x / y / split / aux_*),
dataset.json (channel names, shapes), preview.png (first samples) and summary.md. SIMP takes seconds per sample."""),
)

COMMANDS.add(
    "train", help=("モデルを学習する", "train a model"),
    args=[Doc("spec", "ml-model-tool-model/1", shape=MODEL_SHAPE, stage="spec",
              help=("モデル仕様（JSON）。model / inputs / loss / train。全部省略可（{} で既定）",
                    "model spec (JSON): model / inputs / loss / train; all optional ({} = defaults)")),
          File("data", resolve=_dataset_file, help=("data ジョブのフォルダか dataset.npz", "a data job folder or dataset.npz"))],
    check=_check_train,
    description=("""データセットでモデル（PhysicsNeMo の FNO）を学習する（`mmt train`）。

inputs でデータの入力チャネルの一部だけを使える（入力の効果を 1 つずつ確かめる）。train.n_train で学習件数を絞れる。
毎エポックの学習・検証の損失を history.csv と loss.png に残し、検証の損失が最小のエポックを model.mdlus に保存する。""",
                 """Train a model (PhysicsNeMo FNO) on a dataset (`mmt train`).

inputs selects a subset of the dataset's input channels (to test inputs one at a time); train.n_train limits the
training samples. Train / validation losses of every epoch go to history.csv and loss.png; the epoch with the lowest
validation loss is saved as model.mdlus."""),
)

COMMANDS.add(
    "evaluate", help=("学習済みモデルを評価する", "evaluate a trained model"),
    args=[File("model", resolve=_model_folder, help=("train ジョブのフォルダ", "a train job folder")),
          File("data", resolve=_dataset_file, help=("data ジョブのフォルダか dataset.npz / level_<k>.npz",
                                                     "a data job folder or dataset.npz / level_<k>.npz")),
          Value("split", str, "test", choices=("test", "val", "train"),
                help=("評価する分け方（既定 test）", "which split to evaluate (default test)"))],
    description=("""学習済みモデルを予測させて正解と比べる（`mmt evaluate`）。

密度: IoU・MAE・体積の誤差。場: 相対 L2 誤差。simp-topopt はさらに予測形状のコンプライアンスを FEM で解いて SIMP と比べる。
TopoDiff のテスト（level_<k>.npz）は TopoDiff の評価コードで CE / VFE / LD / FM を出す。図は compare.png。""",
                 """Predict with a trained model and compare with the truth (`mmt evaluate`).

Density: IoU, MAE, volume error. Fields: relative L2 error. simp-topopt also solves the predicted design by FEM and
compares its compliance with SIMP. TopoDiff test levels (level_<k>.npz) get CE / VFE / LD / FM with TopoDiff's
analysis. Figure: compare.png."""),
)

COMMANDS.add(
    "predict", help=("学習済みモデルで予測する", "predict with a trained model"),
    args=[File("model", resolve=_model_folder, help=("train ジョブのフォルダ", "a train job folder")),
          File("data", resolve=_dataset_file, help=("入力を持つデータセット", "a dataset with the inputs")),
          Value("split", str, "all", choices=("all", "test", "val", "train"),
                help=("予測する分け方（既定 all）", "which split to predict (default all)"))],
    description=("predictions.npz（index と pred）と preview.png を書く（`mmt predict`）。",
                 "Writes predictions.npz (index and pred) and preview.png (`mmt predict`)."),
)

COMMANDS.add_job_commands(schema=jobs.JOB_SCHEMA, jobspec_schema=jobs.JOBSPEC_SCHEMA)


def run(name: str, args: dict, out: Optional[str], base: str) -> dict:
    """Run one declared job command (the CLI, the window and the MCP tool all end up here)."""
    if name == "data":
        return jobs.run_data(specs.data_spec(load_data_spec(args["spec"])), out)
    if name == "train":
        return jobs.run_train(read_json(args["spec"]), args["data"], out)
    if name == "evaluate":
        return jobs.run_evaluate(args["model"], args["data"], args["split"], out)
    if name == "predict":
        return jobs.run_predict(args["model"], args["data"], args["split"], out)
    raise KeyError(name)
