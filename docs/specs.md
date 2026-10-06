# 仕様の書き方（データ仕様・モデル仕様）

どちらも JSON。知らないキーは走らせる前に差し戻される（書き間違いを黙って既定値で走らせない）。
省略したキーは既定値になる。例は `skills/ml-model/examples/`。

## データ仕様（`data --spec data.json`）

`source` で種類を選ぶ。種類ごとに使えるキーが違う。

### `simp-topopt` — SIMP のトポロジー最適化結果

| キー | 既定 | 意味 |
|---|---|---|
| `n` | 600 | 件数（1 件ごとに SIMP を最後まで回す。64×32 で数秒） |
| `nelx`, `nely` | 64, 32 | 要素数（横・縦） |
| `volfrac` | [0.3, 0.5] | 体積率の範囲 [最小, 最大] |
| `supports` | 3 種すべて | 固定の種類: `cantilever`（左端を固定）・`both-ends`（左右端）・`pin-roller`（左下 3 節点を x・y、右下 3 節点を y だけ固定） |
| `load_nodes` | [2, 3] | 線荷重の節点数の範囲（境界上。合計の大きさ 1、向きはランダム） |
| `rmin`, `maxiter` | 1.5, 200 | SIMP の密度フィルタ半径・最大反復 |
| `workers` | 1 | 並列のプロセス数 |
| `n_test`, `n_val`, `seed` | 60, 30, 0 | テスト・検証の件数と乱数の種（同じ種なら同じテスト件） |

入力（要素の格子）: `volfrac`・`fix_x`・`fix_y`・`load_x`・`load_y`・`sed`（全面材料の FEM のひずみエネルギー密度。
log(sed/平均) を [-10, 10] に切って 5 で割る）。出力: `density`。`aux`: 節点の固定・荷重、SIMP 結果のコンプライアンス。

### `fea-elastic` — 線形弾性の変位場

| キー | 既定 | 意味 |
|---|---|---|
| `n` | 2060 | 件数（1 件 FEM 1 回） |
| `size` | 64 | 正方形の一辺の要素数 |
| `material` | `"full"` | 材料の領域（いまは全面だけ） |
| `min_support_distance` | 16 | 2 本の線拘束の最小間隔（要素数）。近いとほぼ自由に回転して変位が極端になる |
| `n_test`, `n_val`, `seed` | 60, 40, 0 | |

拘束: 2〜3 節点の水平・垂直の線 2 本（x・y 固定）。荷重: 2〜3 節点の線 1 本（合計 1、向きランダム、拘束と重ならない）。
入力（節点の格子）: `material`・`fix_x`・`fix_y`・`load_x`・`load_y`。出力: `ux`・`uy`。

### `topodiff` — TopoDiff データセットの取り込み

| キー | 既定 | 意味 |
|---|---|---|
| `root` | （必須） | `dataset_1_diff` のフォルダ（仕様ファイルからの相対か絶対） |
| `limit` | 0 | 学習データの先頭から使う件数（0 = 全部、約 3 万件） |
| `levels` | [1, 2] | 取り込むテスト（`level_<k>.npz`。正解の形は無く、SIMP のコンプライアンスだけ） |
| `topodiff_repo` | "" | github.com/francoismaze/topodiff のクローン（テストを TopoDiff の評価で測るとき必須） |
| `n_test`, `n_val`, `seed` | 300, 300, 0 | 学習データから取り分ける件数 |

入力: `volfrac`・`sed`・`von_mises`・`load_x`・`load_y`（TopoDiff と同じ値をそのまま）。出力: `density`。

## モデル仕様（`train --spec model.json --data <データ>`）

`model.type` でモデルの種類を選ぶ（省略で `fno`）。種類ごとに `model` の鍵が違い、ほかの種類の鍵を書くと差し戻す。

| `model.type` | モデル | 1 条件に対する答え |
|---|---|---|
| `fno`（既定） | PhysicsNeMo の FNO（回帰・サロゲート） | 1 つ |
| `diffusion` | 条件付き拡散モデル（PhysicsNeMo の TopoDiff の U-Net、DDPM で学習・DDIM で生成） | 何通りでも（`sample.samples` 案） |
| `flow` | Flow matching（同じ U-Net。ノイズから形へのまっすぐな流れを学び、Euler 法で生成） | 何通りでも |
| `vae` | 条件付き VAE（潜在ベクトルで動く U-Net。1 回の計算で生成） | 何通りでも |
| `gan` | 条件付き GAN（潜在ベクトルで動く U-Net と PatchGAN の識別器。1 回の計算で生成） | 何通りでも |

`diffusion` / `flow` / `vae` / `gan` を生成モデルと呼ぶ。U-Net の鍵（`model_channels`・`channel_mult`・`num_blocks`・
`attn_levels`・`dropout`）と `train` の既定、評価のしかたは 4 つで共通。

### `fno`

```json
{"model":  {"type": "fno", "latent_channels": 64, "num_fno_layers": 4, "num_fno_modes": 16,
            "padding": 8, "decoder_layers": 1, "decoder_layer_size": 64, "coord_features": true},
 "inputs": ["volfrac", "fix_x", "fix_y", "load_x", "load_y"],
 "loss":   "bce",
 "train":  {"epochs": 40, "batch": 32, "lr": 0.001, "weight_decay": 0.0001,
            "n_train": 0, "seed": 0, "device": "auto", "ema": 0}}
```

- `model`: PhysicsNeMo の FNO の大きさ。`coord_features` は座標 2 チャネルを自動で足す（PhysicsNeMo の既定）
- `loss`: `bce`（密度。出力に sigmoid）か `mse`（場）。省略でデータの種類から決まる

### `diffusion` — 条件付き拡散モデル

```json
{"model":  {"type": "diffusion", "model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_blocks": 2,
            "attn_levels": [2], "dropout": 0.1, "timesteps": 1000, "schedule": "cosine"},
 "inputs": ["volfrac", "fix_x", "fix_y", "load_x", "load_y", "sed"],
 "train":  {"epochs": 200, "batch": 32, "lr": 0.0002, "weight_decay": 0, "ema": 0.999,
            "n_train": 0, "seed": 0, "device": "auto"},
 "sample": {"samples": 4, "steps": 50, "eta": 0.0, "seed": 0}}
```

正解（密度・場）にノイズを足していき、ノイズから正解へ戻す向きを、入力チャネルを条件として学ぶ。
生成はノイズから始めるので、同じ条件でもノイズの引き方ごとに違う案になる（TopoDiff と同じ考え方）。

| キー | 既定 | 意味 |
|---|---|---|
| `model_channels` | 64 | U-Net の基本のチャネル数（4 の倍数） |
| `channel_mult` | [1, 2, 2, 2] | 段ごとのチャネル倍率。段の数 = 長さ。1 段下るごとに格子が半分になる |
| `num_blocks` | 2 | 段ごとの残差ブロック数 |
| `attn_levels` | [2] | 自己注意（attention）を入れる段（0 = 元の解像度、1 = 1/2、…）。`[]` で無し |
| `dropout` | 0.1 | |
| `timesteps` | 1000 | 拡散の段数（学習） |
| `schedule` | `"cosine"` | ノイズの計画: `cosine`（Nichol & Dhariwal）・`linear`（Ho ら） |

- attention は 64 チャネルに 1 つの頭（head）なので、入れる段のチャネル数（`model_channels × channel_mult`）は 64 以上。
  U-Net の最下段は常に attention を持つので（4 つとも）、`model_channels × channel_mult[-1]` も 64 以上
- 格子は U-Net の段数に合わせて右・下に 0 を埋めて広げ（例: 65×65 → 72×72）、損失は元の領域だけで測る。
  生成した案は元の大きさに切り戻す
- 密度は [0, 1] → [-1, 1] に移して学び、生成の途中の推定も [-1, 1] に収める。場は学習データの平均・標準偏差で正規化する
- `loss` は書かない（生成モデルはそれぞれの損失を持つ。書くと差し戻す）
- `train`: 既定が FNO と違う（`epochs` 200・`lr` 0.0002・`weight_decay` 0・`ema` 0.999）。
  生成モデル 4 つで共通。`ema` は重みの移動平均で、検証と保存は平均した重みで行う。`device: auto` は CUDA → Apple の GPU（MPS）→ CPU の順。
  拡散の U-Net は CPU では非常に遅い（Apple シリコンの Mac で測ると、CPU は MPS の約 100 倍の時間がかかった）
- 検証の損失は毎エポック同じ段・同じノイズで測るので、エポックどうしを比べられる
- `sample`: `evaluate` / `predict` の既定。`samples` は 1 条件あたりの案の数（1〜64）、`steps` は生成の段数
  （DDIM。1〜`timesteps`。少ないほど速い）、`eta` は 0 で DDIM（ノイズを足さない）・1 で DDPM に近い、`seed` は
  ノイズの種（同じ種なら同じ案）。コマンド行の `--samples` / `--steps` で上書きできる

### `flow` — Flow matching

```json
{"model":  {"type": "flow", "model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_blocks": 2,
            "attn_levels": [2], "dropout": 0.1},
 "sample": {"samples": 4, "steps": 20, "seed": 0}}
```

ノイズ（t = 0）と正解（t = 1）を直線で結び、その上の点で「正解 − ノイズ」の向き（速度）を当てるよう学ぶ
（Lipman ら 2023、Liu ら 2023）。生成はノイズから `steps` 回の Euler 法で進める。拡散より少ない段数で済むのが狙い。
U-Net は `diffusion` と同じ（PhysicsNeMo の TopoDiff）。`sample` に `eta` は無い。

### `vae` — 条件付き VAE

```json
{"model":  {"type": "vae", "model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_blocks": 2,
            "attn_levels": [2], "dropout": 0.1, "latent_dim": 32, "kl_weight": 0.01},
 "sample": {"samples": 4, "seed": 0}}
```

学習では、正解と条件から潜在ベクトル z を推定する encoder と、条件と z から形を作る decoder（U-Net）を一緒に学ぶ。
生成では z を標準正規分布から引き、decoder を 1 回通す。保存するのは decoder だけ。

| キー | 既定 | 意味 |
|---|---|---|
| `latent_dim` | 32 | 潜在ベクトルの次元 |
| `kl_weight` | 0.01 | 損失 = 再構成の二乗誤差（セル平均）+ `kl_weight` × KL ÷ セル数。大きいほど z が正規分布に近づく（案を引きやすい）が形がぼやけやすい |

### `gan` — 条件付き GAN

```json
{"model":  {"type": "gan", "model_channels": 64, "channel_mult": [1, 2, 2, 2], "num_blocks": 2,
            "attn_levels": [2], "dropout": 0.1, "latent_dim": 64, "l1_weight": 10, "diversity_weight": 1,
            "disc_channels": 64},
 "sample": {"samples": 4, "seed": 0}}
```

生成器（条件と z から形を作る U-Net）と識別器（条件と形の組が本物かを小領域ごとに見る PatchGAN、
spectral normalization 付き）を交互に学ぶ。生成は z を引いて生成器を 1 回通す。保存するのは生成器だけ。

| キー | 既定 | 意味 |
|---|---|---|
| `latent_dim` | 64 | 潜在ベクトルの次元 |
| `l1_weight` | 10 | 正解との L1 誤差の重み。大きいほど正解に近づくが、z によらず同じ形になりやすい |
| `diversity_weight` | 1 | 2 つの z から違う形を出させる項の重み（Yang ら 2019。差は 1 で頭打ち）。0 で無し |
| `disc_channels` | 64 | 識別器の最初の層のチャネル数 |

- GAN の損失はエポックの良し悪しを表さないので、**最後のエポック**を保存する。`history.csv` の検証の損失は
  決まったノイズで生成した形の正解との L1 誤差（参考）。学習と検証で量が違うので `overfit_ratio` は出さない
- 学習が不安定になったら `lr` を下げる。似た形しか出ない（`diversity` が 0 に近い）なら `diversity_weight` を上げるか
  `l1_weight` を下げる

### 共通

- `inputs`: データの入力チャネルのうち使うもの（省略で全部）。生成モデルでは生成の条件になる
- `train.n_train`: 学習に使う件数（0 = 全部）。`device`: `auto`・`cpu`・`cuda`・`mps`（FNO の FFT が MPS で動くかは未確認）
- 学習率は OneCycle。検証の損失が最小のエポックを `model.mdlus` に保存する

## 生成モデルの評価（`evaluate`）

1 条件から K 案（`--samples`、既定は `sample.samples`）を作る。

- いつもの数値（IoU・MAE・体積の誤差・相対 L2・コンプライアンス比・TopoDiff の評価指標）は **1 案目**で出す。
  1 回引いた案の性能なので、FNO とそのまま比べられる
- `generative.diversity`: K 案どうしの違い（全組の平均）。密度は材料（0.5 以上）の判定が食い違うセルの割合、
  場は 2 案の差の相対 L2。0 なら毎回同じ形しか出ない
- `generative.best_of_k`: K 案の中で最も良い案の数値。IoU（正解に最も近い案）、相対 L2、`simp-topopt` は
  FEM で解いたコンプライアンスで選んだ案、TopoDiff のテストは CE が最も小さい案。FEM で候補を選ぶ使い方の上限の目安
- `generative.compliance_ratio_all`: 全案のコンプライアンス比（破綻する案の割合を見る）
- `floating`（`simp-topopt`。FNO も）: 浮いた材料。材料（密度 0.5 以上）のうち、要素の辺を通って拘束に触れる要素まで
  つながっていない部分。角だけで接する要素はそこで回転できるので、つながっていないとみなす。`share` は浮いた材料を
  含む案の割合、`share_gt_1pct` は浮いた材料が材料の 1% を超える案の割合、`volume_mean` は材料に占める割合の平均。
  生成モデルは 1 案目の値で、`generative.floating_all` が全案の値。`compare.png` では浮いた材料を含む案に `float` と付く
- `compare.png` は 1 件ごとに正解と最大 4 案を並べる。`predictions.npz` に `samples`（件 × 案 × チャネル × 縦 × 横）
  と `pred`（= 1 案目）

## ジョブのフォルダ

| コマンド | 主な成果物 |
|---|---|
| `data` | `dataset.npz`（x / y / split / aux_*）・`dataset.json`・`preview.png`・`summary.md`（TopoDiff は `level_<k>.npz` も） |
| `train` | `model.mdlus`・`model.json`・`history.csv`（毎エポックの学習・検証の損失）・`loss.png`・`summary.md` |
| `evaluate` | `metrics.json`・`compare.png`（先頭 8 件の正解と予測。生成モデルは最大 4 案）・`predictions.npz`・`summary.md` |
| `predict` | `predictions.npz`（index と pred。生成モデルは samples も）・`preview.png`・`summary.md` |

どれも `status.json`（state・progress・summary・error・artifacts）を持つ。
