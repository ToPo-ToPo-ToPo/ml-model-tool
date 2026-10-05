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

```json
{"model":  {"type": "fno", "latent_channels": 64, "num_fno_layers": 4, "num_fno_modes": 16,
            "padding": 8, "decoder_layers": 1, "decoder_layer_size": 64, "coord_features": true},
 "inputs": ["volfrac", "fix_x", "fix_y", "load_x", "load_y"],
 "loss":   "bce",
 "train":  {"epochs": 40, "batch": 32, "lr": 0.001, "weight_decay": 0.0001,
            "n_train": 0, "seed": 0, "device": "auto"}}
```

- `model`: PhysicsNeMo の FNO の大きさ。`coord_features` は座標 2 チャネルを自動で足す（PhysicsNeMo の既定）
- `inputs`: データの入力チャネルのうち使うもの（省略で全部）
- `loss`: `bce`（密度。出力に sigmoid）か `mse`（場）。省略でデータの種類から決まる
- `train.n_train`: 学習に使う件数（0 = 全部）。`device`: `auto`（CUDA があれば使う）・`cpu`・`cuda`・`mps`（Apple の GPU。FFT が動くかは未確認）
- 学習率は OneCycle。検証の損失が最小のエポックを `model.mdlus` に保存する

## ジョブのフォルダ

| コマンド | 主な成果物 |
|---|---|
| `data` | `dataset.npz`（x / y / split / aux_*）・`dataset.json`・`preview.png`・`summary.md`（TopoDiff は `level_<k>.npz` も） |
| `train` | `model.mdlus`・`model.json`・`history.csv`（毎エポックの学習・検証の損失）・`loss.png`・`summary.md` |
| `evaluate` | `metrics.json`・`compare.png`（先頭 8 件の正解と予測）・`predictions.npz`・`summary.md` |
| `predict` | `predictions.npz`（index と pred）・`preview.png`・`summary.md` |

どれも `status.json`（state・progress・summary・error・artifacts）を持つ。
