# 仕様の書き方（データ仕様・モデル仕様）

どちらも JSON。知らないキーは走らせる前に差し戻される（書き間違いを黙って既定値で走らせない）。
省略したキーは既定値になる。例は `skills/ml-model/examples/`。

鍵の表（既定・値・意味）は `src/ml_model_tool/keys.py` の宣言から作っている（`spec data` / `spec model` の答えと同じ）。
鍵を変えたら `uv run python -m ml_model_tool.keys docs/specs.md` で表を作り直す（ずれているとテストが落ちる）。
誤りは `{stage: "spec", where: <鍵>, message, expected}` で返る（例: `where: "train.lr"`）。

## データ仕様（`data --spec data.json`）

`source` で種類を選ぶ。種類ごとに使えるキーが違う。

### `simp-topopt` — SIMP のトポロジー最適化結果

<!-- keys:data/simp-topopt -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `n` | `600` | 10 以上の整数 | 件数（1 件ごとに SIMP を最後まで回す。64×32 で数秒） |
| `nelx` | `64` | 8 以上 512 以下の整数 | 横の要素数 |
| `nely` | `32` | 8 以上 512 以下の整数 | 縦の要素数 |
| `volfrac` | `[0.3, 0.5]` | [最小, 最大] の数 2 つ（0.05 ≦ 最小 ≦ 最大 ≦ 0.95） | 体積率の範囲 [最小, 最大]（1 件ごとにこの範囲から引く） |
| `supports` | 3 種すべて | ["cantilever", "both-ends", "pin-roller"] から選んだリスト（1 つ以上） | 拘束の種類: cantilever（左端を固定）・both-ends（左右端）・pin-roller（左下 3 節点を x・y、右下 3 節点を y だけ固定） |
| `load_nodes` | `[2, 3]` | [最小, 最大] の整数 2 つ（2 ≦ 最小 ≦ 最大 ≦ 8） | 線荷重の節点数の範囲（境界上。合計の大きさ 1、向きはランダム。点荷重は 2 次元弾性で特異なので使わない） |
| `rmin` | `1.5` | 0 より大きく 10 以下の数 | SIMP の密度フィルタの半径（要素数） |
| `maxiter` | `200` | 1 以上 10000 以下の整数 | SIMP の最大反復回数 |
| `workers` | `1` | 1 以上 256 以下の整数 | 並列のプロセス数 |
| `n_test` | `60` | 0 以上の整数 | テストの件数（学習に使わない。評価はこれで行う） |
| `n_val` | `30` | 0 以上の整数 | 検証の件数（毎エポックの検証の損失。最良のエポックの選択に使う） |
| `seed` | `0` | 0 以上の整数 | 乱数の種（同じ種なら同じデータ・同じテスト件） |
<!-- /keys -->

入力（要素の格子）: `volfrac`・`fix_x`・`fix_y`・`load_x`・`load_y`・`sed`（全面材料の FEM のひずみエネルギー密度。
log(sed/平均) を [-10, 10] に切って 5 で割る）。出力: `density`。`aux`: 節点の固定・荷重、SIMP 結果のコンプライアンス。

### `optopus-overhang` — 最適化結果を自己支持の形に直す（積層造形。ソルバは JuliaOptopus）

入力はトポロジー最適化の結果、出力は同じ条件でサポート無し（自己支持）で造れるように最適化した結果。
生成モデルに「この形を、積層造形でサポート無しに造れる形に直した案」を作らせるためのデータ。
どちらも [JuliaOptopus](https://github.com/ToPo-ToPo-ToPo/JuliaOptopus) で解く（ドライバ
`main/linear_elasticity/TO/am_overhang/ml_dataset.jl`）。

<!-- keys:data/optopus-overhang -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `n` | `600` | 10 以上の整数 | 件数（1 件ごとに JuliaOptopus で最適化を 2 回） |
| `nelx` | `64` | 8 以上 512 以下の整数 | 横の要素数 |
| `nely` | `32` | 8 以上 512 以下の整数 | 縦の要素数 |
| `volfrac` | `[0.3, 0.5]` | [最小, 最大] の数 2 つ（0.05 ≦ 最小 ≦ 最大 ≦ 0.95） | 体積率の範囲 [最小, 最大]（1 件ごとにこの範囲から引く） |
| `supports` | 3 種すべて | ["cantilever", "both-ends", "pin-roller"] から選んだリスト（1 つ以上） | 拘束の種類: cantilever（左端を固定）・both-ends（左右端）・pin-roller（左下 3 節点を x・y、右下 3 節点を y だけ固定） |
| `load_nodes` | `[2, 3]` | [最小, 最大] の整数 2 つ（2 ≦ 最小 ≦ 最大 ≦ 8） | 線荷重の節点数の範囲（境界上。合計の大きさ 1、向きはランダム。点荷重は 2 次元弾性で特異なので使わない） |
| `build_directions` | `["up"]` | ["up", "down", "left", "right"] から選んだリスト（1 つ以上） | 造形の向き（形が伸びていく向き）。up はベースプレートが下辺、down は上辺、left は右辺、right は左辺。複数書くと 1 件ごとにランダムに選ぶ |
| `julia_optopus` | `""` | 文字列 | JuliaOptopus のフォルダ（仕様ファイルからの相対か絶対）。空なら環境変数 JULIA_OPTOPUS |
| `iters` | `100` | 1 以上 10000 以下の整数 | 1 回の最適化の MMA の反復の上限（収束したら早く終わる） |
| `radius` | `1.5` | 1 以上 6 以下の数 | 密度フィルタの半径（要素数） |
| `workers` | `4` | 1 以上 256 以下の整数 | 並列に走らせる Julia のプロセス数（それぞれ最初に JuliaOptopus をコンパイルする） |
| `n_test` | `60` | 0 以上の整数 | テストの件数（学習に使わない。評価はこれで行う） |
| `n_val` | `30` | 0 以上の整数 | 検証の件数（毎エポックの検証の損失。最良のエポックの選択に使う） |
| `seed` | `0` | 0 以上の整数 | 乱数の種（同じ種なら同じデータ・同じテスト件） |
<!-- /keys -->

1 件ごとに JuliaOptopus の 2 つのモードを同じ設定（quad4 要素・大きさ 1、射影 β = 5、p = 3、MMA）で解く。

- `free`（AM の制約なし）→ 入力の形
- `self`（Langelaar のサポート共最適化で、サポートを禁止したもの = 自己支持の形だけ。張り出しは造形方向から 45° まで）→ 正解

体積率・拘束・荷重は入力と正解で同じ。Julia（juliaup など）が要る。

1 件解くごとに `--out` の下の `optopus/cases/` に保存する。途中で止めても（Ctrl-C・`kill`）、同じ仕様・同じ `--out` で
もう一度走らせればまだの件だけを解く。件数・`seed`・`iters`・`radius` などを変えると最初から解く。

入力（要素の格子）: `design`（free の結果）・`volfrac`・`fix_x`・`fix_y`・`load_x`・`load_y`・`base`
（ベースプレートに載る層が 1。造形の向きを表す）。出力: `density`（self の結果）。`aux`: 節点の固定・荷重、正解の
コンプライアンス（評価と同じ Python の FEM で計算）、入力の形のコンプライアンス（`design_compliance`）、造形の向き
（`build`）、JuliaOptopus 自身のコンプライアンス（`optopus_compliance`）。

`data` の `summary.md` に、入力・正解のオーバーハング（`overhang_input_mean` / `overhang_target_mean`: 造れない材料の
割合の平均。正解はほぼ 0 のはず）、自己支持にして増えたコンプライアンス（`compliance_target_over_input_median`）、
正解の灰色率（`gray_target_median`）、Python の FEM と JuliaOptopus のコンプライアンスの差（
`compliance_fem_vs_optopus_max_rel_diff`。同じ FEM なので 1% 未満。JuliaOptopus の密度はわずかに 1 を超えることがあり、
Python 側で [0, 1] に切るぶんだけずれる）が出る。

評価（`task: topopt-am`）: `simp-topopt` と同じコンプライアンス比（正解 = self の結果との比）と浮いた材料に加えて、
`overhang`（造れない材料: 密度 0.5 以上の材料のうち、ベースプレートから下の 3 要素を伝ってつながらない部分。
JuliaOptopus の `floating_fraction` と同じ判定。`share` はそれを含む案の割合、`volume_mean` は材料に占める割合の平均）と、
比べる基準として入力の形の値（`overhang_input`）。生成モデルは全案の値も出る（`generative.overhang_all`）。
`compare.png` は入力の形・正解・案を並べ、造れない材料を含む案に `OH 3%` のように付く。ベースプレートは緑の線。

### `simp-fea` — 最適化した形状の FEM 結果（FEM のサロゲート）

<!-- keys:data/simp-fea -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `n` | `1000` | 10 以上の整数 | 形状の数（1 つごとに SIMP を最後まで回す）。件数 = n × load_cases |
| `load_cases` | `4` | 1 以上 16 以下の整数 | 1 形状にかける荷重ケースの数。1 つ目は最適化に使った荷重、残りは材料に接する境界上の新しい線荷重 |
| `target` | `"fields"` | "fields" / "displacement" のどれか | 学ぶもの。fields: 要素の格子の ux・uy・von_mises・energy。displacement: 節点の格子の ux・uy（FEM の未知数そのもの。train.physics を使える） |
| `nelx` | `64` | 8 以上 512 以下の整数 | 横の要素数 |
| `nely` | `32` | 8 以上 512 以下の整数 | 縦の要素数 |
| `volfrac` | `[0.3, 0.5]` | [最小, 最大] の数 2 つ（0.05 ≦ 最小 ≦ 最大 ≦ 0.95） | 体積率の範囲 [最小, 最大]（1 件ごとにこの範囲から引く） |
| `supports` | 3 種すべて | ["cantilever", "both-ends", "pin-roller"] から選んだリスト（1 つ以上） | 拘束の種類: cantilever（左端を固定）・both-ends（左右端）・pin-roller（左下 3 節点を x・y、右下 3 節点を y だけ固定） |
| `load_nodes` | `[2, 3]` | [最小, 最大] の整数 2 つ（2 ≦ 最小 ≦ 最大 ≦ 8） | 線荷重の節点数の範囲（境界上。合計の大きさ 1、向きはランダム。点荷重は 2 次元弾性で特異なので使わない） |
| `rmin` | `1.5` | 0 より大きく 10 以下の数 | SIMP の密度フィルタの半径（要素数） |
| `maxiter` | `200` | 1 以上 10000 以下の整数 | SIMP の最大反復回数 |
| `workers` | `1` | 1 以上 256 以下の整数 | 並列のプロセス数 |
| `n_test` | `100` | 0 以上の整数 | テストの形状の数（学習に使わない。評価はこれで行う） |
| `n_val` | `50` | 0 以上の整数 | 検証の形状の数（毎エポックの検証の損失。最良のエポックの選択に使う） |
| `seed` | `0` | 0 以上の整数 | 乱数の種（同じ種なら同じデータ・同じテスト件） |
<!-- /keys -->

形状は `simp-topopt` と同じ作り方（拘束 3 種・境界上の線荷重・体積率・SIMP）。荷重ケースの 1 つ目は最適化に使った荷重、
残りは材料に接する境界上の新しい線荷重（2〜3 節点、合計 1、向きランダム。拘束は同じ）。最適化に使っていない荷重も
かけるのは He ら（CMAME 2023）と同じで、同じ形状に違う荷重をかけた結果を学ぶため。形状は SIMP の密度のまま使う
（E = Emin + x³(1 − Emin)）。

入力（要素の格子）: `density`・`fix_x`・`fix_y`・`load_x`・`load_y`。出力（要素の格子、FEM 1 回）:
`ux`・`uy`（要素の 4 節点の平均。y は上向き）・`von_mises`（要素中心のミーゼス応力、平面応力）・`energy`
（要素ごとのコンプライアンスへの寄与。全要素の合計がコンプライアンス Fᵀu に一致する）。`aux`: 節点の固定・荷重、
コンプライアンス、形状の番号。

最適化に使っていない荷重では、細い部材が大きく曲がって変位が非常に大きくなる件がある（変位の最大値は中央値の
100 倍を超えることがある）。そのため場の既定の損失は件ごとの相対 L2 誤差にしている（下のモデル仕様）。

`target: "displacement"` では、入力・出力とも節点の格子（`(nely+1, nelx+1)`）になる。入力の `density` は節点のまわりの
要素の平均、`fix_*`・`load_*` は節点の値。出力は節点の変位 `ux`・`uy`。`aux` に要素の密度（`density`）も持つ。

評価（`task: fea`）: 量ごとの相対 L2 誤差（`rel_l2_by_channel`）、最大ミーゼス応力の誤差
（`peak_stress_error` = |予測の最大 − 正解の最大| ÷ 正解の最大。先行研究で必ず報告される）、予測した `energy` の合計
から出したコンプライアンスの誤差（`compliance_error`）。節点の変位を予測するモデルでは、予測した変位から FEM と同じ式で
応力と energy を計算して同じ指標を出し（`derived_rel_l2`）、つり合いの残差 ‖K(ρ)u − F‖ ÷ ‖F‖（`equilibrium_residual`。
FEM の解では 0）も出す。

### `fea-elastic` — 線形弾性の変位場

<!-- keys:data/fea-elastic -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `n` | `2060` | 10 以上の整数 | 件数（1 件 FEM 1 回） |
| `size` | `64` | 8 以上 256 以下の整数 | 正方形の一辺の要素数 |
| `material` | `"full"` | "full" のどれか | 材料の領域（いまは全面だけ） |
| `min_support_distance` | `16` | 2 以上 256 以下の整数 | 2 本の線拘束の最小間隔（要素数。size 以下）。近いとほぼ自由に回転して変位が極端になる |
| `n_test` | `60` | 0 以上の整数 | テストの件数（学習に使わない。評価はこれで行う） |
| `n_val` | `40` | 0 以上の整数 | 検証の件数（毎エポックの検証の損失。最良のエポックの選択に使う） |
| `seed` | `0` | 0 以上の整数 | 乱数の種（同じ種なら同じデータ・同じテスト件） |
<!-- /keys -->

拘束: 2〜3 節点の水平・垂直の線 2 本（x・y 固定）。荷重: 2〜3 節点の線 1 本（合計 1、向きランダム、拘束と重ならない）。
入力（節点の格子）: `material`・`fix_x`・`fix_y`・`load_x`・`load_y`。出力: `ux`・`uy`。

### `topodiff` — TopoDiff データセットの取り込み

<!-- keys:data/topodiff -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `root` | （必須） | 文字列 | dataset_1_diff のフォルダ（仕様ファイルからの相対か絶対） |
| `limit` | `0` | 0 以上の整数 | 学習データの先頭から使う件数（0 = 全部、約 3 万件） |
| `levels` | `[1, 2]` | [1, 2] から選んだリスト（[] も可） | 取り込むテスト（level_<k>.npz。正解の形は無く、SIMP のコンプライアンスだけ） |
| `topodiff_repo` | `""` | 文字列 | github.com/francoismaze/topodiff のクローン（テストを TopoDiff の評価で測るとき必須） |
| `n_test` | `300` | 0 以上の整数 | テストの件数（学習に使わない。評価はこれで行う） |
| `n_val` | `300` | 0 以上の整数 | 検証の件数（毎エポックの検証の損失。最良のエポックの選択に使う） |
| `seed` | `0` | 0 以上の整数 | 乱数の種（同じ種なら同じデータ・同じテスト件） |
<!-- /keys -->

入力: `volfrac`・`sed`・`von_mises`・`load_x`・`load_y`（TopoDiff と同じ値をそのまま）。出力: `density`。

## モデル仕様（`train --spec model.json --data <データ>`）

最上段の鍵:

<!-- keys:spec -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `model` | {}（FNO の既定） | 鍵と値の組（{…}） | モデルの種類と大きさ（model.type とその鍵） |
| `inputs` | 全部 | 名前のリスト | データの入力チャネルのうち使うもの。生成モデルでは生成の条件になる |
| `loss` | 密度は bce、場は rel_l2 | "bce" / "mse" / "rel_l2" のどれか | 損失（FNO だけ）: bce（密度）・rel_l2（場の既定。件ごとの相対 L2）・mse |
| `train` | {} | 鍵と値の組（{…}） | 学習の設定 |
| `sample` | {} | 鍵と値の組（{…}） | 生成モデルだけ: evaluate / predict の既定の案の数など |
<!-- /keys -->

`model.type` でモデルの種類を選ぶ（省略で `fno`）。種類ごとに `model` の鍵が違い、ほかの種類の鍵を書くと差し戻す。
種類ごとの鍵と、全部の鍵を既定値で埋めた例は `spec model --type <種類>` でも返る。

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

<!-- keys:model/fno -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `type` | `"fno"` | "fno" / "diffusion" / "flow" / "vae" / "gan" のどれか | モデルの種類 |
| `latent_channels` | `64` | 1 以上の整数 | FNO の内部のチャネル数 |
| `num_fno_layers` | `4` | 1 以上の整数 | FNO の層の数 |
| `num_fno_modes` | `16` | 1 以上の整数 | 残すフーリエモードの数（格子の半分以下が目安） |
| `padding` | `8` | 0 以上の整数 | 周期でない境界のための詰め物（セル数） |
| `decoder_layers` | `1` | 1 以上の整数 | 出力側の全結合層の数 |
| `decoder_layer_size` | `64` | 1 以上の整数 | 出力側の全結合層の幅 |
| `coord_features` | `true` | true か false | 座標 2 チャネルを入力に足す（PhysicsNeMo の既定） |
<!-- /keys -->

- `loss`: `bce`（密度。出力に sigmoid）・`rel_l2`（場の既定。件ごと・量ごとの相対 L2 誤差の平均。FNO の論文と同じで、
  値の大きい件に引きずられない）・`mse`。省略でデータの種類から決まる

### `diffusion` — 条件付き拡散モデル

<!-- keys:model/diffusion -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `type` | `"fno"` | "fno" / "diffusion" / "flow" / "vae" / "gan" のどれか | モデルの種類 |
| `model_channels` | `64` | 4 以上の整数 | U-Net の基本のチャネル数（4 の倍数） |
| `channel_mult` | `[1, 2, 2, 2]` | 1〜16 の整数のリスト（1〜6 個） | 段ごとのチャネル倍率。段の数 = 長さ。1 段下るごとに格子が半分になる |
| `num_blocks` | `2` | 1 以上 8 以下の整数 | 段ごとの残差ブロック数 |
| `attn_levels` | `[2]` | 0〜5 の整数のリスト（0〜6 個） | 自己注意を入れる段（0 = 元の解像度、1 = 1/2、…）。[] で無し。最下段は常に入る |
| `dropout` | `0.1` | 0 以上 1 未満の数 | ドロップアウトの割合 |
| `timesteps` | `1000` | 1 以上 10000 以下の整数 | 拡散の段数（学習） |
| `schedule` | `"cosine"` | "cosine" / "linear" のどれか | ノイズの計画: cosine（Nichol & Dhariwal）・linear（Ho ら） |
<!-- /keys -->

`sample`（`evaluate` / `predict` の既定。コマンド行の `--samples` / `--steps` で上書きできる）:

<!-- keys:sample/diffusion -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `samples` | `4` | 1 以上 64 以下の整数 | 1 条件あたりの案の数 |
| `steps` | `50` | 1 以上 1000 以下の整数 | 生成の段数（少ないほど速いが粗い） |
| `eta` | `0.0` | 0 以上 1 以下の数 | 0 で DDIM（ノイズを足さない）、1 で DDPM に近い |
| `seed` | `0` | 0 以上の整数 | ノイズの種（同じ種なら同じ案） |
<!-- /keys -->

正解（密度・場）にノイズを足していき、ノイズから正解へ戻す向きを、入力チャネルを条件として学ぶ。
生成はノイズから始めるので、同じ条件でもノイズの引き方ごとに違う案になる（TopoDiff と同じ考え方）。

- attention は 64 チャネルに 1 つの頭（head）なので、入れる段のチャネル数（`model_channels × channel_mult`）は 64 以上。
  U-Net の最下段は常に attention を持つので（4 つとも）、`model_channels × channel_mult[-1]` も 64 以上
- 格子は U-Net の段数に合わせて右・下に 0 を埋めて広げ（例: 65×65 → 72×72）、損失は元の領域だけで測る。
  生成した案は元の大きさに切り戻す
- 密度は [0, 1] → [-1, 1] に移して学び、生成の途中の推定も [-1, 1] に収める。場は学習データの平均・標準偏差で正規化する
- `loss` は書かない（生成モデルはそれぞれの損失を持つ。書くと差し戻す）
- 拡散の U-Net は CPU では非常に遅い（Apple シリコンの Mac で測ると、CPU は MPS の約 100 倍の時間がかかった）
- 検証の損失は毎エポック同じ段・同じノイズで測るので、エポックどうしを比べられる
- `sample.steps` は `model.timesteps` 以下

### `flow` — Flow matching

<!-- keys:model/flow -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `type` | `"fno"` | "fno" / "diffusion" / "flow" / "vae" / "gan" のどれか | モデルの種類 |
| `model_channels` | `64` | 4 以上の整数 | U-Net の基本のチャネル数（4 の倍数） |
| `channel_mult` | `[1, 2, 2, 2]` | 1〜16 の整数のリスト（1〜6 個） | 段ごとのチャネル倍率。段の数 = 長さ。1 段下るごとに格子が半分になる |
| `num_blocks` | `2` | 1 以上 8 以下の整数 | 段ごとの残差ブロック数 |
| `attn_levels` | `[2]` | 0〜5 の整数のリスト（0〜6 個） | 自己注意を入れる段（0 = 元の解像度、1 = 1/2、…）。[] で無し。最下段は常に入る |
| `dropout` | `0.1` | 0 以上 1 未満の数 | ドロップアウトの割合 |
<!-- /keys -->

<!-- keys:sample/flow -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `samples` | `4` | 1 以上 64 以下の整数 | 1 条件あたりの案の数 |
| `steps` | `20` | 1 以上 1000 以下の整数 | 生成の段数（少ないほど速いが粗い） |
| `seed` | `0` | 0 以上の整数 | ノイズの種（同じ種なら同じ案） |
<!-- /keys -->

ノイズ（t = 0）と正解（t = 1）を直線で結び、その上の点で「正解 − ノイズ」の向き（速度）を当てるよう学ぶ
（Lipman ら 2023、Liu ら 2023）。生成はノイズから `steps` 回の Euler 法で進める。拡散より少ない段数で済むのが狙い。

### `vae` — 条件付き VAE

<!-- keys:model/vae -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `type` | `"fno"` | "fno" / "diffusion" / "flow" / "vae" / "gan" のどれか | モデルの種類 |
| `model_channels` | `64` | 4 以上の整数 | U-Net の基本のチャネル数（4 の倍数） |
| `channel_mult` | `[1, 2, 2, 2]` | 1〜16 の整数のリスト（1〜6 個） | 段ごとのチャネル倍率。段の数 = 長さ。1 段下るごとに格子が半分になる |
| `num_blocks` | `2` | 1 以上 8 以下の整数 | 段ごとの残差ブロック数 |
| `attn_levels` | `[2]` | 0〜5 の整数のリスト（0〜6 個） | 自己注意を入れる段（0 = 元の解像度、1 = 1/2、…）。[] で無し。最下段は常に入る |
| `dropout` | `0.1` | 0 以上 1 未満の数 | ドロップアウトの割合 |
| `latent_dim` | `32` | 1 以上 4096 以下の整数 | 潜在ベクトルの次元 |
| `kl_weight` | `0.01` | 0 以上の数 | KL の重み。大きいほど案を引きやすいが形がぼやける |
<!-- /keys -->

<!-- keys:sample/vae -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `samples` | `4` | 1 以上 64 以下の整数 | 1 条件あたりの案の数 |
| `seed` | `0` | 0 以上の整数 | ノイズの種（同じ種なら同じ案） |
<!-- /keys -->

学習では、正解と条件から潜在ベクトル z を推定する encoder と、条件と z から形を作る decoder（U-Net）を一緒に学ぶ。
生成では z を標準正規分布から引き、decoder を 1 回通す。保存するのは decoder だけ。
損失 = 再構成の二乗誤差（セル平均）+ `kl_weight` × KL ÷ セル数。

### `gan` — 条件付き GAN

<!-- keys:model/gan -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `type` | `"fno"` | "fno" / "diffusion" / "flow" / "vae" / "gan" のどれか | モデルの種類 |
| `model_channels` | `64` | 4 以上の整数 | U-Net の基本のチャネル数（4 の倍数） |
| `channel_mult` | `[1, 2, 2, 2]` | 1〜16 の整数のリスト（1〜6 個） | 段ごとのチャネル倍率。段の数 = 長さ。1 段下るごとに格子が半分になる |
| `num_blocks` | `2` | 1 以上 8 以下の整数 | 段ごとの残差ブロック数 |
| `attn_levels` | `[2]` | 0〜5 の整数のリスト（0〜6 個） | 自己注意を入れる段（0 = 元の解像度、1 = 1/2、…）。[] で無し。最下段は常に入る |
| `dropout` | `0.1` | 0 以上 1 未満の数 | ドロップアウトの割合 |
| `latent_dim` | `64` | 1 以上 4096 以下の整数 | 潜在ベクトルの次元 |
| `l1_weight` | `10.0` | 0 以上の数 | 正解との L1 誤差の重み。大きいほど正解に近いが、同じ形になりやすい |
| `diversity_weight` | `1.0` | 0 以上の数 | 2 つの潜在ベクトルから違う形を出させる項の重み（0 で無し） |
| `disc_channels` | `64` | 1 以上の整数 | 識別器の最初の層のチャネル数 |
<!-- /keys -->

<!-- keys:sample/gan -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `samples` | `4` | 1 以上 64 以下の整数 | 1 条件あたりの案の数 |
| `seed` | `0` | 0 以上の整数 | ノイズの種（同じ種なら同じ案） |
<!-- /keys -->

生成器（条件と z から形を作る U-Net）と識別器（条件と形の組が本物かを小領域ごとに見る PatchGAN、
spectral normalization 付き）を交互に学ぶ。生成は z を引いて生成器を 1 回通す。保存するのは生成器だけ。
`diversity_weight` の項は Yang ら 2019（差は 1 で頭打ち）。

- GAN の損失はエポックの良し悪しを表さないので、**最後のエポック**を保存する。`history.csv` の検証の損失は
  決まったノイズで生成した形の正解との L1 誤差（参考）。学習と検証で量が違うので `overfit_ratio` は出さない
- 学習が不安定になったら `lr` を下げる。似た形しか出ない（`diversity` が 0 に近い）なら `diversity_weight` を上げるか
  `l1_weight` を下げる

### `train`（全種類で共通。既定は FNO の値で、生成モデルは `epochs` 200・`lr` 0.0002・`weight_decay` 0・`ema` 0.999）

<!-- keys:train -->
| キー | 既定 | 値 | 意味 |
|---|---|---|---|
| `epochs` | `40` | 1 以上 100000 以下の整数 | 学習の回数（データ全体を何周するか） |
| `batch` | `32` | 1 以上 4096 以下の整数 | 1 回の更新に使う件数 |
| `lr` | `0.001` | 0 より大きく 1 以下の数 | 学習率（OneCycle の最大値） |
| `weight_decay` | `0.0001` | 0 以上の数 | 重みの減衰（AdamW） |
| `n_train` | `0` | 0 以上の整数 | 学習に使う件数（0 = 全部）。データ量の効果を確かめるとき |
| `seed` | `0` | 0 以上の整数 | 乱数の種 |
| `device` | `"auto"` | "auto" / "cpu" / "cuda" / "mps" のどれか | 計算する装置。auto は CUDA → Apple の GPU（MPS）→ CPU の順 |
| `ema` | `0.0` | 0 以上 1 未満の数 | 重みの移動平均（0 = 無し）。検証と保存は平均した重みで行う |
| `physics` | `0.0` | 0 以上の数 | FEM のつり合いの式の重み（0 = 無し）。FNO と節点の変位のデータ（simp-fea の target: displacement）だけ |
<!-- /keys -->

- 学習率は OneCycle。検証の損失が最小のエポックを `model.mdlus` に保存する（GAN は最後のエポック）
- `ema` は重みの移動平均で、検証と保存は平均した重みで行う
- `device: auto` は CUDA → Apple の GPU（MPS）→ CPU の順。FNO も MPS で CPU と同じ結果（差 1e-7 程度）になり、約 20 倍速い
- `physics`: 損失 = データの損失 + `physics` × ‖K(ρ)u − F‖ ÷ ‖F‖（拘束していない自由度で）。K(ρ)u は行列を作らずに
  要素ごとに足し合わせて求め（`physics.py`）、微分できる。検証の損失はデータの損失のままなので、重みを変えた学習どうしを
  比べられる。`history.csv` に `train_residual` の列が増える

## 生成モデルの評価（`evaluate`）

1 条件から K 案（`--samples`、既定は `sample.samples`）を作る。

- いつもの数値（IoU・MAE・体積の誤差・相対 L2・コンプライアンス比・TopoDiff の評価指標）は **1 案目**で出す。
  1 回引いた案の性能なので、FNO とそのまま比べられる
- `generative.diversity`: K 案どうしの違い（全組の平均）。密度は材料（0.5 以上）の判定が食い違うセルの割合、
  場は 2 案の差の相対 L2。0 なら毎回同じ形しか出ない
- `generative.best_of_k`: K 案の中で最も良い案の数値。IoU（正解に最も近い案）、相対 L2、`simp-topopt` は
  FEM で解いたコンプライアンスで選んだ案、TopoDiff のテストは CE が最も小さい案。FEM で候補を選ぶ使い方の上限の目安
- `generative.compliance_ratio_all`: 全案のコンプライアンス比（破綻する案の割合を見る）
- `overhang`（`optopus-overhang`）: 造れない材料（上の `optopus-overhang` の節）
- `floating`（`simp-topopt`・`optopus-overhang`。FNO も）: 浮いた材料。材料（密度 0.5 以上）のうち、要素の辺を通って拘束に触れる要素まで
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
