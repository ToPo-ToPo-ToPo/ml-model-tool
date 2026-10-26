# 引き継ぎ: オーバーハングを直す生成モデル（JuliaOptopus の self で教師データを作る）

2026-10-20 時点。別の PC でデータ作成から続けるための資料。

## 1. 目的

トポロジー最適化の結果を入力すると、積層造形でサポート無しに造れる（自己支持。張り出しは造形方向から 45° まで）
形を何案も作る生成モデルを作る。

```
入力: 普通のトポロジー最適化の結果 ＋ 条件（荷重・拘束・体積率・造形の向き）
  ↓ 生成モデル（diffusion / flow / vae / gan。ml-model-tool に既にある）
出力: オーバーハングの無い形（何案でも）
```

教師データの組は JuliaOptopus で作る。1 件ごとに同じ条件で 2 回解く。

| | JuliaOptopus のモード | データでの役割 |
|---|---|---|
| 入力 | `free`（AM の制約なし） | x の `design` チャネル |
| 正解 | `self`（Langelaar のサポート共最適化で、サポートを禁止 = 自己支持だけ） | y の `density` |

## 2. 現状

| 項目 | 状態 |
|---|---|
| コード（2 つのリポジトリ） | できている。テスト 95 件すべて通過（本物の JuliaOptopus を使う試験を含む。2026-10-20） |
| データ作成 | 600・2000・5000 件が済み（2026-10-20。14 コアの Mac で `workers` 12）。600 件は 7.5 分、2000 件・5000 件は前の件を使い回して新しい件だけを解いた（1400 件で 15.7 分、3000 件で 38 分）。データはその PC の `~/overhang_work/jobs/data1`・`data2`・`data3`（リポジトリには入れていない）。結果は「7.」 |
| 学習・評価 | 拡散モデルで済み（2026-10-20）。5000 件で学習すると、オーバーハングのある案 18%（材料の 0.30%）、コンプライアンス比 1.024（1.05 以下が 78%）。結果と次の案は「7.」「8.」 |
| 計算が遅い問題 | 解決（2026-10-20）。下の「4.1」 |

## 3. ブランチ

どちらも **main に取り込み済み**（2026-10-20）。別の PC では main をそのまま使う。
元のブランチで既に clone してある場合は、それぞれのフォルダで `git checkout main` と `git pull` をする。

| リポジトリ | 取り込んだ PR | 元のブランチ |
|---|---|---|
| github.com/ToPo-ToPo-ToPo/ml-model-tool | [#4](https://github.com/ToPo-ToPo-ToPo/ml-model-tool/pull/4) | `claude/topology-optimization-generative-model-6d790b` |
| github.com/ToPo-ToPo-ToPo/JuliaOptopus | [#150](https://github.com/ToPo-ToPo-ToPo/JuliaOptopus/pull/150) | `claude/ml-overhang-data` |

### ml-model-tool の変更

| ファイル | 内容 |
|---|---|
| `src/ml_model_tool/sources/optopus_overhang.py` | 新しいデータ源 `optopus-overhang`。条件を作り、JuliaOptopus を並列に起動して free / self を解かせ、データセットにまとめる |
| `src/ml_model_tool/overhang.py` | 造れるかの判定（`unprintable` / `overhang_volume`）と造形の向きの扱い |
| `src/ml_model_tool/evaluation.py`・`plots.py`・`src/ml_model_tool_cli/jobs.py` | 評価に `overhang`（造れない材料の割合）を追加。図に入力の形とベースプレート（緑）を表示 |
| `skills/ml-model/examples/data_optopus_overhang.json`・`model_overhang_diffusion.json` | 仕様の例 |
| `tests/test_overhang.py` | テスト。環境変数 `JULIA_OPTOPUS` があると本物の JuliaOptopus でも試す |
| `docs/specs.md`・`skills/ml-model/SKILL*.md`・`README.md` | 使い方 |

### JuliaOptopus の変更

| ファイル | 内容 |
|---|---|
| `lib/JuliaOptopusAMBenchmark/src/support_cone.jl`（新規） | self / free の計算本体（問題の構築・評価の連鎖・MMA のループ・浮きの判定）を `langelaar_support_cone.jl` から移した。main のドライバどうしは include できない規約（R4）のため。変えたのは、境界条件を差し替える口 `bc_builder` を足したことなど 3 点だけ |
| `main/linear_elasticity/TO/am_overhang/langelaar_support_cone.jl` | 移した関数をライブラリから読むように変更。**移す前と同じ結果**（self を 30×15 で回して C・V・灰色率が全桁一致）、`mode=check` の感度の数値微分照合も通過 |
| `main/linear_elasticity/TO/am_overhang/ml_dataset.jl`（新規） | 教師データ作成のドライバ。条件の npz を読み、1 件ごとに free と self を解き、結果を npz に書く |
| `main/INDEX.md` | `tools/main_index.jl` で再生成（main で既にずれていた他の行も更新された） |

## 4. 既知の問題（4.1・4.2 は解決済み）

### 4.1 計算がとても遅い（2026-10-20 解決）

原因は並列の取り合い。JuliaOptopus は疎行列の直接法（UMFPACK）を呼ぶ区間だけ BLAS のスレッドを全コア
（`Sys.CPU_THREADS`）へ上げる（`lib/JuliaOptopusCore/src/LinearAlgebraKit/parallel_blas.jl` の
`with_parallel_blas`）。`OPENBLAS_NUM_THREADS=1` はこれで上書きされるので、16 プロセス同時だと
16 × 全コアのスレッドが取り合っていた。止めるには環境変数 `JULIAOPTOPUS_BLAS_THREADS=1` を渡す。
`optopus_overhang.py` の `run_optopus` で渡すように直した。結果は変わらない（C が全桁一致）。

実測（2026-10-20、Apple シリコン Mac・14 コア、64×32・反復 100、条件の最初の 3 件）:

| 条件 | 1 件（free + self）の時間 |
|---|---|
| 1 プロセス・既定 | 4.4〜9.2 秒 |
| 1 プロセス・`JULIAOPTOPUS_BLAS_THREADS=1` | 3.8〜8.4 秒 |
| 8 プロセス同時・既定（今までの設定） | **約 360 秒**（1 件目） |
| 8 プロセス同時・`JULIAOPTOPUS_BLAS_THREADS=1` | 4.8〜10.2 秒 |
| 直した後の `mmt data`（`workers` 8・16 件） | 3.9〜14.3 秒。全体で 36.5 秒（コンパイル込み） |

既定のままだと、各プロセスのスレッドが 4 本から 22 本に増え、そのうち 10 本ほどが同時に動いていた。
600 件なら、8 プロセスで 12 分前後の見込み。

参考: 収束までの反復は free が 50 回前後、self は 100 回（上限）に届くことが多い。

### 4.2 途中で止めると結果が全部消える（2026-10-20 解決）

`ml_dataset.jl` に `outdir=` を足し、1 件解くごとに `outdir/case_<番号>.npz` を書くようにした（一時ファイルに
書いてから名前を変えるので、書いている途中で止めても壊れたファイルは残らない）。既にある件は飛ばす。
`optopus_overhang.py` の `run_optopus` は `--out` の下の `optopus/cases/` を見て、まだ無い件だけを各プロセスに
1 件おきに配り、最後に全件を集める。

- 止めるのは Ctrl-C でも `kill` でもよい（どちらも Julia のプロセスまで止まる）。同じ仕様・同じ `--out` で
  もう一度 `mmt data` を走らせれば続きから解く（試験: 24 件のうち 6 件で止め、再実行で残りの 18 件だけを解いた）
- `seed`・格子・`iters`・`radius` などを変えると、保存済みの件を消して最初から解く。件数 `n` を増やすだけなら、
  前に解いた件（件 i の条件は seed と i だけで決まる）はそのまま使い、新しい件だけを解く。別の `--out` に
  増やすときは、前の `optopus/cases/` と `optopus/cases.npz` を写しておけば使い回せる（2000 件はこうして作った）

### 4.3 その他

- 正解（self）の密度はわずかに 1 を超えることがある。Python 側で [0, 1] に切るので、Python の FEM と
  JuliaOptopus のコンプライアンスが 0.5% ほどずれる（free は一致）
- 反復が少ないと、正解にも 0.5 付近の密度の要素が残り、わずかにオーバーハングと判定されることがある（30 反復の試験で
  材料の 0.2%）
- データは要素設計（`design=element`）で作っている。JuliaOptopus の self の推奨設定は節点設計だが、学習の格子と
  評価の FEM（要素の格子）に合わせるため要素にした。ドライバは `design=node` も受け付ける（Python 側は未対応）

## 5. 別の PC での準備

```bash
git clone https://github.com/ToPo-ToPo-ToPo/ml-model-tool.git
cd ml-model-tool
uv sync
cd ..
```

```bash
git clone https://github.com/ToPo-ToPo-ToPo/JuliaOptopus.git
cd JuliaOptopus
```

Julia は **1.10.11 固定**（JuliaOptopus の README。1.11 以降は不可）。`julia` コマンドが 1.10.11 になるようにする。

```bash
juliaup add 1.10.11
juliaup default 1.10.11
```

JuliaOptopus のフォルダで環境を作り、一度読み込んでコンパイルしておく（数分〜十数分）。

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); using JuliaOptopus, JuliaOptopusAMBenchmark'
```

ml-model-tool に JuliaOptopus の場所を教える（データ仕様の `julia_optopus` に書いてもよい）。

```bash
export JULIA_OPTOPUS=/path/to/JuliaOptopus
```

動作確認（ml-model-tool のフォルダで。1 分ほど。68 件が通れば準備完了）:

```bash
uv run pytest -q
```

## 6. 作業手順

作業フォルダを作って、その中で進める（以下 ml-model-tool のフォルダを `$MMT` とする）。

```bash
mkdir -p ~/overhang_work && cd ~/overhang_work
```

### 6.1 まず速度を測る（4.1）

`speed.json`:

```json
{"source": "optopus-overhang", "n": 10, "n_test": 2, "n_val": 2, "workers": 1, "iters": 100}
```

```bash
uv run --project $MMT mmt data --spec speed.json --out "$PWD/jobs/speed1"
```

1 件ごとの時間は `jobs/speed1/optopus/optopus_0.log` の `[ml_dataset] case ...  xxx s` の行に出る。
`workers` を 4・8 と増やして、1 件の時間がどう変わるかも見る。

### 6.2 データを作る（600 件）

`$MMT/skills/ml-model/examples/data_optopus_overhang.json` を写して `workers` を PC のコア数に合わせる。

```bash
cp $MMT/skills/ml-model/examples/data_optopus_overhang.json data.json
```

```bash
uv run --project $MMT mmt data --spec data.json --out "$PWD/jobs/data1"
```

- 主なキー: `n`（件数）・`nelx` / `nely`（64×32）・`build_directions`（既定 `["up"]`）・`iters`（100）・
  `radius`（1.5）・`workers`。全部のキーは `$MMT/docs/specs.md` の `optopus-overhang` の節
- 進み具合は `jobs/data1/status.json`、Julia のログは `jobs/data1/optopus/optopus_<k>.log`、解いた件は
  `jobs/data1/optopus/cases/case_<番号>.npz`
- 途中で止めても、同じコマンドをもう一度走らせれば続きから解く（4.2）
- できたら `jobs/data1/summary.md` を見る。確かめる値: `overhang_target_mean`（正解のオーバーハング。ほぼ 0）・
  `overhang_input_mean`（入力のオーバーハング）・`compliance_fem_vs_optopus_max_rel_diff`（1% 未満のはず）

### 6.3 学習・評価

```bash
cp $MMT/skills/ml-model/examples/model_overhang_diffusion.json model.json
```

```bash
uv run --project $MMT mmt train --spec model.json --data jobs/data1 --out "$PWD/jobs/diff1"
```

```bash
uv run --project $MMT mmt evaluate --model jobs/diff1 --data jobs/data1 --samples 4 --out "$PWD/jobs/eval1"
```

`jobs/eval1/summary.md` と `compare.png` を見る。主な指標:

| 指標 | 意味 | ほしい値 |
|---|---|---|
| `overhang.share` | 造れない材料を含む案の割合 | ほぼ 0 |
| `overhang_input.share` | 同じものを入力の形で測った値（比べる基準） | — |
| `compliance_ratio.median` | 生成した案のコンプライアンス ÷ 正解（self） | 1 に近い |
| `floating.share` | 浮いた材料を含む案の割合 | ほぼ 0 |
| `generative.best_of_k` | K 案から FEM で最良を選んだときの値 | — |

## 7. これまでの結果（参考）

### JuliaOptopus のデータ（600 件、64×32、2026-10-20）

`skills/ml-model/examples/data_optopus_overhang.json` の `workers` を 12 にしただけの仕様。1 件 3.1〜21.3 秒
（中央値 7.1 秒）、全体で 7.5 分。`summary.md` の値:

| 値 | 結果 | 意味 |
|---|---|---|
| `overhang_input_mean` / `_share` | 0.247 / 95.5% | 入力（free）の造れない材料の割合の平均 / それがある件の割合 |
| `overhang_target_mean` / `_share` | 0.0000 / 3.3% | 正解（self）の同じ値。少しでも残る件が 20 件あるが量はほぼ 0（4.3） |
| `compliance_target_over_input_median` | 1.02 | 自己支持にして増えるコンプライアンス |
| `gray_target_median` | 0.151 | 正解の灰色率 |
| `optopus_floating_target_max` | 0.002 | 正解の浮き（JuliaOptopus の判定）の最大 |
| `compliance_fem_vs_optopus_max_rel_diff` | 0.0058 | Python の FEM と JuliaOptopus の差（1% 未満） |
| 支え方 | cantilever 189・both-ends 203・pin-roller 208 | |

### 拡散モデル: 600・2000・5000 件の比較（2026-10-20）

データを 5000 件（訓練 4250・検証 250・テスト 500）に増やし、同じ仕様で学習した。データは 2000 件を使い回して
新しい 3000 件だけを解いた（38 分）。学習は 113 分（1 エポック約 34 秒）、検証の損失の最小はエポック 132（0.0136。
2000 件は 0.0177）。結果は `~/overhang_work/jobs/data3`・`diff3`。

比べ方: **5000 件のテストのうち番号 2000 以降の 291 件**（3 つのモデルのどれも見ていない。`jobs/test_common3`）で
3 つとも評価した（`jobs/eval_diff{1,2,3}_common3`）。1 条件 4 案。

| 指標（共通のテスト 291 件） | 入力の形 | 600 件 | 2000 件 | 5000 件 |
|---|---|---|---|---|
| オーバーハングがある案の割合（1 案目） | 95% | 53% | 28% | **18%** |
| 　うち材料の 1% を超える案 | — | 31% | 11% | **5%** |
| 造れない材料の割合（平均） | 25.0% | 2.6% | 0.46% | **0.30%** |
| 浮いた材料がある案の割合 | — | 50% | 39% | 35% |
| 　うち材料の 1% を超える案 | — | 25% | 12% | 11% |
| IoU（正解と。中央値） | 0.71 | 0.70 | 0.77 | **0.78** |
| 体積の誤差（中央値） | — | 0.048 | 0.013 | **0.006** |
| コンプライアンス比（中央値） | 0.98 | 1.21 | 1.07 | **1.024** |
| 　1.05 以下の案 / 2 倍を超える案 | — | 9% / 12% | 28% / 3% | **78% / 1%** |
| 4 案から FEM で最良を選んだコンプライアンス比 | — | 1.10 | 1.03 | **1.007**（1.05 以下 93%） |
| 4 案すべてのオーバーハングがある割合 | — | 54% | 30% | **17%** |
| 案どうしの違い（中央値） | — | 0.16 | 0.11 | 0.09 |

- 5000 件で、特にコンプライアンスが良くなった（1.05 以下の案が 28% → 78%）。オーバーハングが材料の 1% を超える案も
  11% → 5%。図（`jobs/eval_diff3_common3/compare.png`）でも、ほとんどの案が正解とほぼ同じ形になっている
- 伸びが鈍ってきたもの: 浮いた材料（1% を超える案 12% → 11%）と IoU（0.77 → 0.78）。データを増やすだけでは
  減りにくいので、浮いた小さなかけらは後処理で消すのが早そう
- 案どうしの違いが小さくなっている（0.11 → 0.09）。正解に近づいた分、似た案が多い。いろいろな案がほしいなら
  サンプリングの設定（DDIM の eta など）を見直す
- 2000 件モデルの値が下の「600 件と 2000 件の比較」とわずかに違うのは、テストの件が違うため（148 件 → 291 件）

### 拡散モデル: 600 件と 2000 件の比較（2026-10-20）

同じ仕様（`model_overhang_diffusion.json`。200 エポック）で、データだけを 600 件 → 2000 件（訓練 1700・検証 100・
テスト 200）にした。2000 件は 600 件の結果を使い回し、新しい 1400 件だけを解いた（15.7 分）。学習は 45.9 分。

比べ方: 2000 件のテストには 600 件モデルの訓練に使った件が混ざるので、**2000 件のテストのうち番号 600 以降の
148 件**（どちらのモデルも見ていない件。`~/overhang_work/jobs/test_common`）で両方を評価した。1 条件 4 案。

| 指標（共通のテスト 148 件） | 入力の形 | 600 件モデル | 2000 件モデル |
|---|---|---|---|
| オーバーハングがある案の割合（1 案目） | 97% | 48% | **20%** |
| 　うち材料の 1% を超える案 | — | 27% | **3%** |
| 造れない材料の割合（平均） | 27.6% | 2.6% | **0.25%** |
| 浮いた材料がある案の割合 | — | 48% | 34% |
| 　うち材料の 1% を超える案 | — | 19% | **7%** |
| IoU（正解と。中央値） | 0.67 | 0.69 | **0.77** |
| 体積の誤差（中央値） | — | 0.044 | 0.014 |
| コンプライアンス比（中央値） | 0.97 | 1.21（2 倍超 12%） | **1.07（2 倍超 2%）** |
| 4 案から FEM で最良を選んだコンプライアンス比 | — | 1.07（1.05 以下 38%） | **1.03（1.05 以下 79%）** |
| 4 案すべてのオーバーハングがある割合 | — | 51% | 26% |
| 案どうしの違い（中央値） | — | 0.19 | 0.13 |

- どの指標もはっきり良くなった。データの量が効いている。図（`jobs/eval2_common/compare.png`）でも、600 件で
  目立ったごま塩状のノイズとぼやけがほぼ消え、正解と同じ骨組みになっている
- 学習: 検証の損失の最小はエポック 135（0.0177）。その後わずかに上がる（訓練 0.0115 / 検証 0.0195）が、
  600 件（エポック 111 から上昇）より遅い
- 残る課題: 浮いた材料が少しある案が 34%（ほとんどは材料の 1% 以下の小さなかけら）。ときどき条件に合わない形が
  出る（コンプライアンスが桁違いの案。最悪 2.39 倍 / 4 案のうちでは 240 倍もある）
- 2000 件モデルを 2000 件のテスト 200 件で測った値もほぼ同じ（オーバーハング 22%、コンプライアンス比 1.07、
  FEM で最良 1.03。`jobs/eval2`）

### 拡散モデル（上の 600 件で学習、2026-10-20）

`skills/ml-model/examples/model_overhang_diffusion.json` のまま（200 エポック・batch 32・lr 2e-4・EMA 0.999）。
Apple の GPU（MPS）で学習 13.9 分、評価 1 分。結果は `~/overhang_work/jobs/diff1`・`jobs/eval1`（その PC だけ）。

- 検証の損失はエポック 111 で最小（0.0175）、その後は上がって 200 で 0.028（過学習。訓練は 0.014）。
  保存されるのは最小のエポック（111）のモデル
- テスト 60 件・1 条件 4 案。生成の段数を 50 → 200 にしても値はほぼ同じ（`jobs/eval1_s200`）。案に残る
  ごま塩状のノイズは段数のせいではない

| 指標 | 入力の形 | 生成した案（1 案目） | 4 案すべて | 参考: Python 版（7 の下） |
|---|---|---|---|---|
| オーバーハングがある案の割合 | 93% | 47% | 51% | 38% |
| 造れない材料の割合（平均） | 26.5% | 3.8% | — | 2.5% |
| 浮いた材料がある案の割合 | — | 42% | 45% | — |
| IoU（正解と。中央値） | 0.72 | 0.67（4 案の最良 0.79） | — | 0.84 |
| コンプライアンス比（÷ 正解。中央値） | 0.97 | 1.24（2 倍超 23%） | 4 案から FEM で最良: 1.09（2 倍超 0%） | 1.14 / 最良 1.06 |

読み方:
- オーバーハングは減る（材料の 26.5% → 3.8%）が、残る案が半分ある。浮いた材料も 4 割の案に出る
- 1 案目の IoU（0.67）は、入力をそのまま写した場合（0.72）より低い。正解（self）は入力（free）を少し直した
  形ではなく、別の最適解になっていることが多い（入力と正解の IoU は最小 0.28）。Python 版より難しい組
- 図（`compare.png`）では、ごま塩状のノイズが残る案・ぼやけた案・条件に合わない形（コンプライアンスが桁違い）が
  混ざる。訓練 510 件に対してモデル（1070 万パラメータ）が大きく、エポック 111 以降は過学習している

### JuliaOptopus のデータ（止めるまでの 33 件、64×32、2026-10-19）

- 正解（self）の浮き（造れない材料）は 33 件すべて 0
- self ÷ free のコンプライアンス: 0.995〜1.19、中央値 約 1.03（自己支持にしても剛性はあまり落ちない）

### Python の簡易 SIMP で作ったデータ（2026-10-19。今は使っていない）

JuliaOptopus に切り替える前に、Python で書いた AM フィルタ付き SIMP（このブランチの途中の版。今は削除済み）で
600 件を作り（約 6 分）、拡散モデルを 100 エポック学習した結果。

| 指標 | 入力の形 | 生成した案（1 案目） |
|---|---|---|
| オーバーハングがある案の割合 | 95% | 38% |
| 造れない材料の割合（平均） | 25.7% | 2.5% |

IoU 中央値 0.84、コンプライアンス比 中央値 1.14（2 倍超が 10%）、4 案から最良を選ぶと 1.06。ノイズの多い案が
混ざっていた。学習が短い（100 エポック）のが主な原因と見ているが、確かめていない。

## 8. 次にやること

1. ~~速度の原因を調べて直す（4.1）~~ 済み（2026-10-20）
2. ~~1 件ごとに保存する形に直す（4.2）~~ 済み（2026-10-20）
3. ~~600 件のデータを作る（6.2）~~ 済み（2026-10-20。7.）
4. ~~拡散モデルを 200 エポックで学習・評価する（6.3）。7 の Python 版の結果と比べる~~ 済み（2026-10-20。7.）。
   オーバーハングは減ったが、残る案が約半分で Python 版より悪い
5. 残るオーバーハングを減らす（上ほど手軽）
   - ~~データを増やす~~ 2000 件・5000 件で済み（2026-10-20）。共通のテスト 291 件で、オーバーハングのある案
     53% → 28% → 18%、コンプライアンス比 1.21 → 1.07 → 1.024。浮いた材料と IoU は伸びが鈍ってきた
   - 後処理: 生成した案を 0.5 で白黒にし、ベースプレートから支えられない材料を消す（`overhang.py` の判定を使う）。
     浮いた材料も同じく消す。**今いちばん効きそう**（データを増やしても浮いたかけらが減りにくい）
   - 左右反転でデータを倍にする（造形の向きが up なら、拘束・荷重ごと左右に反転しても同じ問題）
   - 4 案から「造れる・浮きが無い」案のうち FEM で最良を選ぶ（今の best_of_k はコンプライアンスだけで選ぶ）
