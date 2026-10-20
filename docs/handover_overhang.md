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
| 600 件のデータ作成 | **未完**。33 件で止めた（他の計算を優先するため）。結果は保存されていない |
| 学習・評価 | 未実施（JuliaOptopus のデータでは） |
| 計算が遅い問題 | 解決（2026-10-20）。下の「4.1」 |

## 3. ブランチ

| リポジトリ | ブランチ | 元 |
|---|---|---|
| github.com/ToPo-ToPo-ToPo/ml-model-tool | `claude/topology-optimization-generative-model-6d790b` | main |
| github.com/ToPo-ToPo-ToPo/JuliaOptopus | `claude/ml-overhang-data` | main（2026-10-19 の PR #149 の時点。`feat/am-overhang-constraints` は main に取り込み済み） |

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
- 条件（件数・seed など）や `iters`・`radius` を変えると、保存済みの件を消して最初から解く

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
git checkout claude/topology-optimization-generative-model-6d790b
uv sync
cd ..
```

```bash
git clone https://github.com/ToPo-ToPo-ToPo/JuliaOptopus.git
cd JuliaOptopus
git checkout claude/ml-overhang-data
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

### JuliaOptopus のデータ（止めるまでの 33 件、64×32）

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
3. 600 件のデータを作る（6.2）
4. 拡散モデルを 200 エポックで学習・評価する（6.3）。7 の Python 版の結果と比べる
5. 残るオーバーハングが多ければ、データを増やす・後処理で消すなどを検討する
