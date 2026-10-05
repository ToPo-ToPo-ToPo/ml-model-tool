---
name: ml-model
description: 物理問題の機械学習モデル（PhysicsNeMo の FNO）を作るときの手順。学習データを作る（トポロジー最適化・線形弾性 FEM・TopoDiff の取り込み）、モデル仕様で学習する、正解と比べて評価する、条件を 1 つずつ変えて詰める。道具 ml-model-tool はコマンド行で使い、文章は解釈しない
---

# 物理の機械学習モデルを作る（`mmt`）

道具は `ml-model-tool`。**文章は解釈しない**。何を学習させるか・どの条件を変えるか・結果をどう読むかは
あなたの仕事で、道具の仕事は「仕様どおりにデータを作る・学習する・評価して図と数値を出す」の 3 つ。

## 守ること

1. **1 回に変えるのは 1 つだけ。** 入力チャネル・データ量・エポック数・損失・モデルの大きさを同時に変えると、
   何が効いたのか分からなくなる。比較するときは同じデータ（同じ data ジョブ）と同じテスト件で比べる。
2. **学習の損失だけで判断しない。** 必ず `evaluate` で**テスト**を評価し、`compare.png`（正解と予測の図）を見る。
   学習の損失が下がっても、検証の損失が下がらなければ丸暗記（過学習）。`overfit_ratio`（検証 ÷ 学習の損失）が
   2 を超えたらデータを増やすか、入力・モデルを見直す。
3. **点荷重・点拘束は使わない**（2 次元弾性では特異点）。データの source はすでに線（2〜3 節点）で与える。
4. **長い処理は待たない。** `data`（SIMP は 1 件数秒）と `train`（CPU で数十分〜数時間）はジョブになる。
   `state: accepted` で返ったら、`job status --job-dir jobs/<…>` で進み具合を見る。止めるのは `job stop`。
5. **失敗は `error.stage` と文面を読んで直す。** 同じ行を打ち直さない。`spec` の誤り（知らないキー・範囲外の値）は
   仕様の JSON だけを直す。

## 場所と実行の形

道具は `app(name, line)` の 1 つ。仕様の JSON は作業フォルダに `write_file` で書き、コマンド行を `line` に渡す:

- `app("ml-model-tool", "data --spec data.json")` … データセットを作る
- `app("ml-model-tool", "train --spec model.json --data jobs/<data のジョブ>")` … 学習する
- `app("ml-model-tool", "evaluate --model jobs/<train のジョブ> --data jobs/<data のジョブ>")` … 評価する
- `app("ml-model-tool", "predict --model jobs/<train のジョブ> --data <データセット>")` … 予測する
- `app("ml-model-tool", "help train")` … 引数の意味と既定値（打つ前に確かめられる）

パスは作業フォルダからの相対パス。返事の `artifacts` に成果物のパスが並ぶので、次の行にそのまま転記する。
`summary.md` を最初に読み、図（`preview.png` / `loss.png` / `compare.png`）は `read_file` で見る。

## 手順

1. **データを作る**: `data.json` を書いて `data --spec data.json`。source は 3 つ（鍵の一覧は docs/specs.md）。
   - `simp-topopt`: 長方形（既定 64×32）のトポロジー最適化。固定 3 種（cantilever / both-ends / pin-roller）、
     境界上の線荷重 1 本、体積率の範囲。入力 6 チャネル（volfrac, fix_x, fix_y, load_x, load_y, sed）→ 密度
   - `fea-elastic`: 正方形（既定 64×64）全面材料の線形弾性。線拘束 2 本・線荷重 1 本を領域のどこにでも置く。
     入力 5 チャネル（material, fix_x, fix_y, load_x, load_y）→ 変位 ux, uy
   - `topodiff`: TopoDiff のデータセットの取り込み（人が置いた `dataset_1_diff` を `root` に書く）。
     入力は TopoDiff と同じ 5 チャネル。`levels` のテストは `level_1.npz` / `level_2.npz` になる
2. **学習する**: `model.json` を書いて `train --spec model.json --data jobs/<data>`。`{}` でも既定で走る。
   - `inputs`: データの入力チャネルの一部だけを使う（例: sed を外す）。**入力の効果を確かめる正しいやり方**
   - `train.n_train`: 学習件数を絞る（データ量の効果を確かめる）。`train.epochs`・`lr`・`batch`
   - `loss`: 密度は `bce`、場は `mse`（既定でそうなる）
3. **評価する**: `evaluate --model jobs/<train> --data jobs/<data>`（既定 `--split test`）。
   `--split train` も評価すると、学習データとテストの差（丸暗記の度合い）が分かる。
   - 密度: IoU（0.5 で二値化した材料の重なり）・MAE・体積の誤差
   - `simp-topopt` はさらに、予測した形を FEM で解いたコンプライアンス ÷ SIMP（1 に近いほど良い。2 倍以上は破綻）
   - 場: 相対 L2 誤差（1 は「全部 0 と予測」と同じ）
   - TopoDiff のテスト: `--data jobs/<data>/level_1.npz` で TopoDiff の評価（CE / VFE / LD / FM）
4. **詰める**: 結果を見て次に変える 1 つを決める（下の「分かっていること」）。前回と同じ `data` ジョブ・
   同じテスト件で比べ、`summary.md` の数値と `compare.png` を並べて報告する。

## 分かっていること（このツールでの検証から）

- **データ量が最も効く。** 540 件では学習データをほぼ丸暗記してテストで崩れた。2,340 件で太い部材と外形が改善した。
- **全面材料の FEM の結果（sed）を入力に入れると大きく効く**（情報は増えないが、力の流れを解いた結果を渡すことになる）。
  TopoDiff も同じ考え方で、ひずみエネルギー密度とミーゼス応力を入力にしている。
- **拘束の位置が自由に変わる問題は難しい。** fea-elastic（拘束 2 本・荷重 1 本を自由配置、2,000 件）では
  テストにほとんど汎化しなかった。先に拘束を固定した易しい問題で確かめてから広げる。
- 画素の誤差が小さくても、荷重経路が 1 本切れていれば性能は何十倍も悪い。トポロジー最適化は必ず
  コンプライアンス（FEM）でも確かめる。

## 例

`examples/` に仕様の例がある（`read_skill("ml-model", "examples/data_simp_topopt.json")` で読める）:
`data_simp_topopt.json`・`data_fea_elastic.json`・`data_topodiff.json`・`model_default.json`・
`model_ablation_no_sed.json`（sed を外した比較用）。
