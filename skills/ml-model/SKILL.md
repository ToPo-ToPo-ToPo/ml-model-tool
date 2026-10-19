---
name: ml-model
description: 物理問題の機械学習モデルを作るときの手順。学習データを作る（トポロジー最適化・FEM・TopoDiff の取り込み）、モデルを学習する（1 条件に 1 つの答えの FNO か、何通りもの案を出す生成モデル: 拡散・Flow matching・VAE・GAN）、評価する、複数のモデルを比べる。道具 ml-model-tool はコマンド行で使い、文章は解釈しない。仕様の鍵は spec data / spec model で確かめる
---

# 物理の機械学習モデルを作る（`mmt`）

道具は `ml-model-tool`。**文章は解釈しない**。何を学ばせるか・何を変えるか・結果をどう読むかはあなたの仕事で、
道具の仕事は「仕様どおりにデータを作る・学習する・評価する・比べる」こと。

## 守ること

1. **1 回に変えるのは 1 つだけ。** 比べるときは同じ data ジョブ・同じテスト件で比べる（`compare` が違いを警告する）。
2. **学習の損失で判断しない。** 必ず `evaluate`（テスト）の数値と `compare.png` の図を見る。損失は種類ごとに違う量で、
   種類どうしでは比べられない。
3. **鍵は推測しない。** 仕様を書く前に `spec data --source <source>` / `spec model --type <type>` を打ち、返る `example`
   をファイルに書いて、変えたい鍵だけ直す。
4. **差し戻しは `error.where` の鍵だけを直す。** `where` が直す場所、`expected` が書ける値。同じ行を打ち直さない。
5. **長い処理は待たない。** `data` と `train` はジョブになる。`job status --job-dir jobs/<…>` で進み具合を見る。止めるのは `job stop`。

## 流れ

道具は `app("ml-model-tool", line)` の 1 つ。仕様の JSON は作業フォルダに `write_file` で書く。パスは作業フォルダからの
相対で、返事の `artifacts` のパスを次の行にそのまま使う。まず `summary.md` を読み、図は `read_file` で見る。

| 段 | 打つ行 | 返るもの |
|---|---|---|
| 1. データの種類を選ぶ | `spec data` → `spec data --source simp-topopt` | source の一覧 → 鍵・既定値・意味・例 |
| 2. データを作る | `data --spec data.json` | `dataset.npz`・`preview.png` |
| 3. モデルの種類を選ぶ | `spec model` → `spec model --type flow` | 種類の一覧 → 鍵・既定値・意味・例 |
| 4. 学習する | `train --spec model.json --data jobs/<data>` | `model.mdlus`・`loss.png` |
| 5. 評価する | `evaluate --model jobs/<train> --data jobs/<data>` | `metrics.json`・`cases.csv`・`compare.png` |
| 6. 比べる | `compare --evals jobs/<評価>,jobs/<評価> [--by support]` | 比較表 `compare.md`・同じ件を並べた `compare.png` |

ほかに `predict --model … --data …`（予測だけ）、`help <コマンド>`（引数の意味）、`job list`。

## モデルの種類

| 目的 | `model.type` | メモ |
|---|---|---|
| 1 条件に 1 つの答え（場の予測・サロゲート） | `fno`（既定） | CPU でも動く。速い |
| 1 条件から何通りもの案（形の生成） | `flow`（まず試す）・`diffusion`・`vae`・`gan` | GPU か Apple の GPU（MPS）で学習。CPU では非常に遅い |

- 生成モデルは `evaluate` / `predict` に `--samples K`（案の数）、`flow` / `diffusion` は `--steps S`（生成の段数）を足せる。
- 生成モデルの数値は 1 案目（FNO とそのまま比べられる）。加えて `diversity`（案どうしの違い）と `best_of_k_*`（K 案中の最良）。

## 結果の読み方

`compare` の表では ↓ は小さいほど良い、↑ は大きいほど良い数値で、行の最良が太字になる。

| 数値 | 意味 | 目安 |
|---|---|---|
| `iou` ↑ | 材料（0.5 以上）の重なり | 画素が合っても性能が良いとは限らない |
| `compliance_ratio` ↓ | 予測した形を FEM で解いた剛性の逆数 ÷ SIMP | 1 に近いほど良い。2 以上は破綻 |
| `floating_share` ↓ | 拘束までつながっていない材料を含む案の割合 | 0 が理想 |
| `rel_l2` ↓ | 場の相対誤差（量ごとに `rel_l2/<量>`） | 1 は「全部 0 と予測」と同じ |
| `peak_stress_error` ↓ | 最大ミーゼス応力の誤差 | FEM の結果を予測するとき |
| `diversity` | 生成した案どうしの違い | 0 に近いなら毎回同じ形 |
| `best_of_k_compliance_ratio` ↓ | K 案を FEM で解いて選んだ最良 | 1 案目よりはっきり良いなら「多く出して FEM で選ぶ」が効く |

学習の `overfit_ratio`（検証 ÷ 学習の損失）が 2 を超えたら丸暗記。データを増やすか、入力・モデルを見直す。

## よくある作業

- **入力の効果を確かめる**: 同じデータで `inputs` だけを変えた 2 つを学習・評価し、`compare` する（例: `examples/model_ablation_no_sed.json`）。
- **FNO と生成モデルを比べる**: 同じデータで `model.type` だけを変え、`compare --evals …,… --by support` で拘束の種類ごとにも見る。
- **データ量の効果を確かめる**: `train.n_train` だけを変える。
- **FEM の結果を予測する**: `simp-fea` のデータで FNO を学習する。荷重ケースごとの差は `compare --by load_kind`（最適化に使った荷重か新しい荷重か）。

## 分かっていること（このツールでの検証から）

- **データ量が最も効く。** 540 件では丸暗記してテストで崩れた。2,340 件で太い部材と外形が改善した。
- **全面材料の FEM の結果（sed）を入力に入れると大きく効く**（TopoDiff も同じ考え方）。
- **トポロジー最適化の生成**（64×32・600 件・テスト 60 件・各 8 案）: コンプライアンス比の中央値は FNO 1.05・拡散 1.14・
  Flow 1.02・VAE 1.08・GAN 1.01。Flow は 8 案から FEM で選ぶと 1.006。GAN は毎回ほぼ同じ形（`diversity` 0.003）。
  浮いた材料を含む割合は FNO 50%・Flow 11%・GAN 9%（FNO は形が崩れやすい）。
- **FEM の結果の予測**（simp-fea・3,000 形状 × 4 荷重、FNO）: 量ごとの相対 L2 は約 20%、最大応力の誤差 4%、
  コンプライアンスの誤差 7%。応力の傾向はつかめるが、大きく曲がる件の変形は小さめに出る。つり合いの式を損失に
  足す `train.physics` は残差を下げたが、大きな変形はかえって悪くなった（試すなら 0.1 程度と 0 を比べる）。
- **拘束の位置が自由に変わる問題は難しい。** fea-elastic（2,000 件）ではテストにほとんど汎化しなかった。
- 画素の誤差が小さくても、荷重経路が 1 本切れれば性能は何十倍も悪い。トポロジー最適化は必ずコンプライアンスでも確かめる。

## 例

`examples/` の仕様（`read_skill("ml-model", "examples/<名前>")`）: `data_simp_topopt.json`・`data_fea_elastic.json`・
`data_topodiff.json`・`model_default.json`・`model_ablation_no_sed.json`・`model_diffusion.json`・`model_flow.json`・
`model_vae.json`・`model_gan.json`。鍵の一覧は `spec data` / `spec model` が正本。
