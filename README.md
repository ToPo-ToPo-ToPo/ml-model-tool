# ml-model-tool

機械学習による物理モデル（サロゲート・生成モデル）を作るツール。JSON の仕様書から
**学習データを作り → PhysicsNeMo の FNO を学習し → 正解と比べて評価する**。

- 人は CLI（`mmt`）か窓（入力欄 1 つ）で使う
- 会社の AIOS のアプリ（規約 2.10）として、ローカル LLM エージェントが `app("ml-model-tool", line)` で使う。
  使い方は手順書 [`skills/ml-model/SKILL.md`](skills/ml-model/SKILL.md)。エージェントは仕様の JSON を書いて
  コマンド行を打ち、結果（`summary.md`・図・数値）を読んで次の条件を決める
- 中核は Python ライブラリ `ml_model_tool`（コマンド層 `ml_model_tool_cli` から独立）

## できること

| コマンド | 内容 |
|---|---|
| `data --spec data.json` | 学習データを作る。`simp-topopt`（SIMP のトポロジー最適化）・`fea-elastic`（線形弾性 FEM の変位場）・`topodiff`（TopoDiff データセットの取り込み） |
| `train --spec model.json --data <data のジョブ>` | FNO を学習する。入力チャネルの選択・学習件数・エポック数などを仕様で指定 |
| `evaluate --model <train のジョブ> --data <データ>` | 正解との誤差、FEM で解いたコンプライアンス（トポロジー最適化）、TopoDiff の評価指標 |
| `predict --model <train のジョブ> --data <データ>` | 予測 |
| `help [<コマンド>]` / `job list` / `job status` / `job stop` | 書式の説明・ジョブの一覧・状態・停止 |

仕様のキーは [docs/specs.md](docs/specs.md)。

## 動かす

Apple シリコン Mac（macOS 14 以降。CPU）か Linux（CUDA があれば GPU）。

```bash
uv sync                      # TopoDiff の評価も使うなら: uv sync --extra topodiff
mkdir -p workspace && cd workspace
cp ../skills/ml-model/examples/data_simp_topopt.json data.json
cp ../skills/ml-model/examples/model_default.json model.json
uv run mmt data --spec data.json --out "$PWD/jobs/data1"
uv run mmt train --spec model.json --data jobs/data1 --out "$PWD/jobs/train1"
uv run mmt evaluate --model jobs/train1 --data jobs/data1 --out "$PWD/jobs/eval1"
```

標準出力は結果の JSON 1 行（終了コード 0 = 済んだ・2 = 走らせる前の差し戻し・1 = 走って失敗）。
各ジョブのフォルダに `summary.md`・`metrics.json`・図・`status.json` ができる。

窓: `uv run python webapp/app.py --open`（入力欄に同じコマンド行を打つ。ジョブは別プロセスで走り、停止ボタンで止まる）。

## 会社の AIOS に組み込む

- アプリの宣言は [`.mcp.json`](.mcp.json)（窓のポート 5080）。`local-aios doctor` の静的チェックは通過済み
- 会社リポジトリ側では `aios-apps.repos` への追加・`docs/ports.md` への 5080 の登録・エージェントの `apps.repos` で
  見せる設定が要る

## 構成

```text
src/ml_model_tool/          ライブラリ
  sources/                  データ仕様 → 標準データセット（simp_topopt / fea_elastic / topodiff）
  dataset.py                標準データセット（dataset.npz + dataset.json）
  spec.py                   データ仕様・モデル仕様の既定値と検査
  fem.py                    2D 線形弾性 FEM と SIMP（top88 相当）
  model.py / training.py    FNO の構築・保存、学習ループ
  evaluation.py / plots.py  評価（誤差・FEM・TopoDiff の評価）と図
  mcp_server.py             MCP サーバ（道具は run 1 つ）
src/ml_model_tool_cli/      コマンドの宣言（agk.connect）・CLI・ジョブ
webapp/app.py               窓
skills/ml-model/            エージェント向けの手順書と仕様の例
tests/                      pytest（CPU で数十秒）
```

新しい物理問題を足すときは `sources/` に 1 つ足す（学習・評価は共通）。

## 注意

- TopoDiff の評価コードは SolidsPy 1.0 向けで、今の NumPy では動かない。SolidsPy 1.1 の呼び方に置き換えた
  関数を使い、荷重節点を 1 件しか設定しない不具合も避けている（`evaluation.py`）
- 点荷重・点拘束は使わない（2 次元弾性の特異点）。荷重・拘束は 2〜3 節点の線で与える

## ライセンス

Apache-2.0
