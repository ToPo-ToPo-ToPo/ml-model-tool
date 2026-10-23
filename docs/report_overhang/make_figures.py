"""報告書の図を作る。
    python make_figures.py <jobs フォルダ（~/overhang_work/jobs）> <出力フォルダ>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml_model_tool import dataset
from ml_model_tool.overhang import BUILD_DIRECTIONS, unprintable

plt.rcParams["font.family"] = "Hiragino Sans"
J, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SUPPORT = {"cantilever": "片持ち梁", "both-ends": "両端固定", "pin-roller": "ピンとローラー"}
RED = [0.84, 0.16, 0.14]


def rgb(img, direction, red=True):
    img = np.clip(img, 0, 1)
    out = np.repeat((1 - img)[..., None], 3, axis=2)
    if red:
        out[unprintable(img, direction)] = RED
    return out


def evaluation(name):
    d = J / name
    pred = np.load(d / "predictions.npz")["samples"][:, 0, 0]
    rows = {int(r["index"]): r for r in csv.DictReader(open(d / "cases.csv"))}
    return pred, rows


def cap(row):
    return f"C {float(row['compliance_ratio']):.2f} 倍　造れない {float(row['overhang']):.1%}"


def grid(path, cases, cols, row_label):
    """cols: [(見出し, 件 → 画像, 件 → 説明 or None, 赤を付けるか)]。"""
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(2.35 * len(cols), 1.38 * len(cases) + 0.35),
                             squeeze=False)
    for r, (i, d) in enumerate(cases):
        for c, (title, image, text, red) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(rgb(image(i), d, red), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5); s.set_color("0.6")
            t = text(i) if text else ""
            ax.set_title((title + "\n" if r == 0 else "") + t, fontsize=7.5, linespacing=1.3)
        axes[r, 0].set_ylabel(row_label(r, i), fontsize=7.5)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print("saved", path)


def pick(ds, per_support):
    """支え方ごとに、テストの先頭から per_support 件ずつ選ぶ。"""
    sup = ds.aux["support"].astype(int)
    names = list(SUPPORT)
    out = []
    for k in range(len(names)):
        out += [int(i) for i in np.flatnonzero(sup == k)[:per_support]]
    return out


def support_name(ds, i):
    return SUPPORT[list(SUPPORT)[int(ds.aux["support"][i])]]


# 図 1: データの例（入力 = 普通の最適化、正解 = 自己支持の最適化）
c3 = dataset.load(J / "test_common3")
d3 = lambda i: BUILD_DIRECTIONS[int(c3.aux["build"][i])]
sel = pick(c3, 1)
over = lambda img, d: float(unprintable(np.clip(img, 0, 1), d).sum() / max((np.clip(img, 0, 1) > 0.5).sum(), 1))
grid(OUT / "fig_data.png", [(i, d3(i)) for i in sel],
     [("入力（オーバーハング制約なし）", lambda i: c3.x[i, 0], lambda i: f"造れない材料 {over(c3.x[i, 0], d3(i)):.1%}", True),
      ("正解（自己支持、半径 1.5）", lambda i: c3.y[i, 0], None, True)],
     lambda r, i: support_name(c3, i))

# 図 2: 最初のデータセット（5000 件）で学習した拡散モデルの出力（テストのうち番号 2000 以降の 291 件）
p5, r5 = evaluation("eval_diff3_common3")
sel = pick(c3, 2)[::2] + [pick(c3, 2)[1]]
grid(OUT / "fig_first.png", [(i, d3(i)) for i in sel],
     [("入力", lambda i: c3.x[i, 0], lambda i: f"造れない材料 {float(r5[i]['overhang_input']):.1%}", True),
      ("正解", lambda i: c3.y[i, 0], None, True),
      ("拡散モデル（5000 件で学習）", lambda i: p5[i], lambda i: cap(r5[i]), True)],
     lambda r, i: support_name(c3, i))

# 図 3: グレーと白黒の正解（5000 件、拡散モデル）
c3b = dataset.load(J / "test_common3b")
gray_p, gray_r = evaluation("eval_diff3_common3")
bin_p, bin_r = evaluation("eval_diffbin3_common3b")
sel = pick(c3, 1)
grid(OUT / "fig_binarize.png", [(i, d3(i)) for i in sel],
     [("正解（グレーあり）", lambda i: c3.y[i, 0], None, False),
      ("グレーありで学習", lambda i: gray_p[i], None, False),
      ("正解（0.5 で白黒）", lambda i: c3b.y[i, 0], None, False),
      ("白黒で学習", lambda i: bin_p[i], None, False)],
     lambda r, i: support_name(c3, i))

# 図 4: 正解の単純化（元の正解と、半径 4・β の継続法で解き直した正解）
sf = dataset.load(J / "test_sf70")
old = dataset.load(J / "data3")
src = sf.aux["source_index"].astype(int)
dsf = lambda i: BUILD_DIRECTIONS[int(sf.aux["build"][i])]
sel = [0, 4, 5, 6]
grid(OUT / "fig_simplify.png", [(i, dsf(i)) for i in sel],
     [("元の正解（半径 1.5・β 5）", lambda i: old.y[src[i], 0], None, False),
      ("新しい正解（半径 4・β 5→32、白黒）", lambda i: sf.y[i, 0], None, False)],
     lambda r, i: support_name(sf, i))

# 図 5: 新しい正解で学習したモデルの結果（どのモデルも見ていない 70 件）
fp, fr = evaluation("eval_flowsf3_test_sf70")
dp, dr = evaluation("eval_diffsf3_test_sf70")
sel = list(range(8))
grid(OUT / "fig_results.png", [(i, dsf(i)) for i in sel],
     [("入力", lambda i: sf.x[i, 0], lambda i: f"造れない材料 {float(fr[i]['overhang_input']):.1%}", True),
      ("正解", lambda i: sf.y[i, 0], None, True),
      ("フローマッチング", lambda i: fp[i], lambda i: cap(fr[i]), True),
      ("拡散モデル", lambda i: dp[i], lambda i: cap(dr[i]), True)],
     lambda r, i: f"件 {r + 1}")
