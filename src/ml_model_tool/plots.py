"""Figures for summaries (PNG). Density targets in grey (black = material), fields in red/blue.

Supports (channels ``fix_x`` / ``fix_y``) are drawn as blue marks and the load (``load_x`` /
``load_y``) as an orange arrow at its centroid, whenever the inputs have those channels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

INK, MUTED, SUPPORT, LOAD = "#1f2328", "#6e7781", "#2f6fbf", "#d4580f"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Hiragino Sans", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    return plt


def _overlay(ax, x: np.ndarray, channels: list[str]) -> None:
    def ch(name):
        return x[channels.index(name)] if name in channels else None

    fx, fy = ch("fix_x"), ch("fix_y")
    if fx is not None and fy is not None:
        iy, ix = np.nonzero((fx > 0) | (fy > 0))
        ax.scatter(ix, iy, s=6, marker="s", color=SUPPORT, linewidths=0, zorder=3)
    lx, ly = ch("load_x"), ch("load_y")
    if lx is not None and ly is not None:
        mag = np.hypot(lx, ly)
        if mag.sum() > 0:
            iy, ix = np.nonzero(mag)
            cy, cx = np.average(iy, weights=mag[iy, ix]), np.average(ix, weights=mag[iy, ix])
            v = np.array([lx.sum(), -ly.sum()])          # y up in the solver, down in the image
            v = v / (np.linalg.norm(v) + 1e-12) * 0.15 * max(x.shape[1:])
            ax.annotate("", xy=(cx + v[0], cy + v[1]), xytext=(cx, cy), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=LOAD, lw=2, mutation_scale=12))


def _show(ax, img: np.ndarray, kind: str, vmax: Optional[float] = None):
    ax.set_xticks([]); ax.set_yticks([])
    if kind == "density":
        return ax.imshow(img, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    vmax = vmax or (np.abs(img).max() + 1e-12)
    return ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")


def compare(path, x: np.ndarray, x_channels: list[str], pred: np.ndarray, truth: Optional[np.ndarray],
            y_channels: list[str], y_kind: str, title: str, notes: Optional[list] = None) -> None:
    """One row per case: for every output channel, truth (if any) and prediction side by side.

    ``pred`` is (N, C, H, W), or (N, K, C, H, W) for K generated designs per case (shown as sample 1..K).
    ``notes`` is one string per case, or one list of K strings per case."""
    plt = _plt()
    generative = pred.ndim == 5
    if not generative:
        pred = pred[:, None]
        notes = None if notes is None else [[s] for s in notes]
    n, k = pred.shape[:2]
    per = k + (1 if truth is not None else 0)
    cols = len(y_channels) * per
    fig, axes = plt.subplots(n, cols, figsize=(2.6 * cols + 0.4, 2.4 * n + 0.6), squeeze=False, facecolor="white")
    for i in range(n):
        c = 0
        for j, name in enumerate(y_channels):
            vmax = np.abs(truth[i, j]).max() if truth is not None and y_kind != "density" else None
            if truth is not None:
                _show(axes[i, c], truth[i, j], y_kind, vmax)
                _overlay(axes[i, c], x[i], x_channels)
                axes[i, c].set_title(f"case {i + 1}  {name} truth", fontsize=8, loc="left", color=INK)
                c += 1
            for s in range(k):
                _show(axes[i, c], pred[i, s, j], y_kind, vmax)
                _overlay(axes[i, c], x[i], x_channels)
                note = notes[i][s] if notes and s < len(notes[i]) and notes[i][s] else ""
                label = f"sample {s + 1}" if generative else "predicted"
                head = f"case {i + 1}  " if truth is None and s == 0 else ""
                axes[i, c].set_title(f"{head}{name} {label}" + (f"  {note}" if note else ""), fontsize=8,
                                     loc="left", color=INK)
                c += 1
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def side_by_side(path, x: np.ndarray, x_channels: list[str], truth: Optional[np.ndarray], cols: list,
                 y_kind: str, title: str) -> None:
    """One row per case: truth (if any), then one column per model. ``cols`` is [(name, (N, H, W))]."""
    plt = _plt()
    n = len(x)
    ncol = len(cols) + (1 if truth is not None else 0)
    h, w = cols[0][1].shape[1:]
    row = 2.6 * h / w + 0.35                       # panel height from the grid's aspect, plus the title
    fig, axes = plt.subplots(n, ncol, figsize=(2.6 * ncol + 0.4, row * n + 0.6), squeeze=False, facecolor="white")
    for i in range(n):
        vmax = np.abs(truth[i]).max() if truth is not None and y_kind != "density" else None
        panels = ([("truth", truth[i])] if truth is not None else []) + [(name, img[i]) for name, img in cols]
        for c, (name, img) in enumerate(panels):
            _show(axes[i, c], img, y_kind, vmax)
            _overlay(axes[i, c], x[i], x_channels)
            axes[i, c].set_title((f"case {i + 1}  " if c == 0 else "") + name, fontsize=8, loc="left", color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def samples(path, x: np.ndarray, x_channels: list[str], y: Optional[np.ndarray], y_channels: list[str],
            y_kind: str, title: str, n: int = 6) -> None:
    """Preview of a dataset: the first n samples (all input channels and the targets)."""
    plt = _plt()
    n = min(n, len(x))
    names = list(x_channels) + (list(y_channels) if y is not None else [])
    fig, axes = plt.subplots(n, len(names), figsize=(1.9 * len(names) + 0.4, 1.9 * n + 0.6), squeeze=False,
                             facecolor="white")
    for k in range(n):
        for j, name in enumerate(names):
            ax = axes[k, j]
            if j < len(x_channels):
                img = x[k, j]
                if name.startswith("load"):               # signed: red / blue around zero
                    v = np.abs(img).max() + 1e-12
                    ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                else:
                    ax.imshow(img, cmap="viridis", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                _show(ax, y[k, j - len(x_channels)], y_kind)
                _overlay(ax, x[k], x_channels)
            if k == 0:
                ax.set_title(name, fontsize=8, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def loss_curve(history: list, path, loss_name: str) -> None:
    plt = _plt()
    h = np.asarray(history, float)
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor="white")
    ax.plot(h[:, 0], h[:, 1], color="#2f6fbf", lw=2, label="train")
    ax.plot(h[:, 0], h[:, 2], color="#d4580f", lw=2, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel(f"{loss_name} loss")
    if h[:, 1:3].min() > 0 and h[:, 1:3].max() / h[:, 1:3].min() > 30:
        ax.set_yscale("log")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
