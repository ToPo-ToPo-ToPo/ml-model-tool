"""Evaluations side by side (``compare``): the key numbers as one table, a breakdown by a group of cases,
and one figure with the same test cases from every model.

An evaluation is an evaluate job folder (metrics.json, cases.csv, predictions.npz). The table takes the key
numbers of each (``evaluation.key_numbers``) and the training numbers of its model (parameters, epochs,
minutes); where it is clear which way is better, the best value of a row is marked. Evaluations of different
datasets or splits are compared too, with a warning: their numbers are not of the same cases.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .evaluation import better, key_numbers

MAX_RUNS = 8
N_FIGURE = 6
# per-case columns summarized by ``--by`` (when present): (column, how)
BY_COLUMNS = [("iou", "median"), ("rel_l2", "median"), ("compliance_ratio", "median"),
              ("compliance_ratio", "share_le_1.05"), ("floating", "share_gt_0"),
              ("best_of_k_compliance_ratio", "median"), ("peak_stress_error", "median"),
              ("compliance_error", "median"), ("equilibrium_residual", "median"), ("CE", "mean")]


@dataclass
class Run:
    name: str
    folder: Path
    metrics: dict
    train: dict = field(default_factory=dict)

    @property
    def numbers(self) -> dict:
        kn = key_numbers(self.metrics)
        n = {"model_type": self.metrics.get("model_type")}
        for k in ("parameters", "epochs", "minutes"):
            if k in self.train:
                n[f"train_{k}" if k != "parameters" else k] = self.train[k]
        if "samples" in kn:
            n["samples"] = kn.pop("samples")
        n.update(kn)
        return n


def load_run(folder: Path, name: str = "") -> Run:
    m_path = folder / "metrics.json"
    if not m_path.is_file() or not (folder / "predictions.npz").is_file():
        raise ValueError(f"not an evaluate job folder: {folder} (needs metrics.json and predictions.npz)")
    m = json.loads(m_path.read_text(encoding="utf-8"))
    if "split" not in m or "model_type" not in m:
        raise ValueError(f"not an evaluate job folder: {folder} (metrics.json of another command)")
    train = {}
    t_path = Path(m.get("model") or "") / "metrics.json"
    if m.get("model") and t_path.is_file():
        train = json.loads(t_path.read_text(encoding="utf-8"))
    return Run(name or m["model_type"], folder, m, train)


def parse_evals(value: str, base: Path) -> list[Path]:
    """``a,b,c`` (evaluate job folders, relative to ``base``) -> checked absolute folders."""
    parts = [v.strip() for v in str(value or "").split(",") if v.strip()]
    if not 2 <= len(parts) <= MAX_RUNS:
        raise ValueError(f"pass 2-{MAX_RUNS} evaluate job folders separated by commas (got {len(parts)})")
    out = []
    for v in parts:
        p = Path(v).expanduser()
        p = (p if p.is_absolute() else Path(base) / p).resolve()
        load_run(p)
        out.append(p)
    return out


def unique_names(runs: list[Run]) -> None:
    seen: dict[str, int] = {}
    for r in runs:
        seen[r.name] = seen.get(r.name, 0) + 1
    count: dict[str, int] = {}
    for r in runs:
        if seen[r.name] > 1:
            count[r.name] = count.get(r.name, 0) + 1
            r.name = f"{r.name}-{count[r.name]}"


def warnings(runs: list[Run]) -> list[str]:
    out = []
    datas = {Path(r.metrics.get("data") or "").resolve() for r in runs if r.metrics.get("data")}
    if len(datas) > 1:
        out.append("the evaluations use different datasets: their numbers are not of the same cases")
    if len({r.metrics.get("split") for r in runs}) > 1:
        out.append("the evaluations use different splits: their numbers are not of the same cases")
    if len({r.metrics.get("cases") for r in runs}) > 1:
        out.append("the evaluations have different numbers of cases")
    return out


def table(runs: list[Run]) -> tuple[list[str], dict]:
    """Rows (number names in a stable order) and {row: {run name: value}}."""
    rows: list[str] = []
    for r in runs:
        for k in r.numbers:
            if k not in rows:
                rows.append(k)
    return rows, {k: {r.name: r.numbers.get(k) for r in runs} for k in rows}


def best_of_row(row: str, values: dict) -> Optional[str]:
    way = better(row)
    vals = {n: v for n, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not way or len(vals) < 2 or len(set(vals.values())) == 1:
        return None
    return (min if way == "low" else max)(vals, key=vals.get)


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def markdown(runs: list[Run], rows: list[str], tab: dict) -> str:
    names = [r.name for r in runs]
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for row in rows:
        best = best_of_row(row, tab[row])
        cells = [(f"**{_fmt(tab[row][n])}**" if n == best else _fmt(tab[row][n])) for n in names]
        way = {"low": " ↓", "high": " ↑"}.get(better(row), "")
        lines.append(f"| {row}{way} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# breakdown by a group of cases
# ---------------------------------------------------------------------------
def read_cases(folder: Path) -> list[dict]:
    p = folder / "cases.csv"
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def groups_of(runs: list[Run]) -> list[str]:
    """Groups of cases every run can be broken down by (the dataset's ``groups``, in cases.csv)."""
    common = [g for g in runs[0].metrics.get("groups") or []]
    return [g for g in common if all(g in (r.metrics.get("groups") or []) for r in runs)]


def _summarize(v: np.ndarray, how: str) -> float:
    if how == "median":
        return round(float(np.median(v)), 4)
    if how == "mean":
        return round(float(np.mean(v)), 4)
    if how == "share_le_1.05":
        return round(float(np.mean(v <= 1.05)), 3)
    return round(float(np.mean(v > 0)), 3)          # share_gt_0


def breakdown(runs: list[Run], by: str) -> dict:
    """{group value: {"cases": n, "<column> <how>": {run name: value}}} from cases.csv."""
    out: dict = {}
    for r in runs:
        rows = read_cases(r.folder)
        if not rows or by not in rows[0]:
            raise ValueError(f"--by {by}: {r.folder.name}/cases.csv has no column {by!r} "
                             f"(columns to break down by: {', '.join(groups_of(runs)) or 'none'})")
        for value in dict.fromkeys(row[by] for row in rows):
            part = [row for row in rows if row[by] == value]
            g = out.setdefault(value, {"cases": {}})
            g["cases"][r.name] = len(part)
            for col, how in BY_COLUMNS:
                if col in part[0] and part[0][col] != "":
                    g.setdefault(f"{col} {how}", {})[r.name] = _summarize(np.array([float(x[col]) for x in part]), how)
    return out


def breakdown_markdown(runs: list[Run], by: str, parts: dict) -> str:
    names = [r.name for r in runs]
    blocks = []
    for value, g in parts.items():
        lines = [f"**{by} = {value}**", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for row, vals in g.items():
            col = row.split(" ", 1)[0]
            way = "high" if row.endswith("share_le_1.05") else better(col) if not row.endswith("share_gt_0") else "low"
            vals_num = {n: v for n, v in vals.items() if isinstance(v, (int, float))}
            pick = None
            if row != "cases" and way and len(vals_num) > 1 and len(set(vals_num.values())) > 1:
                pick = (min if way == "low" else max)(vals_num, key=vals_num.get)
            cells = [(f"**{_fmt(vals.get(n))}**" if n == pick else _fmt(vals.get(n))) for n in names]
            lines.append(f"| {row} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def figure(path: Path, runs: list[Run], channel: str = "") -> Optional[str]:
    """The first cases shared by every run: truth (if any) and each run's prediction (sample 1), one channel.
    Returns the channel shown, or None when the runs share no cases (or their dataset is gone)."""
    from . import dataset, plots

    data = runs[0].metrics.get("data")
    if not data or not Path(data).exists():
        return None
    ds = dataset.load(data)
    preds = [np.load(r.folder / "predictions.npz") for r in runs]
    common = set(preds[0]["index"].tolist())
    for p in preds[1:]:
        common &= set(p["index"].tolist())
    idx = [i for i in preds[0]["index"].tolist() if i in common][:N_FIGURE]
    if not idx:
        return None
    chans = ds.meta["y_channels"]
    ch = channel or chans[0]
    if ch not in chans:
        raise ValueError(f"--channel {ch}: the dataset's output channels are {', '.join(chans)}")
    j = chans.index(ch)
    cols = []
    for r, p in zip(runs, preds):
        pos = {int(i): k for k, i in enumerate(p["index"])}
        cols.append((r.name, np.stack([p["pred"][pos[i], j] for i in idx])))
    truth = ds.y[idx, j] if ds.y is not None else None
    plots.side_by_side(path, ds.x[idx], ds.x_channels, truth, cols, ds.meta["y_kind"],
                       f"{ch}: the same {len(idx)} {runs[0].metrics.get('split', '')} cases, sample 1 of each model")
    return ch


def compare(folders: list[Path], out: Path, names: Optional[list[str]] = None, by: str = "",
            channel: str = "") -> dict:
    """Writes compare.md / compare.json / compare.csv (and compare.png) into ``out``; returns compare.json."""
    if not 2 <= len(folders) <= MAX_RUNS:
        raise ValueError(f"compare takes 2-{MAX_RUNS} evaluate job folders (got {len(folders)})")
    if names and len(names) != len(folders):
        raise ValueError(f"--names has {len(names)} names for {len(folders)} evaluations")
    runs = [load_run(f, (names or [""] * len(folders))[i]) for i, f in enumerate(folders)]
    if not names:
        unique_names(runs)
    rows, tab = table(runs)
    warn = warnings(runs)
    parts = breakdown(runs, by) if by else {}
    shown = figure(out / "compare.png", runs, channel)
    md = [markdown(runs, rows, tab)]
    if parts:
        md += ["", f"## by {by}", "", breakdown_markdown(runs, by, parts)]
    best = {row: b for row in rows if (b := best_of_row(row, tab[row]))}
    doc = {"runs": [{"name": r.name, "evaluation": str(r.folder), "model": r.metrics.get("model"),
                     "data": r.metrics.get("data"), "split": r.metrics.get("split")} for r in runs],
           "table": tab, "best": best, "warnings": warn, "groups": groups_of(runs)}
    if parts:
        doc["by"] = {"group": by, "parts": parts}
    if shown:
        doc["figure_channel"] = shown
    (out / "compare.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (out / "compare.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", *(r.name for r in runs)])
        for row in rows:
            w.writerow([row, *("" if tab[row][r.name] is None else tab[row][r.name] for r in runs)])
    doc["markdown"] = "\n".join(md)
    return doc
