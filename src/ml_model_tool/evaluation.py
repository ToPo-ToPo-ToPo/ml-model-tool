"""Evaluation of a trained model on one split of a dataset.

Always (when the split has targets):
  density targets: mae, bce, iou (material = density >= 0.5), volume_error
  field targets:   mse, rel_l2 (per sample ||pred - y|| / ||y||; median and 90th percentile)
Domain checks, chosen by the dataset's task:
  topopt          compliance of the predicted density by FEM / compliance of the SIMP design
                  (median, share <= 1.05, share > 2, worst)
  topodiff-level  TopoDiff's own analysis (SolidsPy FEM): CE, VFE, LD, FM as in the paper
Writes metrics.json, predictions.npz and compare.png (the first 8 cases of the split).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import plots
from .dataset import Dataset
from .model import Bundle

N_FIGURE = 8


def _stats(v: np.ndarray, digits: int = 4) -> dict:
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), digits), "median": round(float(np.median(v)), digits),
            "p90": round(float(np.percentile(v, 90)), digits), "max": round(float(v.max()), digits)}


def generic_metrics(pred: np.ndarray, y: np.ndarray, y_kind: str) -> dict:
    n = len(pred)
    if y_kind == "density":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).reshape(n, -1).mean(1)
        pb, yb = pred >= 0.5, y >= 0.5
        inter = (pb & yb).reshape(n, -1).sum(1)
        union = np.maximum((pb | yb).reshape(n, -1).sum(1), 1)
        return {"mae": _stats(np.abs(pred - y).reshape(n, -1).mean(1)), "bce": _stats(bce),
                "iou": _stats(inter / union),
                "volume_error": _stats(np.abs(pred.reshape(n, -1).mean(1) - y.reshape(n, -1).mean(1)))}
    err = (pred - y).reshape(n, -1)
    rel = np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(y.reshape(n, -1), axis=1), 1e-12)
    return {"mse": _stats((err ** 2).mean(1), 6), "rel_l2": _stats(rel)}


def topopt_compliance(ds: Dataset, idx: np.ndarray, pred: np.ndarray, progress) -> dict:
    from .fem import FEM, node_masks_to_dofs

    g = ds.meta["grid"]
    fem = FEM(g["nelx"], g["nely"])
    ratios = []
    for k, i in enumerate(idx):
        fixed, F = node_masks_to_dofs(ds.aux["fix"][i], ds.aux["load"][i])
        ratios.append(fem.compliance(pred[k, 0], F, fixed) / float(ds.aux["compliance"][i]))
        if (k + 1) % 10 == 0:
            progress(0.5 + 0.45 * (k + 1) / len(idx), f"FEM check {k + 1}/{len(idx)}")
    r = np.asarray(ratios)
    return {"compliance_ratio": {**_stats(r), "share_le_1.05": round(float(np.mean(r <= 1.05)), 3),
                                 "share_gt_2": round(float(np.mean(r > 2)), 3)}}, r


def _solidspy_v11(path):
    """TopoDiff's ``mysolidspy`` for SolidsPy 1.1 (same files, same outputs; 1.0.x does not run on NumPy >= 1.24)."""
    import solidspy.assemutil as ass
    import solidspy.postprocesor as pos
    import solidspy.preprocesor as pre
    import solidspy.solutil as sol

    nodes, mats, elements, loads = pre.readin(folder=path)
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], elements)
    stiff, _ = ass.assembler(elements, mats, nodes[:, :3], neq, assem_op)
    disp = sol.static_sol(stiff, ass.loadasem(loads, bc_array, neq))
    full = pos.complete_disp(bc_array, nodes, disp)
    strain, stress = pos.strain_nodes(nodes, elements, mats, full)
    return full, strain, stress


def topodiff_analysis(ds: Dataset, pred: np.ndarray, work: Path, progress) -> dict:
    """CE / VFE / LD / FM with TopoDiff's analysis code (its load-node bug avoided, see docs)."""
    repo = ds.meta["params"].get("topodiff_repo") or ""
    if not repo or not (Path(repo) / "topodiff").is_dir():
        raise ValueError("this level needs topodiff_repo (a clone of github.com/francoismaze/topodiff) in the data spec")
    sys.path.insert(0, str(Path(repo).resolve()))
    from topodiff import topodiff_analysis as ta

    ta.mysolidspy = _solidspy_v11
    cases = json.loads((ds.path.parent / ds.meta["params"]["cases"]).read_text(encoding="utf-8"))
    for c in cases:     # TopoDiff's pre_process_summaries only fills the last case (indentation slip)
        c["load_nodes"] = np.asarray([max(int(round(64 * x) * 65 + round(64 * (1.0 - y))), 0) + 1
                                      for x, y in c["load_coord"]], int)
    img = np.where((1 - pred[:, 0]) * 255 < 127, 0.0, 255.0)[..., None]   # TopoDiff's binarization
    cwd = os.getcwd()
    (work / "fem_files").mkdir(exist_ok=True)
    os.chdir(work)                              # TopoDiff writes its FEM input files to ./fem_files/
    try:
        ce, ld, vfe, fm = [], [], [], []
        for i in range(len(img)):
            with contextlib.redirect_stdout(io.StringIO()):
                c = ta.fem_compliance_i(img, cases, i)
            ce.append(c / float(ds.aux["compliance"][i]) - 1)
            ld.append(bool(ta.check_load(cases[i]["load_coord"][0], img[i])))
            vfe.append(abs(ta.compute_vf(img[i]) - cases[i]["VF"]) / cases[i]["VF"])
            fm.append(bool(ta.check_floating_material(img[i].astype(np.uint8))))
            if (i + 1) % 10 == 0:
                progress(0.5 + 0.45 * (i + 1) / len(img), f"TopoDiff analysis {i + 1}/{len(img)}")
    finally:
        os.chdir(cwd)
    ce = np.asarray(ce)
    return {"topodiff": {"CE_mean": round(float(ce.mean()), 4), "CE_median": round(float(np.median(ce)), 4),
                         "CE_share_gt_0.3": round(float(np.mean(ce > 0.3)), 3),
                         "VFE_mean": round(float(np.mean(vfe)), 4), "LD_share": round(float(np.mean(ld)), 3),
                         "FM_share": round(float(np.mean(fm)), 3), "cases": len(ce)}}


def evaluate(bundle: Bundle, ds: Dataset, split: str, out_dir: str | Path,
             progress: Callable[[float, str], None]) -> dict:
    out = Path(out_dir)
    idx = ds.indices(split)
    if len(idx) == 0:
        raise ValueError(f"the dataset has no {split} samples")
    progress(0.05, f"predicting {len(idx)} {split} cases")
    pred = bundle.predict(ds.x[idx], ds.x_channels)
    y = ds.y[idx] if ds.y is not None else None
    y_kind = ds.meta["y_kind"]
    metrics: dict = {"split": split, "cases": int(len(idx)), "model": str(bundle.folder),
                     "inputs": bundle.inputs}
    if y is not None:
        metrics.update(generic_metrics(pred, y, y_kind))
    notes = None
    task = ds.meta.get("task")
    if task == "topopt" and "compliance" in ds.aux:
        dom, ratios = topopt_compliance(ds, idx, pred, progress)
        metrics.update(dom)
        notes = [f"C x{r:.2f}" for r in ratios[:N_FIGURE]]
    elif task == "topodiff-level":
        metrics.update(topodiff_analysis(ds, pred, out, progress))
    np.savez_compressed(out / "predictions.npz", index=idx, pred=pred.astype(np.float32))
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    k = min(N_FIGURE, len(idx))
    plots.compare(out / "compare.png", ds.x[idx[:k]], ds.x_channels, pred[:k], None if y is None else y[:k],
                  ds.meta["y_channels"], y_kind, f"{split}: first {k} cases ({Path(bundle.folder).name})", notes)
    return metrics
