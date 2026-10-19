"""Evaluation of a trained model on one split of a dataset.

Always (when the split has targets):
  density targets: mae, bce, iou (material = density >= 0.5), volume_error
  field targets:   mse, rel_l2 (per sample ||pred - y|| / ||y||; median and 90th percentile)
Domain checks, chosen by the dataset's task:
  topopt          compliance of the predicted density by FEM / compliance of the SIMP design
                  (median, share <= 1.05, share > 2, worst), and floating material: material
                  (density >= 0.5) not joined through element edges to any supported element
  fea             FEM results on designs (simp-fea): relative L2 per channel, the peak von Mises stress
                  error (|max pred - max truth| / max truth, as the literature reports) and the compliance
                  from the predicted energies (sum of ``energy``) against the FEM compliance. For nodal
                  displacements (target displacement) the stress and energy are computed from the predicted
                  displacement as the FEM does, and the equilibrium residual ||K u - F|| / ||F|| is added
  topodiff-level  TopoDiff's own analysis (SolidsPy FEM): CE, VFE, LD, FM as in the paper
Generative models make K designs per condition. The metrics above are of the first design (one draw,
comparable with an FNO); ``generative`` adds the diversity of the K designs and best-of-K numbers
(the best design by IoU / relative L2 against the truth, and by FEM compliance for topopt).
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
N_FIGURE_SAMPLES = 4


def _stats(v: np.ndarray, digits: int = 4) -> dict:
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), digits), "median": round(float(np.median(v)), digits),
            "p90": round(float(np.percentile(v, 90)), digits), "max": round(float(v.max()), digits)}


def _iou(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(pred)
    pb, yb = pred >= 0.5, y >= 0.5
    inter = (pb & yb).reshape(n, -1).sum(1)
    return inter / np.maximum((pb | yb).reshape(n, -1).sum(1), 1)


def _rel_l2(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(pred)
    err = (pred - y).reshape(n, -1)
    return np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(y.reshape(n, -1), axis=1), 1e-12)


def generic_metrics(pred: np.ndarray, y: np.ndarray, y_kind: str, y_channels=None) -> dict:
    n = len(pred)
    if y_kind == "density":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).reshape(n, -1).mean(1)
        return {"mae": _stats(np.abs(pred - y).reshape(n, -1).mean(1)), "bce": _stats(bce),
                "iou": _stats(_iou(pred, y)),
                "volume_error": _stats(np.abs(pred.reshape(n, -1).mean(1) - y.reshape(n, -1).mean(1)))}
    out = {"mse": _stats(((pred - y).reshape(n, -1) ** 2).mean(1), 6), "rel_l2": _stats(_rel_l2(pred, y))}
    if y_channels and len(y_channels) > 1:       # channels of different units: also one by one
        out["rel_l2_by_channel"] = {c: _stats(_rel_l2(pred[:, j], y[:, j])) for j, c in enumerate(y_channels)}
    return out


def fea_metrics(pred: np.ndarray, y: np.ndarray, y_channels: list[str], compliance=None) -> dict:
    """Peak von Mises stress error and the compliance from the predicted element energies."""
    out = {}
    n = len(pred)
    if "von_mises" in y_channels:
        j = y_channels.index("von_mises")
        pk, tk = pred[:, j].reshape(n, -1).max(1), y[:, j].reshape(n, -1).max(1)
        out["peak_stress_error"] = _stats(np.abs(pk - tk) / np.maximum(tk, 1e-12))
    if "energy" in y_channels and compliance is not None:
        c = np.asarray(compliance, float)
        out["compliance_error"] = _stats(np.abs(pred[:, y_channels.index("energy")].reshape(n, -1).sum(1) - c) / c)
    return out


def diversity(samples: np.ndarray, y_kind: str) -> np.ndarray:
    """How different the K designs of one condition are, per case (mean over pairs).
    density: share of cells where two designs disagree on material (>= 0.5); field: relative L2 between two."""
    n, k = samples.shape[:2]
    flat = samples.reshape(n, k, -1)
    vals = []
    for a in range(k):
        for b in range(a + 1, k):
            if y_kind == "density":
                vals.append(((flat[:, a] >= 0.5) != (flat[:, b] >= 0.5)).mean(1))
            else:
                scale = 0.5 * (np.linalg.norm(flat[:, a], axis=1) + np.linalg.norm(flat[:, b], axis=1))
                vals.append(np.linalg.norm(flat[:, a] - flat[:, b], axis=1) / np.maximum(scale, 1e-12))
    return np.mean(vals, axis=0) if vals else np.zeros(n)


def best_of_k(samples: np.ndarray, y: np.ndarray, y_kind: str) -> dict:
    """The best of the K designs against the truth (an upper bound of what picking among them gives)."""
    k = samples.shape[1]
    if y_kind == "density":
        return {"iou": _stats(np.max([_iou(samples[:, j], y) for j in range(k)], axis=0))}
    return {"rel_l2": _stats(np.min([_rel_l2(samples[:, j], y) for j in range(k)], axis=0))}


def _compliance_stats(r: np.ndarray) -> dict:
    return {**_stats(r), "share_le_1.05": round(float(np.mean(r <= 1.05)), 3),
            "share_gt_2": round(float(np.mean(r > 2)), 3)}


def support_elements(fix: np.ndarray) -> np.ndarray:
    """Elements touching a fixed node, from the node image ``(2, nely+1, nelx+1)``."""
    f = fix.max(0) > 0
    return f[:-1, :-1] | f[1:, :-1] | f[:-1, 1:] | f[1:, 1:]


def floating_volume(design: np.ndarray, supported: np.ndarray) -> float:
    """Share of the material (density >= 0.5) not joined to a supported element through element edges
    (elements meeting only at a corner are hinged there, so they do not count as joined). 0 = none floats;
    an empty design counts as 1."""
    from scipy import ndimage

    solid = design >= 0.5
    total = int(solid.sum())
    if total == 0:
        return 1.0
    lab, _ = ndimage.label(solid)                     # 4-connectivity: shared edges only
    held = np.unique(lab[solid & supported])
    return float((solid & ~np.isin(lab, held[held > 0])).sum() / total)


def _floating_stats(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"share": round(float(np.mean(v > 0)), 3), "share_gt_1pct": round(float(np.mean(v > 0.01)), 3),
            "volume_mean": round(float(v.mean()), 4), "volume_max": round(float(v.max()), 4)}


def topopt_floating(ds: Dataset, idx: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Floating share of the material for every design, (N, K)."""
    out = np.zeros(samples.shape[:2])
    for i, case in enumerate(idx):
        sup = support_elements(ds.aux["fix"][case])
        for j in range(samples.shape[1]):
            out[i, j] = floating_volume(samples[i, j, 0], sup)
    return out


def fea_nodal_metrics(ds: Dataset, idx: np.ndarray, pred: np.ndarray) -> dict:
    """Stress, energy and compliance from predicted nodal displacements, against the same from the FEM's."""
    import torch

    from .fem import FEM, node_image_to_dofs
    from .physics import Equilibrium

    g, ph = ds.meta["grid"], ds.meta["physics"]
    fem = FEM(g["nelx"], g["nely"], ph["penal"], ph["emin"], ph["nu"])
    n = len(idx)
    vm_p, vm_t, en_p, en_t = (np.zeros((n, g["nely"], g["nelx"])) for _ in range(4))
    for k, i in enumerate(idx):
        xvec = fem.img_to_vec(ds.aux["density"][i]).astype(float)
        fp = fem.fields_from_displacement(xvec, node_image_to_dofs(pred[k]))
        ft = fem.fields_from_displacement(xvec, node_image_to_dofs(ds.y[i]))
        vm_p[k], vm_t[k], en_p[k], en_t[k] = fp["von_mises"], ft["von_mises"], fp["energy"], ft["energy"]
    out = {"derived_rel_l2": {"von_mises": _stats(_rel_l2(vm_p, vm_t)), "energy": _stats(_rel_l2(en_p, en_t))}}
    out.update(fea_metrics(np.stack([vm_p, en_p], 1), np.stack([vm_t, en_t], 1), ["von_mises", "energy"],
                           ds.aux["compliance"][idx]))
    eq = Equilibrium(ph["penal"], ph["emin"], ph["nu"])
    eq.KE = eq.KE.double()
    T = lambda a: torch.from_numpy(np.asarray(a, np.float64))
    r = eq.residual(T(pred), T(ds.aux["density"][idx]), T(ds.aux["fix"][idx]), T(ds.aux["load"][idx])).numpy()
    out["equilibrium_residual"] = _stats(r)
    return out


def topopt_ratios(ds: Dataset, idx: np.ndarray, samples: np.ndarray, progress) -> np.ndarray:
    """Compliance by FEM / SIMP compliance for every design, (N, K)."""
    from .fem import FEM, node_masks_to_dofs

    g = ds.meta["grid"]
    fem = FEM(g["nelx"], g["nely"])
    n, k = samples.shape[:2]
    ratios = np.zeros((n, k))
    for i, case in enumerate(idx):
        fixed, F = node_masks_to_dofs(ds.aux["fix"][case], ds.aux["load"][case])
        for j in range(k):
            ratios[i, j] = fem.compliance(samples[i, j, 0], F, fixed) / float(ds.aux["compliance"][case])
        if (i + 1) % 10 == 0:
            progress(0.5 + 0.45 * (i + 1) / n, f"FEM check {(i + 1) * k}/{n * k}")
    return ratios


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


def topodiff_cases(ds: Dataset, pred: np.ndarray, work: Path, progress, lo: float = 0.5,
                   hi: float = 0.95) -> dict:
    """Per case CE / LD / VFE / FM with TopoDiff's analysis code (its load-node bug avoided, see docs)."""
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
                progress(lo + (hi - lo) * (i + 1) / len(img), f"TopoDiff analysis {i + 1}/{len(img)}")
    finally:
        os.chdir(cwd)
    return {"ce": np.asarray(ce), "ld": np.asarray(ld), "vfe": np.asarray(vfe), "fm": np.asarray(fm)}


def _topodiff_stats(c: dict) -> dict:
    ce = c["ce"]
    return {"CE_mean": round(float(ce.mean()), 4), "CE_median": round(float(np.median(ce)), 4),
            "CE_share_gt_0.3": round(float(np.mean(ce > 0.3)), 3),
            "VFE_mean": round(float(np.mean(c["vfe"])), 4), "LD_share": round(float(np.mean(c["ld"])), 3),
            "FM_share": round(float(np.mean(c["fm"])), 3), "cases": len(ce)}


def topodiff_analysis(ds: Dataset, pred: np.ndarray, work: Path, progress) -> dict:
    return {"topodiff": _topodiff_stats(topodiff_cases(ds, pred, work, progress))}


def evaluate(bundle: Bundle, ds: Dataset, split: str, out_dir: str | Path,
             progress: Callable[[float, str], None], samples: int = 0, steps: int = 0) -> dict:
    """``samples`` / ``steps``: generative models only (0 = the model spec's ``sample``)."""
    out = Path(out_dir)
    idx = ds.indices(split)
    if len(idx) == 0:
        raise ValueError(f"the dataset has no {split} samples")
    y = ds.y[idx] if ds.y is not None else None
    y_kind = ds.meta["y_kind"]
    metrics: dict = {"split": split, "cases": int(len(idx)), "model": str(bundle.folder),
                     "model_type": bundle.kind, "inputs": bundle.inputs}
    if bundle.generative:
        s = bundle.sampling(samples, steps)
        progress(0.05, f"sampling {len(idx)} {split} cases x {s['samples']}")
        S = bundle.sample(ds.x[idx], ds.x_channels, samples, steps,
                          progress=lambda r, m: progress(0.05 + 0.45 * r, m))
        metrics["generative"] = dict(s)
    else:
        progress(0.05, f"predicting {len(idx)} {split} cases")
        S = bundle.predict(ds.x[idx], ds.x_channels)[:, None]
    pred, k = S[:, 0], S.shape[1]
    if y is not None:
        metrics.update(generic_metrics(pred, y, y_kind, ds.meta["y_channels"]))
    if bundle.generative and k > 1:
        metrics["generative"]["diversity"] = _stats(diversity(S, y_kind))
        if y is not None:
            metrics["generative"]["best_of_k"] = best_of_k(S, y, y_kind)
    notes = None
    task = ds.meta.get("task")
    if task == "topopt" and "compliance" in ds.aux:
        R = topopt_ratios(ds, idx, S, progress)
        FL = topopt_floating(ds, idx, S)
        metrics["compliance_ratio"] = _compliance_stats(R[:, 0])
        metrics["floating"] = _floating_stats(FL[:, 0])
        if k > 1:
            metrics["generative"].setdefault("best_of_k", {})["compliance_ratio"] = _compliance_stats(R.min(1))
            metrics["generative"]["compliance_ratio_all"] = _compliance_stats(R.ravel())
            metrics["generative"]["floating_all"] = _floating_stats(FL.ravel())
        notes = [[f"C x{r:.2f}" + (" float" if f > 0 else "") for r, f in zip(row[:N_FIGURE_SAMPLES], frow)]
                 for row, frow in zip(R[:N_FIGURE], FL[:N_FIGURE])]
    elif task == "fea" and y is not None:
        if ds.meta.get("grid", {}).get("on") == "nodes" and "density" in ds.aux:
            metrics.update(fea_nodal_metrics(ds, idx, pred))
        else:
            metrics.update(fea_metrics(pred, y, ds.meta["y_channels"], ds.aux["compliance"][idx]
                                       if "compliance" in ds.aux else None))
    elif task == "topodiff-level":
        per = [topodiff_cases(ds, S[:, j], out, progress, 0.5 + 0.45 * j / k, 0.5 + 0.45 * (j + 1) / k)
               for j in range(k)]
        metrics["topodiff"] = _topodiff_stats(per[0])
        if k > 1:
            ce = np.min([c["ce"] for c in per], axis=0)
            metrics["generative"]["best_of_k"] = {"CE_mean": round(float(ce.mean()), 4),
                                                  "CE_median": round(float(np.median(ce)), 4)}
    arrays = {"index": idx, "pred": pred.astype(np.float32)}
    if bundle.generative:
        arrays["samples"] = S.astype(np.float32)
    np.savez_compressed(out / "predictions.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    n_fig = min(N_FIGURE, len(idx))
    shown = S[:n_fig, :N_FIGURE_SAMPLES] if bundle.generative else pred[:n_fig]
    if notes is not None and not bundle.generative:
        notes = [row[0] for row in notes]
    plots.compare(out / "compare.png", ds.x[idx[:n_fig]], ds.x_channels, shown, None if y is None else y[:n_fig],
                  ds.meta["y_channels"], y_kind, f"{split}: first {n_fig} cases ({Path(bundle.folder).name})", notes)
    return metrics
