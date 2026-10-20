"""Evaluation of a trained model on one split of a dataset.

Always (when the split has targets):
  density targets: mae, bce, iou (material = density >= 0.5), volume_error
  field targets:   mse, rel_l2 (per sample ||pred - y|| / ||y||; median and 90th percentile)
Domain checks: one ``Task`` per kind of dataset (``dataset.json``'s ``task``), registered in ``TASKS``:
  topopt          compliance of the predicted density by FEM / compliance of the SIMP design
                  (median, share <= 1.05, share > 2, worst), and floating material: material
                  (density >= 0.5) not joined through element edges to any supported element
  topopt-am       the same against the self-supporting design (optopus-overhang), and overhang: the share of
                  the material that cannot be printed from the base plate (``overhang.unprintable``),
                  also of the input design (``overhang_input``, what the model has to fix)
  fea             FEM results (simp-fea / fea-elastic): relative L2 per channel, the peak von Mises stress
                  error (|max pred - max truth| / max truth, as the literature reports) and the compliance
                  from the predicted energies (sum of ``energy``) against the FEM compliance. For nodal
                  displacements (target displacement) the stress and energy are computed from the predicted
                  displacement as the FEM does, and the equilibrium residual ||K u - F|| / ||F|| is added
  topodiff-level  TopoDiff's own analysis (SolidsPy FEM): CE, VFE, LD, FM as in the paper
A task gives the numbers (``metrics.json``), one value per case (``cases.csv``), notes on the figure,
the summary lines and the key numbers that ``compare`` puts side by side. A new problem = a new ``Task``.

Generative models make K designs per condition. The metrics above are of the first design (one draw,
comparable with an FNO); ``generative`` adds the diversity of the K designs and best-of-K numbers
(the best design by IoU / relative L2 against the truth, and by FEM compliance for topopt).
Writes metrics.json, cases.csv, predictions.npz and compare.png (the first 8 cases of the split).
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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


def generic_cases(pred: np.ndarray, y: np.ndarray, y_kind: str, y_channels=None) -> dict:
    """One value per case for every generic metric (``rel_l2/<channel>`` per channel of a field)."""
    n = len(pred)
    if y_kind == "density":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        return {"iou": _iou(pred, y), "mae": np.abs(pred - y).reshape(n, -1).mean(1),
                "volume_error": np.abs(pred.reshape(n, -1).mean(1) - y.reshape(n, -1).mean(1)),
                "bce": -(y * np.log(p) + (1 - y) * np.log(1 - p)).reshape(n, -1).mean(1)}
    out = {"rel_l2": _rel_l2(pred, y), "mse": ((pred - y).reshape(n, -1) ** 2).mean(1)}
    if y_channels and len(y_channels) > 1:       # channels of different units: also one by one
        out.update({f"rel_l2/{c}": _rel_l2(pred[:, j], y[:, j]) for j, c in enumerate(y_channels)})
    return out


def generic_metrics(cases: dict) -> dict:
    out, by = {}, {}
    for name, v in cases.items():
        if name.startswith("rel_l2/"):
            by[name.split("/", 1)[1]] = _stats(v)
        else:
            out[name] = _stats(v, 6 if name == "mse" else 4)
    if by:
        out["rel_l2_by_channel"] = by
    return out


def fea_cases(pred: np.ndarray, y: np.ndarray, y_channels: list[str], compliance=None) -> dict:
    """Peak von Mises stress error and the compliance from the predicted element energies, per case."""
    out = {}
    n = len(pred)
    if "von_mises" in y_channels:
        j = y_channels.index("von_mises")
        pk, tk = pred[:, j].reshape(n, -1).max(1), y[:, j].reshape(n, -1).max(1)
        out["peak_stress_error"] = np.abs(pk - tk) / np.maximum(tk, 1e-12)
    if "energy" in y_channels and compliance is not None:
        c = np.asarray(compliance, float)
        out["compliance_error"] = np.abs(pred[:, y_channels.index("energy")].reshape(n, -1).sum(1) - c) / c
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
    """The best of the K designs against the truth, per case (an upper bound of what picking among them gives)."""
    k = samples.shape[1]
    if y_kind == "density":
        return {"iou": np.max([_iou(samples[:, j], y) for j in range(k)], axis=0)}
    return {"rel_l2": np.min([_rel_l2(samples[:, j], y) for j in range(k)], axis=0)}


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


def fea_nodal_cases(ds: Dataset, idx: np.ndarray, pred: np.ndarray) -> dict:
    """Stress, energy and compliance from predicted nodal displacements against the same from the FEM's,
    and the equilibrium residual, per case."""
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
    out = {"derived_rel_l2/von_mises": _rel_l2(vm_p, vm_t), "derived_rel_l2/energy": _rel_l2(en_p, en_t)}
    out.update(fea_cases(np.stack([vm_p, en_p], 1), np.stack([vm_t, en_t], 1), ["von_mises", "energy"],
                         ds.aux["compliance"][idx]))
    eq = Equilibrium(ph["penal"], ph["emin"], ph["nu"])
    eq.KE = eq.KE.double()
    T = lambda a: torch.from_numpy(np.asarray(a, np.float64))
    out["equilibrium_residual"] = eq.residual(T(pred), T(ds.aux["density"][idx]), T(ds.aux["fix"][idx]),
                                              T(ds.aux["load"][idx])).numpy()
    return out


def _overhang_stats(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"share": round(float(np.mean(v > 0)), 3), "share_gt_1pct": round(float(np.mean(v > 0.01)), 3),
            "volume_mean": round(float(v.mean()), 4), "volume_max": round(float(v.max()), 4)}


def topopt_overhang(ds: Dataset, idx: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Share of the material that cannot be printed, for every design, (N, K)."""
    from .overhang import overhang_volume

    dirs = ds.meta["build_directions"]
    out = np.zeros(samples.shape[:2])
    for i, case in enumerate(idx):
        for j in range(samples.shape[1]):
            out[i, j] = overhang_volume(samples[i, j, 0], dirs[int(ds.aux["build"][case])])
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




# ---------------------------------------------------------------------------
# domain checks, one per kind of dataset
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """What a task sees: the dataset, the evaluated rows ``idx``, the designs ``S`` (N, K, C, H, W)
    (K = 1 for an FNO), the targets ``y`` (or None) and where to write."""

    ds: Dataset
    idx: np.ndarray
    S: np.ndarray
    y: Optional[np.ndarray]
    out: Path
    progress: Callable[[float, str], None]

    @property
    def pred(self) -> np.ndarray:
        return self.S[:, 0]

    @property
    def k(self) -> int:
        return self.S.shape[1]


@dataclass
class Result:
    metrics: dict                       # into metrics.json (``generative`` merged into metrics["generative"])
    cases: dict                         # name -> one value per case (into cases.csv)
    notes: Optional[list] = None        # per case, per design: a short note on compare.png
    generative: Optional[dict] = None


class Task:
    """Domain checks for one kind of dataset. Subclass, then register in ``TASKS`` under the dataset's task."""

    def evaluate(self, c: Context) -> Result:
        return Result({}, {})

    def lines(self, m: dict) -> list[str]:
        """summary.md lines of the numbers of sample 1 (or the FNO's prediction)."""
        return []

    def lines_k(self, g: dict) -> list[str]:
        """summary.md lines of the K designs (``g`` = metrics["generative"])."""
        return []

    def numbers(self, m: dict) -> dict:
        """The key numbers (flat) for the job summary and ``compare``."""
        return {}


def _median(v) -> Optional[float]:
    return v.get("median") if isinstance(v, dict) else None


class Topopt(Task):
    """FEM compliance of the predicted designs against SIMP's, and floating material."""

    def evaluate(self, c: Context) -> Result:
        if "compliance" not in c.ds.aux:
            return Result({}, {})
        R = topopt_ratios(c.ds, c.idx, c.S, c.progress)
        FL = topopt_floating(c.ds, c.idx, c.S)
        m = {"compliance_ratio": _compliance_stats(R[:, 0]), "floating": _floating_stats(FL[:, 0])}
        cases = {"compliance_ratio": R[:, 0], "floating": FL[:, 0]}
        g = None
        if c.k > 1:
            g = {"best_of_k": {"compliance_ratio": _compliance_stats(R.min(1))},
                 "compliance_ratio_all": _compliance_stats(R.ravel()), "floating_all": _floating_stats(FL.ravel())}
            cases["best_of_k_compliance_ratio"] = R.min(1)
        notes = [[f"C x{r:.2f}" + (" float" if f > 0 else "") for r, f in zip(row, frow)] for row, frow in zip(R, FL)]
        return Result(m, cases, notes, g)

    def lines(self, m: dict) -> list[str]:
        rows = []
        if "compliance_ratio" in m:
            c = m["compliance_ratio"]
            rows.append(f"- compliance / SIMP: median {c['median']}  <=1.05 {c['share_le_1.05']:.0%}  "
                        f">2 {c['share_gt_2']:.0%}  worst {c['max']}")
        if "floating" in m:
            f = m["floating"]
            rows.append(f"- floating material: {f['share']:.0%} of designs (more than 1% of the material: "
                        f"{f['share_gt_1pct']:.0%})  mean share of the material {f['volume_mean']}")
        return rows

    def lines_k(self, g: dict) -> list[str]:
        rows, k = [], g["samples"]
        c = (g.get("best_of_k") or {}).get("compliance_ratio")
        if c:
            rows.append(f"- best of {k} by FEM: compliance / SIMP median {c['median']}  "
                        f"<=1.05 {c['share_le_1.05']:.0%}  >2 {c['share_gt_2']:.0%}")
        if "floating_all" in g:
            f = g["floating_all"]
            rows.append(f"- all {k} designs: floating material {f['share']:.0%} "
                        f"(more than 1% of the material: {f['share_gt_1pct']:.0%})")
        return rows

    def numbers(self, m: dict) -> dict:
        out = {}
        if "compliance_ratio" in m:
            c = m["compliance_ratio"]
            out.update({"compliance_ratio": c["median"], "compliance_le_1.05": c["share_le_1.05"],
                        "compliance_gt_2": c["share_gt_2"]})
        if "floating" in m:
            out.update({"floating_share": m["floating"]["share"], "floating_gt_1pct": m["floating"]["share_gt_1pct"]})
        g = m.get("generative") or {}
        b = (g.get("best_of_k") or {}).get("compliance_ratio")
        if b:
            out.update({"best_of_k_compliance_ratio": b["median"], "best_of_k_compliance_le_1.05": b["share_le_1.05"]})
        if "compliance_ratio_all" in g:
            out["all_compliance_gt_2"] = g["compliance_ratio_all"]["share_gt_2"]
        if "floating_all" in g:
            out["all_floating_share"] = g["floating_all"]["share"]
        return out


class Fea(Task):
    """FEM results: peak stress, compliance from the energies; from nodal displacements also the
    derived stress / energy and the equilibrium residual."""

    def evaluate(self, c: Context) -> Result:
        if c.y is None:
            return Result({}, {})
        ds = c.ds
        if ds.meta.get("grid", {}).get("on") == "nodes" and "density" in ds.aux:
            cases = fea_nodal_cases(ds, c.idx, c.pred)
        else:
            cases = fea_cases(c.pred, c.y, ds.meta["y_channels"],
                              ds.aux["compliance"][c.idx] if "compliance" in ds.aux else None)
        m, derived = {}, {}
        for name, v in cases.items():
            if name.startswith("derived_rel_l2/"):
                derived[name.split("/", 1)[1]] = _stats(v)
            else:
                m[name] = _stats(v)
        if derived:
            m["derived_rel_l2"] = derived
        return Result(m, cases)

    def lines(self, m: dict) -> list[str]:
        rows = [f"  - {ch} (from the predicted displacement): relative L2 median {v['median']}  p90 {v['p90']}"
                for ch, v in (m.get("derived_rel_l2") or {}).items()]
        if "peak_stress_error" in m:
            v = m["peak_stress_error"]
            rows.append(f"- peak von Mises stress error (relative): median {v['median']}  p90 {v['p90']}")
        if "compliance_error" in m:
            v = m["compliance_error"]
            rows.append(f"- compliance from the predicted energies, error (relative): median {v['median']}  p90 {v['p90']}")
        if "equilibrium_residual" in m:
            v = m["equilibrium_residual"]
            rows.append(f"- equilibrium residual ||K u - F|| / ||F||: median {v['median']}  p90 {v['p90']} "
                        "(0 for the FEM solution)")
        return rows

    def numbers(self, m: dict) -> dict:
        out = {k: m[k]["median"] for k in ("peak_stress_error", "compliance_error", "equilibrium_residual") if k in m}
        out.update({f"derived_rel_l2/{ch}": v["median"] for ch, v in (m.get("derived_rel_l2") or {}).items()})
        return out


class TopodiffLevel(Task):
    """TopoDiff's own analysis of its test levels (no target designs, only SIMP compliances)."""

    def evaluate(self, c: Context) -> Result:
        k = c.k
        per = [topodiff_cases(c.ds, c.S[:, j], c.out, c.progress, 0.5 + 0.45 * j / k, 0.5 + 0.45 * (j + 1) / k)
               for j in range(k)]
        cases = {"CE": per[0]["ce"], "VFE": per[0]["vfe"], "LD": per[0]["ld"].astype(float),
                 "FM": per[0]["fm"].astype(float)}
        g = None
        if k > 1:
            ce = np.min([p["ce"] for p in per], axis=0)
            g = {"best_of_k": {"CE_mean": round(float(ce.mean()), 4), "CE_median": round(float(np.median(ce)), 4)}}
            cases["best_of_k_CE"] = ce
        return Result({"topodiff": _topodiff_stats(per[0])}, cases, None, g)

    def lines(self, m: dict) -> list[str]:
        if "topodiff" not in m:
            return []
        t = m["topodiff"]
        return [f"- TopoDiff metrics: CE mean {t['CE_mean']:.2%} median {t['CE_median']:.2%}  "
                f"VFE {t['VFE_mean']:.2%}  LD {t['LD_share']:.1%}  FM {t['FM_share']:.1%}"]

    def lines_k(self, g: dict) -> list[str]:
        b = g.get("best_of_k") or {}
        return [f"- best of {g['samples']}: CE mean {b['CE_mean']:.2%} median {b['CE_median']:.2%}"] if "CE_mean" in b else []

    def numbers(self, m: dict) -> dict:
        t = m.get("topodiff") or {}
        out = {k: t[k] for k in ("CE_mean", "CE_median", "VFE_mean", "LD_share", "FM_share") if k in t}
        b = (m.get("generative") or {}).get("best_of_k") or {}
        if "CE_mean" in b:
            out["best_of_k_CE_mean"] = b["CE_mean"]
        return out


class TopoptAM(Topopt):
    """``Topopt`` against the self-supporting design, plus overhang: material that cannot be printed from the
    base plate, of the predicted designs and (as the baseline) of the input design."""

    def evaluate(self, c: Context) -> Result:
        res = super().evaluate(c)
        if "build" not in c.ds.aux:
            return res
        OH = topopt_overhang(c.ds, c.idx, c.S)
        res.metrics["overhang"] = _overhang_stats(OH[:, 0])
        res.cases["overhang"] = OH[:, 0]
        if "design" in c.ds.x_channels:
            design = c.ds.x[c.idx][:, [c.ds.x_channels.index("design")]][:, None]
            OI = topopt_overhang(c.ds, c.idx, design)[:, 0]
            res.metrics["overhang_input"] = _overhang_stats(OI)
            res.cases["overhang_input"] = OI
        if c.k > 1:
            res.generative = {**(res.generative or {}), "overhang_all": _overhang_stats(OH.ravel())}
        if res.notes is not None:
            res.notes = [[n + (f" OH {o:.0%}" if o > 0 else "") for n, o in zip(row, orow)]
                         for row, orow in zip(res.notes, OH)]
        return res

    def lines(self, m: dict) -> list[str]:
        rows = [r.replace("compliance / SIMP", "compliance / self design") for r in super().lines(m)]
        if "overhang" in m:
            o = m["overhang"]
            rows.append(f"- overhang (material that cannot be printed): {o['share']:.0%} of designs (more than 1% of "
                        f"the material: {o['share_gt_1pct']:.0%})  mean share of the material {o['volume_mean']}")
        if "overhang_input" in m:
            o = m["overhang_input"]
            rows.append(f"  - the input designs: {o['share']:.0%} have overhang  mean share of the material "
                        f"{o['volume_mean']}")
        return rows

    def lines_k(self, g: dict) -> list[str]:
        rows = [r.replace("compliance / SIMP", "compliance / self design") for r in super().lines_k(g)]
        if "overhang_all" in g:
            o = g["overhang_all"]
            rows.append(f"- all {g['samples']} designs: overhang {o['share']:.0%} "
                        f"(more than 1% of the material: {o['share_gt_1pct']:.0%})")
        return rows

    def numbers(self, m: dict) -> dict:
        out = super().numbers(m)
        if "overhang" in m:
            out.update({"overhang_share": m["overhang"]["share"], "overhang_gt_1pct": m["overhang"]["share_gt_1pct"],
                        "overhang_volume": m["overhang"]["volume_mean"]})
        if "overhang_input" in m:
            out["overhang_input_share"] = m["overhang_input"]["share"]
        g = m.get("generative") or {}
        if "overhang_all" in g:
            out["all_overhang_share"] = g["overhang_all"]["share"]
        return out


TASKS: dict[str, Task] = {"topopt": Topopt(), "topopt-am": TopoptAM(), "fea": Fea(),
                          "topodiff-level": TopodiffLevel()}

# which way is better, for ``compare`` (numbers not listed are shown without a pick)
LOWER_IS_BETTER = ("mae", "volume_error", "bce", "rel_l2", "mse", "compliance_ratio", "compliance_gt_2",
                   "floating_share", "floating_gt_1pct", "peak_stress_error", "compliance_error", "equilibrium_residual",
                   "best_of_k_rel_l2", "best_of_k_compliance_ratio", "all_compliance_gt_2", "all_floating_share",
                   "CE_mean", "CE_median", "VFE_mean", "LD_share", "FM_share", "best_of_k_CE_mean",
                   "overhang_share", "overhang_gt_1pct", "overhang_volume", "all_overhang_share")
HIGHER_IS_BETTER = ("iou", "compliance_le_1.05", "best_of_k_iou", "best_of_k_compliance_le_1.05")


def better(name: str) -> str:
    base = name.split("/", 1)[0]
    if base in ("rel_l2", "derived_rel_l2") or name in LOWER_IS_BETTER:
        return "low"
    if name in HIGHER_IS_BETTER:
        return "high"
    return ""


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------
def _groups(ds: Dataset, idx: np.ndarray) -> dict:
    """Per-case labels to break results down by (``compare --by``): the dataset's ``groups``."""
    out = {}
    for name, labels in (ds.meta.get("groups") or {}).items():
        if name in ds.aux:
            v = ds.aux[name][idx]
            out[name] = [labels[int(i)] for i in v] if labels else [int(i) for i in v]
    return out


def write_cases(path: Path, idx: np.ndarray, groups: dict, cases: dict) -> None:
    names = list(cases)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", *groups, *names])
        for r, i in enumerate(idx):
            w.writerow([int(i), *(g[r] for g in groups.values()),
                        *(f"{float(cases[n][r]):.6g}" for n in names)])


def evaluate(bundle: Bundle, ds: Dataset, split: str, out_dir: str | Path,
             progress: Callable[[float, str], None], samples: int = 0, steps: int = 0) -> dict:
    """``samples`` / ``steps``: generative models only (0 = the model spec's ``sample``)."""
    out = Path(out_dir)
    idx = ds.indices(split)
    if len(idx) == 0:
        raise ValueError(f"the dataset has no {split} samples")
    y = ds.y[idx] if ds.y is not None else None
    y_kind = ds.meta["y_kind"]
    metrics: dict = {"split": split, "cases": int(len(idx)), "model": str(bundle.folder), "data": str(ds.path),
                     "task": ds.meta.get("task"), "model_type": bundle.kind, "inputs": bundle.inputs}
    if bundle.generative:
        s = bundle.sampling(samples, steps)
        progress(0.05, f"sampling {len(idx)} {split} cases x {s['samples']}")
        S = bundle.sample(ds.x[idx], ds.x_channels, samples, steps,
                          progress=lambda r, m: progress(0.05 + 0.45 * r, m))
        metrics["generative"] = dict(s)
    else:
        progress(0.05, f"predicting {len(idx)} {split} cases")
        S = bundle.predict(ds.x[idx], ds.x_channels)[:, None]
    c = Context(ds, idx, S, y, out, progress)
    cases: dict = {}
    if y is not None:
        cases = generic_cases(c.pred, y, y_kind, ds.meta["y_channels"])
        metrics.update(generic_metrics(cases))
    if bundle.generative and c.k > 1:
        div = diversity(S, y_kind)
        metrics["generative"]["diversity"] = _stats(div)
        cases["diversity"] = div
        if y is not None:
            best = best_of_k(S, y, y_kind)
            metrics["generative"]["best_of_k"] = {k: _stats(v) for k, v in best.items()}
            cases.update({f"best_of_k_{k}": v for k, v in best.items()})
    task = TASKS.get(ds.meta.get("task") or "")
    notes = None
    if task is not None:
        res = task.evaluate(c)
        metrics.update(res.metrics)
        cases.update(res.cases)
        notes = res.notes
        for key, v in (res.generative or {}).items():
            g = metrics.setdefault("generative", {})
            g[key] = {**g.get(key, {}), **v} if isinstance(v, dict) and key == "best_of_k" else v
    arrays = {"index": idx, "pred": c.pred.astype(np.float32)}
    if bundle.generative:
        arrays["samples"] = S.astype(np.float32)
    np.savez_compressed(out / "predictions.npz", **arrays)
    groups = _groups(ds, idx)
    metrics["groups"] = list(groups)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    write_cases(out / "cases.csv", idx, groups, cases)
    n_fig = min(N_FIGURE, len(idx))
    shown = S[:n_fig, :N_FIGURE_SAMPLES] if bundle.generative else c.pred[:n_fig]
    if notes is not None:
        notes = [row[:N_FIGURE_SAMPLES] for row in notes[:n_fig]]
        if not bundle.generative:
            notes = [row[0] for row in notes]
    plots.compare(out / "compare.png", ds.x[idx[:n_fig]], ds.x_channels, shown, None if y is None else y[:n_fig],
                  ds.meta["y_channels"], y_kind, f"{split}: first {n_fig} cases ({Path(bundle.folder).name})", notes)
    return metrics


def _task_of(m: dict) -> Optional[Task]:
    """The task that made these metrics (metrics.json of older versions: by the numbers it leaves)."""
    if m.get("task"):
        return TASKS.get(m["task"])
    if "overhang" in m:
        return TASKS["topopt-am"]
    if "compliance_ratio" in m or "floating" in m:
        return TASKS["topopt"]
    if "topodiff" in m:
        return TASKS["topodiff-level"]
    if any(k in m for k in ("peak_stress_error", "compliance_error", "equilibrium_residual")):
        return TASKS["fea"]
    return None


def headline(m: dict) -> list[str]:
    """The numbers of an evaluation as summary.md lines."""
    rows = []
    g = m.get("generative")
    if g:
        steps = f", {g['steps']} steps" if "steps" in g else ""
        rows.append(f"- generative: {g['samples']} designs per condition{steps}; "
                    "the next lines are of sample 1 (one draw, comparable with an FNO)")
    if "iou" in m:
        rows.append(f"- IoU median {m['iou']['median']}  MAE median {m['mae']['median']}  "
                    f"volume error median {m['volume_error']['median']}")
    if "rel_l2" in m:
        rows.append(f"- relative L2 median {m['rel_l2']['median']}  p90 {m['rel_l2']['p90']}")
    for ch, v in (m.get("rel_l2_by_channel") or {}).items():
        rows.append(f"  - {ch}: relative L2 median {v['median']}  p90 {v['p90']}")
    task = _task_of(m)
    if task is not None:
        rows += task.lines(m)
    if g:
        if "diversity" in g:
            rows.append(f"- diversity (mean over pairs): median {g['diversity']['median']}")
        b = g.get("best_of_k") or {}
        if "iou" in b:
            rows.append(f"- best of {g['samples']}: IoU median {b['iou']['median']}")
        if "rel_l2" in b:
            rows.append(f"- best of {g['samples']}: relative L2 median {b['rel_l2']['median']}")
        if task is not None:
            rows += task.lines_k(g)
    return rows


def key_numbers(m: dict) -> dict:
    """The key numbers of an evaluation, flat (medians and shares): the job summary and the rows of ``compare``."""
    out: dict = {"cases": m.get("cases")}
    for name in ("iou", "mae", "volume_error", "rel_l2"):
        if name in m:
            out[name] = m[name]["median"]
    out.update({f"rel_l2/{ch}": v["median"] for ch, v in (m.get("rel_l2_by_channel") or {}).items()})
    task = _task_of(m)
    if task is not None:
        out.update(task.numbers(m))
    g = m.get("generative") or {}
    if g:
        out["samples"] = g.get("samples")
        if "diversity" in g:
            out["diversity"] = g["diversity"]["median"]
        for name in ("iou", "rel_l2"):
            v = _median((g.get("best_of_k") or {}).get(name))
            if v is not None:
                out[f"best_of_k_{name}"] = v
    return out
