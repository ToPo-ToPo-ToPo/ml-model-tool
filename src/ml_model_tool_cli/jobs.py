"""The jobs: data / train / evaluate / predict. One run = one job folder; ``status.json`` is the record.

Every job folder gets (app contract §10.2): ``summary.md`` (what was done, key numbers, next step),
``metrics.json`` and a figure (``preview.png`` / ``loss.png`` / ``compare.png``).
Progress goes to ``status.json`` (``progress.message``); the window and the brain read it from there.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from agk import jobs as agk_jobs
from agk.lang import tr

JOB_SCHEMA = "ml-model-tool-job/1"
JOBSPEC_SCHEMA = "ml-model-tool-jobspec/1"


class Job:
    """Writes status.json for one job (state, progress, summary, artifacts)."""

    def __init__(self, out: str | Path, command: str):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.status = {"schema": JOB_SCHEMA, "command": command, "state": "running", "pid": os.getpid(),
                       "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "progress": {"message": "", "ratio": 0.0},
                       "summary": {}, "error": None, "artifacts": {}}
        self._write()

    def _write(self) -> None:
        self.status["elapsed_s"] = round(time.time() - self.t0, 1)
        agk_jobs.write_status(self.out, self.status)

    def progress(self, ratio: float, message: str) -> None:
        self.status["progress"] = {"message": message, "ratio": round(float(ratio), 3)}
        self._write()
        print(f"[{self.status['command']}] {message}", file=sys.stderr, flush=True)   # log (run.out)

    def done(self, summary: dict, artifacts: dict) -> dict:
        self.status.update(state="done", summary=summary,
                           artifacts={k: v for k, v in artifacts.items() if (self.out / v).exists()})
        self.status["progress"] = {"message": tr("完了", "done"), "ratio": 1.0}
        self._write()
        return self.status

    def failed(self, stage: str, message: str) -> dict:
        self.status.update(state="failed", error={"stage": stage, "message": message})
        self._write()
        return self.status


def _write_summary(out: Path, title: str, lines: list[str]) -> None:
    (out / "summary.md").write_text(f"# {title}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _rel(p: str | Path, base: Path) -> str:
    try:
        return os.path.relpath(Path(p).resolve(), base.resolve())
    except ValueError:
        return str(p)


def _guard(job: Job, fn: Callable[[], dict]) -> dict:
    try:
        return fn()
    except (ValueError, FileNotFoundError) as exc:
        return job.failed("input", str(exc))
    except KeyboardInterrupt:
        return job.failed("stopped", tr("停止の要求で止めた", "stopped on request"))
    except Exception as exc:  # noqa: BLE001 - close the job with the reason
        import traceback

        traceback.print_exc()
        return job.failed("internal", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
def run_data(spec_doc: dict, out: str) -> dict:
    from ml_model_tool import dataset, plots, spec as specs
    from ml_model_tool.sources import SOURCES

    job = Job(out, "data")

    def body() -> dict:
        spec = specs.data_spec(spec_doc)
        (job.out / "data_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        info = SOURCES[spec["source"]].generate(spec, job.out, job.progress)
        ds = dataset.load(job.out / "dataset.npz")
        plots.samples(job.out / "preview.png", ds.x[:6], ds.x_channels, None if ds.y is None else ds.y[:6],
                      ds.meta["y_channels"], ds.meta["y_kind"], f"{spec['source']}: first samples")
        counts = ds.counts()
        (job.out / "metrics.json").write_text(json.dumps({**info, "counts": counts}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        _write_summary(job.out, tr("データセット", "Dataset") + f" ({spec['source']})", [
            f"- source: `{spec['source']}`  samples: {info['samples']}  "
            f"(train {counts['train']} / val {counts['val']} / test {counts['test']})",
            f"- x channels: {', '.join(ds.x_channels)}",
            f"- y channels: {', '.join(ds.meta['y_channels'])} ({ds.meta['y_kind']})",
            f"- shape: x {list(ds.x.shape)}" + (f", y {list(ds.y.shape)}" if ds.y is not None else ""),
            *(f"- {k}: {v}" for k, v in info.items() if k not in ("samples",)),
            "", tr("次: このフォルダ（または dataset.npz）を train --data に渡す。",
                   "Next: pass this folder (or dataset.npz) to train --data.")])
        arts = {"dataset": "dataset.npz", "dataset_info": "dataset.json", "summary": "summary.md",
                "metrics": "metrics.json", "preview": "preview.png"}
        for f in sorted(job.out.glob("level_*.npz")):
            arts[f.stem] = f.name
        return job.done({"samples": info["samples"], **counts, "x_channels": ds.x_channels,
                         "y_channels": ds.meta["y_channels"]}, arts)

    return _guard(job, body)


def run_train(spec_doc: dict, data: str, out: str) -> dict:
    from ml_model_tool import dataset, spec as specs, training

    job = Job(out, "train")

    def body() -> dict:
        ds = dataset.load(data)
        spec = specs.model_spec(spec_doc, ds.x_channels, ds.meta["y_kind"],
                                grid=None if ds.y is None else tuple(ds.y.shape[2:]))
        (job.out / "model_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
        job.progress(0.0, tr(f"学習の準備（{ds.meta['source']}・入力 {len(spec['inputs'])} チャネル）",
                             f"preparing ({ds.meta['source']}, {len(spec['inputs'])} input channels)"))
        res = training.train(spec, ds, job.out, job.progress)
        (job.out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        hint = (tr("検証の損失が学習の損失よりかなり大きい（過学習）。データを増やすか、入力・モデルを見直す。",
                   "validation loss is much larger than training loss (overfitting): add data or revisit inputs / model.")
                if (res["overfit_ratio"] or 0) > 2 else "")
        generative = "sample" in spec
        nxt = (tr("次: このフォルダを evaluate --model に渡す（--data は同じデータ）。1 条件あたり "
                  f"{spec['sample']['samples']} 案を作って比べる（--samples / --steps で変えられる）。",
                  "Next: pass this folder to evaluate --model (with the same --data); it makes "
                  f"{spec['sample']['samples']} designs per condition (change with --samples / --steps).")
               if generative else
               tr("次: このフォルダを evaluate --model に渡す（--data は同じデータ）。",
                  "Next: pass this folder to evaluate --model (with the same --data)."))
        _write_summary(job.out, tr("学習", "Training"), [
            f"- data: `{_rel(ds.path, job.out)}` ({ds.meta['source']})  train {res['n_train']} / val {res['n_val']}",
            f"- model: {res['model_type']} ({res['parameters']:,} parameters)  device: {res['device']}",
            f"- inputs: {', '.join(res['inputs'])}  loss: {res['loss']}",
            f"- epochs: {res['epochs']}  minutes: {res['minutes']}",
            f"- loss (last): train {res['train_loss_last']:.4g} / val {res['val_loss_last']:.4g}",
            (f"- best val {res['val_loss_best']:.4g} at epoch {res['best_epoch']} (saved as model.mdlus)"
             if res["saved_epoch"] == res["best_epoch"] else
             f"- best val {res['val_loss_best']:.4g} at epoch {res['best_epoch']}; the last epoch "
             f"{res['saved_epoch']} is saved as model.mdlus (GAN losses do not rank epochs)"),
            *( [f"- {hint}"] if hint else []),
            "", nxt])
        return job.done({k: res[k] for k in ("model_type", "epochs", "n_train", "val_loss_best", "best_epoch",
                                             "overfit_ratio", "minutes")},
                        {"model": "model.mdlus", "model_info": "model.json", "history": "history.csv",
                         "loss_plot": "loss.png", "summary": "summary.md", "metrics": "metrics.json"})

    return _guard(job, body)


def _headline(m: dict) -> list[str]:
    rows = []
    if "iou" in m:
        rows.append(f"- IoU median {m['iou']['median']}  MAE median {m['mae']['median']}  "
                    f"volume error median {m['volume_error']['median']}")
    if "rel_l2" in m:
        rows.append(f"- relative L2 median {m['rel_l2']['median']}  p90 {m['rel_l2']['p90']}")
    for c, v in (m.get("rel_l2_by_channel") or {}).items():
        rows.append(f"  - {c}: relative L2 median {v['median']}  p90 {v['p90']}")
    for c, v in (m.get("derived_rel_l2") or {}).items():
        rows.append(f"  - {c} (from the predicted displacement): relative L2 median {v['median']}  p90 {v['p90']}")
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
    if "compliance_ratio" in m:
        c = m["compliance_ratio"]
        rows.append(f"- compliance / SIMP: median {c['median']}  <=1.05 {c['share_le_1.05']:.0%}  "
                    f">2 {c['share_gt_2']:.0%}  worst {c['max']}")
    if "floating" in m:
        f = m["floating"]
        rows.append(f"- floating material: {f['share']:.0%} of designs (more than 1% of the material: "
                    f"{f['share_gt_1pct']:.0%})  mean share of the material {f['volume_mean']}")
    if "overhang" in m:
        o = m["overhang"]
        rows.append(f"- overhang (material that cannot be printed): {o['share']:.0%} of designs (more than 1% of "
                    f"the material: {o['share_gt_1pct']:.0%})  mean share of the material {o['volume_mean']}")
    if "overhang_input" in m:
        o = m["overhang_input"]
        rows.append(f"  - the input designs: {o['share']:.0%} have overhang  mean share of the material "
                    f"{o['volume_mean']}")
    if "topodiff" in m:
        t = m["topodiff"]
        rows.append(f"- TopoDiff metrics: CE mean {t['CE_mean']:.2%} median {t['CE_median']:.2%}  "
                    f"VFE {t['VFE_mean']:.2%}  LD {t['LD_share']:.1%}  FM {t['FM_share']:.1%}")
    g = m.get("generative")
    if g:
        steps = f", {g['steps']} steps" if "steps" in g else ""
        rows.insert(0, f"- generative: {g['samples']} designs per condition{steps}; "
                       "the next lines are of sample 1 (one draw, comparable with an FNO)")
        if "diversity" in g:
            rows.append(f"- diversity (mean over pairs): median {g['diversity']['median']}")
        b = g.get("best_of_k") or {}
        if "iou" in b:
            rows.append(f"- best of {g['samples']}: IoU median {b['iou']['median']}")
        if "rel_l2" in b:
            rows.append(f"- best of {g['samples']}: relative L2 median {b['rel_l2']['median']}")
        if "compliance_ratio" in b:
            c = b["compliance_ratio"]
            rows.append(f"- best of {g['samples']} by FEM: compliance / SIMP median {c['median']}  "
                        f"<=1.05 {c['share_le_1.05']:.0%}  >2 {c['share_gt_2']:.0%}")
        if "floating_all" in g:
            f = g["floating_all"]
            rows.append(f"- all {g['samples']} designs: floating material {f['share']:.0%} "
                        f"(more than 1% of the material: {f['share_gt_1pct']:.0%})")
        if "overhang_all" in g:
            o = g["overhang_all"]
            rows.append(f"- all {g['samples']} designs: overhang {o['share']:.0%} "
                        f"(more than 1% of the material: {o['share_gt_1pct']:.0%})")
        if "CE_mean" in b:
            rows.append(f"- best of {g['samples']}: CE mean {b['CE_mean']:.2%} median {b['CE_median']:.2%}")
    return rows


def run_evaluate(model: str, data: str, split: str, out: str, samples: int = 0, steps: int = 0) -> dict:
    from ml_model_tool import dataset, evaluation
    from ml_model_tool.model import Bundle

    job = Job(out, "evaluate")

    def body() -> dict:
        ds = dataset.load(data)
        bundle = Bundle(model)
        m = evaluation.evaluate(bundle, ds, split, job.out, job.progress, samples, steps)
        figure = (tr("図: compare.png（先頭 8 件。正解と生成した案、最大 4 案）。",
                     "Figure: compare.png (first 8 cases, truth and up to 4 generated designs).")
                  if bundle.generative else
                  tr("図: compare.png（先頭 8 件。正解と予測）。", "Figure: compare.png (first 8 cases, truth and prediction)."))
        _write_summary(job.out, tr("評価", "Evaluation"), [
            f"- model: `{_rel(bundle.folder, job.out)}` ({bundle.kind})  data: `{_rel(ds.path, job.out)}`  "
            f"split: {split} ({m['cases']} cases)",
            *_headline(m),
            "", figure])
        keep = {k: m[k] for k in ("cases", "iou", "rel_l2", "compliance_ratio", "peak_stress_error",
                                  "compliance_error", "equilibrium_residual", "topodiff") if k in m}
        if "floating" in m:
            keep["floating_share"] = m["floating"]["share"]
        if "overhang" in m:
            keep["overhang_share"] = m["overhang"]["share"]
        summary = {"split": split, "model_type": bundle.kind,
                   **{k: (v["median"] if isinstance(v, dict) and "median" in v else v) for k, v in keep.items()}}
        g = m.get("generative") or {}
        if g:
            summary["samples"] = g["samples"]
            if "diversity" in g:
                summary["diversity"] = g["diversity"]["median"]
            summary.update({f"best_of_k_{k}": v["median"] for k, v in (g.get("best_of_k") or {}).items()
                            if isinstance(v, dict) and "median" in v})
        return job.done(summary,
                        {"summary": "summary.md", "metrics": "metrics.json", "compare": "compare.png",
                         "predictions": "predictions.npz"})

    return _guard(job, body)


def run_predict(model: str, data: str, split: str, out: str, samples: int = 0, steps: int = 0) -> dict:
    from ml_model_tool import dataset, plots
    from ml_model_tool.model import Bundle
    import numpy as np

    job = Job(out, "predict")

    def body() -> dict:
        ds = dataset.load(data)
        bundle = Bundle(model)
        idx = np.arange(len(ds.x)) if split == "all" else ds.indices(split)
        if len(idx) == 0:
            raise ValueError(f"the dataset has no {split} samples")
        k = min(8, len(idx))
        lines = [f"- model: `{_rel(bundle.folder, job.out)}` ({bundle.kind})  data: `{_rel(ds.path, job.out)}`  "
                 f"{split}: {len(idx)} cases"]
        if bundle.generative:
            s = bundle.sampling(samples, steps)
            job.progress(0.05, f"sampling {len(idx)} cases x {s['samples']}")
            S = bundle.sample(ds.x[idx], ds.x_channels, samples, steps,
                              progress=lambda r, m: job.progress(0.05 + 0.9 * r, m))
            pred = S[:, 0]
            np.savez_compressed(job.out / "predictions.npz", index=idx, pred=pred.astype(np.float32),
                                samples=S.astype(np.float32))
            plots.compare(job.out / "preview.png", ds.x[idx[:k]], ds.x_channels, S[:k, :4], None,
                          ds.meta["y_channels"], ds.meta["y_kind"], f"generated designs ({split})")
            lines += [f"- {s['samples']} designs per condition ("
                      + ", ".join(f"{k} {v}" for k, v in s.items() if k != "samples") + ")",
                      f"- predictions.npz: index (row in the dataset), samples {list(S.shape)} "
                      f"(case, design, {', '.join(ds.meta['y_channels'])}), pred = samples[:, 0]"]
        else:
            job.progress(0.1, f"predicting {len(idx)} cases")
            pred = bundle.predict(ds.x[idx], ds.x_channels)
            np.savez_compressed(job.out / "predictions.npz", index=idx, pred=pred.astype(np.float32))
            plots.compare(job.out / "preview.png", ds.x[idx[:k]], ds.x_channels, pred[:k], None,
                          ds.meta["y_channels"], ds.meta["y_kind"], f"predictions ({split})")
            lines.append(f"- predictions.npz: index (row in the dataset), pred {list(pred.shape)} "
                         f"({', '.join(ds.meta['y_channels'])})")
        _write_summary(job.out, tr("予測", "Prediction"), lines)
        return job.done({"cases": int(len(idx)), "model_type": bundle.kind,
                         **({"samples": bundle.sampling(samples, steps)["samples"]} if bundle.generative else {})},
                        {"predictions": "predictions.npz", "preview": "preview.png", "summary": "summary.md"})

    return _guard(job, body)
