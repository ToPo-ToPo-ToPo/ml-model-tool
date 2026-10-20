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
            *evaluation.headline(m),
            "", figure, tr("件ごとの数値: cases.csv。ほかの評価と並べるには compare --evals <評価ジョブ>,<評価ジョブ>",
                           "Per-case numbers: cases.csv. To put evaluations side by side: compare --evals <eval job>,<eval job>")])
        summary = {"split": split, "model_type": bundle.kind,
                   **{k: v for k, v in evaluation.key_numbers(m).items() if v is not None}}
        return job.done(summary,
                        {"summary": "summary.md", "metrics": "metrics.json", "compare": "compare.png",
                         "cases": "cases.csv", "predictions": "predictions.npz"})

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


def run_compare(evals: str, base: str, out: str, names: list | None = None, by: str = "", channel: str = "") -> dict:
    """``evals``: evaluate job folders separated by commas, relative to the work folder ``base``."""
    from ml_model_tool import compare

    job = Job(out, "compare")

    def body() -> dict:
        folders = compare.parse_evals(evals, Path(base))
        job.progress(0.1, f"comparing {len(folders)} evaluations")
        doc = compare.compare(folders, job.out, names, by, channel)
        names_ = [r["name"] for r in doc["runs"]]
        lines = [f"- {r['name']}: `{_rel(r['evaluation'], job.out)}`" for r in doc["runs"]]
        lines += [f"- warning: {w}" for w in doc["warnings"]]
        lines += ["", tr("↓ は小さいほど良い、↑ は大きいほど良い。太字が行の最良。",
                         "↓ lower is better, ↑ higher is better; bold is the best of the row."), "", doc["markdown"], ""]
        if doc.get("figure_channel"):
            lines.append(tr(f"図: compare.png（同じ件を並べる。{doc['figure_channel']}、各モデルの 1 案目）。",
                            f"Figure: compare.png (the same cases side by side; {doc['figure_channel']}, sample 1 of each model)."))
        if doc["groups"] and not by:
            lines.append(tr(f"内訳: --by {' / '.join(doc['groups'])}", f"Breakdown: --by {' / '.join(doc['groups'])}"))
        _write_summary(job.out, tr("比較", "Comparison"), lines)
        summary = {"names": names_, "best": doc["best"], "table": doc["table"]}
        if doc["warnings"]:
            summary["warnings"] = doc["warnings"]
        if doc["groups"]:
            summary["groups"] = doc["groups"]
        return job.done(summary, {"summary": "summary.md", "table": "compare.md", "metrics": "compare.json",
                                  "csv": "compare.csv", "compare": "compare.png"})

    return _guard(job, body)
