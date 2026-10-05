"""Data sources: each turns a data spec into the standard dataset (``ml_model_tool.dataset``).

A source is a module with
    ``SHAPE``     the keys its data spec accepts (``agk.shape``; unknown keys are rejected)
    ``DEFAULTS``  default values of those keys
    ``check(spec)``  argument checks that need more than key names (raise ValueError)
    ``generate(spec, out_dir, progress) -> dict``  writes dataset.npz / dataset.json, returns a summary
"""
from __future__ import annotations

from . import fea_elastic, simp_topopt, topodiff

SOURCES = {
    "simp-topopt": simp_topopt,
    "fea-elastic": fea_elastic,
    "topodiff": topodiff,
}
