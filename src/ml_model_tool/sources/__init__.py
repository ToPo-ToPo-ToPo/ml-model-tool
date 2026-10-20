"""Data sources: each turns a data spec into the standard dataset (``ml_model_tool.dataset``).

A source is a module with
    ``ABOUT``       (ja, en) one line: what the data is (shown by ``spec data``)
    ``KEYS``        its data spec keys (``keys.Key``: default, values, meaning)
    ``X_CHANNELS`` / ``Y_CHANNELS``  the input and target channels
    ``check(spec)`` optional: checks across keys (raise ``keys.SpecError``)
    ``generate(spec, out_dir, progress) -> dict``  writes dataset.npz / dataset.json, returns a summary
Adding a source = one module here and one line in ``SOURCES``.
"""
from __future__ import annotations

from . import fea_elastic, simp_fea, simp_topopt, topodiff

SOURCES = {
    "simp-topopt": simp_topopt,
    "simp-fea": simp_fea,
    "fea-elastic": fea_elastic,
    "topodiff": topodiff,
}
