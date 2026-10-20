"""ml-model-tool: make physics surrogate / generative models (PhysicsNeMo) from JSON specs.

Layers:
    keys.py      every spec key, declared once (default, values, meaning): checks, ``spec`` answers, docs tables
    sources/     data spec -> standard dataset (SIMP topology optimization, FEM, TopoDiff import)
    dataset.py   the standard dataset (dataset.npz + dataset.json)
    spec.py      the model types and their keys; data spec / model spec checks
    model.py     model construction (FNO / generative) and the saved bundle (model.mdlus + model.json)
    generative.py generative models (diffusion, flow matching, VAE, GAN): losses and sampling
    nets.py      networks of the VAE / GAN
    training.py  one training loop for every dataset
    evaluation.py metrics, per-problem checks (``Task``: FEM compliance, FEM fields, TopoDiff analysis), cases.csv
    compare.py   evaluations side by side (table, breakdown by a group of cases, figure)
The command layer (``ml_model_tool_cli``) turns these into jobs for the CLI, the window and the MCP tool.
"""
__version__ = "0.3.0"
