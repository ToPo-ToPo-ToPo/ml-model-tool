"""ml-model-tool: make physics surrogate / generative models (PhysicsNeMo) from JSON specs.

Layers:
    sources/     data spec -> standard dataset (SIMP topology optimization, linear-elastic FEM, TopoDiff import)
    dataset.py   the standard dataset (dataset.npz + dataset.json)
    spec.py      data spec / model spec: defaults and checks
    model.py     model construction (FNO / generative) and the saved bundle (model.mdlus + model.json)
    generative.py generative models (diffusion, flow matching, VAE, GAN): losses and sampling
    nets.py      networks of the VAE / GAN
    training.py  one training loop for every dataset
    evaluation.py metrics, domain checks (FEM compliance, TopoDiff analysis) and figures
The command layer (``ml_model_tool_cli``) turns these into jobs for the CLI, the window and the MCP tool.
"""
__version__ = "0.2.0"
