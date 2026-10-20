---
name: ml-model
description: How to make machine-learning models of physics problems (PhysicsNeMo FNO, and generative models: diffusion, flow matching, VAE, GAN). Make training data (topology optimization, linear-elastic FEM, TopoDiff import), train from a model spec (an FNO giving one answer per condition, or a generative model making many designs), evaluate against the truth, and improve by changing one condition at a time. The tool ml-model-tool is used with command lines and does not interpret prose
---

# Making physics ML models (`mmt`)

The tool is `ml-model-tool`. **It does not interpret prose.** What to learn, which condition to change and how to read
the results is your job; the tool's job is to make data, train and evaluate exactly as the specs say, with figures and numbers.

## Rules

1. **Change one thing at a time.** Changing inputs, data size, epochs, loss and model size together makes it impossible
   to tell what worked. Compare on the same data (the same data job) and the same test cases.
2. **Do not judge by the training loss.** Always `evaluate` the **test** split and look at `compare.png`. If the training
   loss falls but the validation loss does not, the model memorized (overfitting). When `overfit_ratio`
   (validation / training loss) is above 2, add data or revisit the inputs / model.
3. **No point loads or point supports** (singular in 2D elasticity). The sources already use lines of 2-3 nodes.
4. **Do not wait on long work.** `data` (SIMP takes seconds per sample) and `train` (minutes to tens of minutes on a GPU / MPS, tens of times longer on a CPU)
   are jobs. If the answer is `state: accepted`, check progress with `job status --job-dir jobs/<…>`; stop with `job stop`.
5. **Fix failures from `error.stage` and the message.** Do not retype the same line. For `spec` errors (unknown key,
   out-of-range value) fix only the JSON.

## Where and how

The tool is the single `app(name, line)`. Write the spec JSON into the working folder with `write_file` and pass the line:

- `app("ml-model-tool", "data --spec data.json")` … make a dataset
- `app("ml-model-tool", "train --spec model.json --data jobs/<data job>")` … train
- `app("ml-model-tool", "evaluate --model jobs/<train job> --data jobs/<data job>")` … evaluate
- `app("ml-model-tool", "predict --model jobs/<train job> --data <dataset>")` … predict (a generative model makes designs)
- generative models take `--samples K` (designs per condition) on `evaluate` / `predict`; diffusion / flow also `--steps S` (sampling steps)
- `app("ml-model-tool", "help train")` … meaning and defaults of the arguments

Paths are relative to the working folder. The answer's `artifacts` lists the outputs; copy those paths into the next line.
Read `summary.md` first and look at the figures (`preview.png` / `loss.png` / `compare.png`) with `read_file`.

## Steps

1. **Make data**: write `data.json`, run `data --spec data.json`. Five sources (all keys in docs/specs.md):
   - `simp-topopt`: topology optimization on a rectangle (64×32 by default); 3 support types, one line load on the boundary,
     a volume fraction range. 6 input channels (volfrac, fix_x, fix_y, load_x, load_y, sed) → density
   - `optopus-overhang`: a topology optimization result → a design printable without supports (self-supporting,
     overhangs up to 45°), both solved by JuliaOptopus (free and self). 7 inputs (design, volfrac, fix_x, fix_y, load_x,
     load_y, base) → density. `build_directions` sets the build direction; `julia_optopus` (or the environment variable
     `JULIA_OPTOPUS`) is the JuliaOptopus folder, `workers` the parallel processes. Data for generating several
     printable designs with a generative model
   - `fea-elastic`: linear elasticity on a full square (64×64 by default); two fixed lines and one line load anywhere.
     5 input channels (material, fix_x, fix_y, load_x, load_y) → displacement ux, uy
   - `simp-fea`: SIMP-optimized designs (64×32 by default) analysed by FEM under `load_cases` loads each.
     5 inputs (density, fix_x, fix_y, load_x, load_y) → ux, uy, von_mises, energy (an FEM surrogate).
     `target: "displacement"` learns the nodal displacements ux, uy (`train.physics` adds the equilibrium to the loss)
   - `topodiff`: import the TopoDiff dataset (`root` = the `dataset_1_diff` folder a person placed). Same 5 inputs as TopoDiff.
     Test `levels` become `level_1.npz` / `level_2.npz`
2. **Train**: write `model.json`, run `train --spec model.json --data jobs/<data>`. `{}` runs with defaults.
   - `inputs`: use only some input channels (e.g. drop sed) — **the right way to test an input**
   - `train.n_train`: limit the training samples (to test data size). `train.epochs`, `lr`, `batch`
   - `loss`: `bce` for density, `rel_l2` for fields (the defaults)
   - `model.type`: `fno` (default; one answer per condition) or a generative `diffusion` / `flow` / `vae` / `gan` (see "Generative models")
3. **Evaluate**: `evaluate --model jobs/<train> --data jobs/<data>` (default `--split test`). Also evaluating `--split train`
   shows the gap between training data and test (how much was memorized).
   - density: IoU (material overlap at 0.5), MAE, volume error
   - `simp-topopt` also: compliance of the predicted design by FEM / SIMP (close to 1 is good; above 2 means a broken design)
     and floating material (material not joined to a support; `floating.share` = share of designs that have some)
   - `optopus-overhang` also: `overhang` (material that cannot be printed from the base plate; `share` = share of designs
     with some) and, as the baseline, the same for the input design (`overhang_input`). The compliance ratio is against
     the self design
   - fields: relative L2 error (1 is the same as predicting all zeros), also per quantity when there are several
   - `simp-fea` also: the peak von Mises stress error and the compliance error from the sum of the predicted energy
   - TopoDiff tests: `--data jobs/<data>/level_1.npz` gives TopoDiff's metrics (CE / VFE / LD / FM)
   - generative models: the numbers above are of design 1, plus `diversity` (how different the designs are) and
     `best_of_k` (the best of the K designs)
4. **Improve**: decide the one thing to change next (see below). Compare against the same data job and test cases, and report
   the numbers from `summary.md` with `compare.png`.

## Generative models (`model.type: diffusion` / `flow` / `vae` / `gan`)

They make a density (or field) from random numbers, with the input channels as the condition. **One condition gives many
designs** — the difference from an FNO.

| type | method | sampling |
|---|---|---|
| `diffusion` | conditional diffusion (the idea of TopoDiff, PhysicsNeMo's TopoDiff U-Net) | `steps` denoising steps (default 50) |
| `flow` | flow matching (same U-Net) | `steps` Euler steps from noise (default 20) |
| `vae` | conditional VAE | one pass |
| `gan` | conditional GAN (PatchGAN discriminator) | one pass |

Examples: `examples/model_diffusion.json`, `model_flow.json`, `model_vae.json`, `model_gan.json`.

1. **Compare with an FNO on the same data and test cases.** Train and evaluate an FNO first, then change only
   `model.type` (same input channels). Compare design 1's IoU and compliance ratio (one draw each).
2. **Train on a GPU or Apple's GPU (MPS).** `device: auto` picks CUDA → MPS → CPU. On a CPU it is very slow (about 100
   times the MPS time for the same U-Net). 200 epochs by default. Each kind has its own loss; losses cannot be compared
   across kinds or with an FNO (compare the evaluation numbers). A GAN saves its last epoch.
3. **What to read in the evaluation**:
   - `generative.diversity` (share of cells where two designs disagree on material) near 0 means one shape per condition
   - if `best_of_k.compliance_ratio` (the best of K designs by FEM) is clearly better than design 1, drawing more and
     picking by FEM pays off. A large `compliance_ratio_all.share_gt_2` means many broken designs
   - `compare.png` shows the truth and 4 designs: check that loads connect to supports and that nothing floats
4. **Designs and steps**: `--samples 8` draws more (time grows in proportion). Fewer `--steps` (diffusion / flow) is
   faster but coarser. Compare at the same `--steps`.
5. **Only similar shapes** (`diversity` near 0): raise `kl_weight` (VAE), or raise `diversity_weight` / lower `l1_weight` (GAN).

## What is known (from studies with this tool)

- **Data size matters most.** With 540 samples the model memorized the training data and broke on the test set; 2,340 samples
  improved thick members and outlines.
- **The FEM result of the fully solid domain (sed) as an input helps a lot** (no new information, but it hands over the solved
  force flow). TopoDiff does the same with strain energy density and von Mises stress.
- **Freely placed supports are hard.** fea-elastic (two supports and a load anywhere, 2,000 samples) barely generalized. Check an
  easier problem with fixed supports first, then widen.
- A small pixel error can still hide a broken load path (tens of times worse performance). Always check topology results by
  compliance (FEM).

## Examples

`examples/` has spec examples (read with `read_skill("ml-model", "examples/data_simp_topopt.json")`):
`data_simp_topopt.json`, `data_fea_elastic.json`, `data_topodiff.json`, `model_default.json`,
`model_ablation_no_sed.json` (for the comparison without sed), `model_diffusion.json`, `model_flow.json`, `model_vae.json`, `model_gan.json` (generative).
