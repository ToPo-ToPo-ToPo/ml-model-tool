---
name: ml-model
description: How to make machine-learning models of physics problems. Make training data (topology optimization, FEM, TopoDiff import), train a model (an FNO giving one answer per condition, or a generative model making many designs: diffusion, flow matching, VAE, GAN), evaluate it, and compare models. The tool ml-model-tool is used with command lines and does not interpret prose; check the spec keys with spec data / spec model
---

# Making physics ML models (`mmt`)

The tool is `ml-model-tool`. **It does not interpret prose.** What to learn, what to change and how to read the
results is your job; the tool makes data, trains, evaluates and compares exactly as the specs say.

## Rules

1. **Change one thing at a time.** Compare on the same data job and the same test cases (`compare` warns otherwise).
2. **Do not judge by the training loss.** Always read the `evaluate` numbers (test split) and look at `compare.png`.
   Losses are different quantities per model type and cannot be compared across types.
3. **Do not guess keys.** Before writing a spec, run `spec data --source <source>` / `spec model --type <type>`, write
   the returned `example` to a file and change only the keys you need.
4. **On a rejection, fix only the key in `error.where`.** `expected` says what can be written. Do not retype the same line.
5. **Do not wait on long work.** `data` and `train` run as jobs. Check with `job status --job-dir jobs/<…>`; stop with `job stop`.

## The flow

The tool is the single `app("ml-model-tool", line)`. Write the JSON specs into the work folder with `write_file`.
Paths are relative to the work folder; reuse the paths in the reply's `artifacts`. Read `summary.md` first and look
at the figures with `read_file`.

| Step | Line | Returns |
|---|---|---|
| 1. Pick the data | `spec data` → `spec data --source simp-topopt` | the sources → keys, defaults, meaning, example |
| 2. Make the data | `data --spec data.json` | `dataset.npz`, `preview.png` |
| 3. Pick the model | `spec model` → `spec model --type flow` | the types → keys, defaults, meaning, example |
| 4. Train | `train --spec model.json --data jobs/<data>` | `model.mdlus`, `loss.png` |
| 5. Evaluate | `evaluate --model jobs/<train> --data jobs/<data>` | `metrics.json`, `cases.csv`, `compare.png` |
| 6. Compare | `compare --evals jobs/<eval>,jobs/<eval> [--by support]` | table `compare.md`, the same cases side by side in `compare.png` |

Also `predict --model … --data …` (predictions only), `help <command>` (what the arguments mean), `job list`.

## Model types

| Purpose | `model.type` | Notes |
|---|---|---|
| One answer per condition (field prediction, surrogate) | `fno` (default) | Runs on a CPU; fast |
| Many designs per condition (generating shapes) | `flow` (try first), `diffusion`, `vae`, `gan` | Train on a GPU or Apple's GPU (MPS); very slow on a CPU |

- Generative models take `--samples K` (designs per condition) in `evaluate` / `predict`; `flow` / `diffusion` also `--steps S`.
- Their numbers are of sample 1 (comparable with an FNO), plus `diversity` (how different the designs are) and `best_of_k_*`.

## Reading the results

In the `compare` table ↓ means lower is better, ↑ higher is better, and the best of a row is bold.

| Number | Meaning | Rule of thumb |
|---|---|---|
| `iou` ↑ | overlap of the material (density >= 0.5) | matching pixels do not guarantee performance |
| `compliance_ratio` ↓ | FEM compliance of the predicted design / SIMP's | near 1 is good; above 2 is broken |
| `floating_share` ↓ | share of designs with material not joined to a support | 0 is ideal |
| `overhang_share` ↓ | share of designs with material that cannot be printed (no 45° support chain from the base plate; `optopus-overhang`) | 0 is ideal; `overhang_input_share` is the input design's value |
| `rel_l2` ↓ | relative error of a field (`rel_l2/<channel>` per channel) | 1 = as bad as predicting all zeros |
| `peak_stress_error` ↓ | error of the peak von Mises stress | when predicting FEM results |
| `diversity` | how different the generated designs are | near 0 = the same design every time |
| `best_of_k_compliance_ratio` ↓ | best of the K designs, picked by FEM | clearly better than sample 1 = "draw many, pick by FEM" works |

A training `overfit_ratio` (validation / training loss) above 2 means memorizing: add data or revisit inputs / model.

## Common tasks

- **Effect of an input**: train two models on the same data that differ only in `inputs`, evaluate both, `compare` (e.g. `examples/model_ablation_no_sed.json`).
- **FNO vs a generative model**: change only `model.type`; `compare --evals …,… --by support` also shows each support type.
- **Effect of data size**: change only `train.n_train`.
- **Predicting FEM results**: train an FNO on `simp-fea` data; `compare --by load_kind` splits optimized loads from new ones.
- **Reworking an optimized design to be self-supporting**: train a generative model on `optopus-overhang` data (input: the optimization result, target: a design printable without supports, both solved by JuliaOptopus; needs `julia_optopus` or the environment variable `JULIA_OPTOPUS`). Compare `overhang_share` with the input design's value; `compare --by build` splits by build direction.

## What we know (from runs of this tool)

- **Data size matters most.** With 540 samples the model memorized and broke on the test set; 2,340 improved thick members and outlines.
- **Feeding the FEM result of the solid domain (sed) helps a lot** (TopoDiff does the same).
- **Generating topologies** (64x32, 600 samples, 60 test cases, 8 designs each): median compliance ratio FNO 1.05,
  diffusion 1.14, flow 1.02, VAE 1.08, GAN 1.01. Flow picking the best of 8 by FEM: 1.006. GAN gives nearly the same
  design every time (`diversity` 0.003). Designs with floating material: FNO 50%, flow 11%, GAN 9% (FNO shapes break up).
- **Predicting FEM results** (simp-fea, 3,000 designs x 4 loads, FNO): relative L2 about 20% per quantity, peak stress
  error 4%, compliance error 7%. Stress patterns are captured, but large deformations come out too small. Adding the
  equilibrium to the loss (`train.physics`) lowered the residual but made large deformations worse (if you try it, compare 0.1 with 0).
- **Problems where the supports move freely are hard.** fea-elastic (2,000 samples) barely generalized to the test set.
- Small pixel errors can hide a cut load path that makes performance tens of times worse: always check topologies by compliance.

## Examples

Specs in `examples/` (`read_skill("ml-model", "examples/<name>")`): `data_simp_topopt.json`, `data_fea_elastic.json`,
`data_topodiff.json`, `model_default.json`, `model_ablation_no_sed.json`, `model_diffusion.json`, `model_flow.json`,
`model_vae.json`, `model_gan.json`, `data_optopus_overhang.json`, `model_overhang_diffusion.json`. The key lists of `spec data` / `spec model` are authoritative.
