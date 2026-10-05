---
name: ml-model
description: How to make machine-learning models of physics problems (PhysicsNeMo FNO). Make training data (topology optimization, linear-elastic FEM, TopoDiff import), train from a model spec, evaluate against the truth, and improve by changing one condition at a time. The tool ml-model-tool is used with command lines and does not interpret prose
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
4. **Do not wait on long work.** `data` (SIMP takes seconds per sample) and `train` (tens of minutes to hours on a CPU)
   are jobs. If the answer is `state: accepted`, check progress with `job status --job-dir jobs/<…>`; stop with `job stop`.
5. **Fix failures from `error.stage` and the message.** Do not retype the same line. For `spec` errors (unknown key,
   out-of-range value) fix only the JSON.

## Where and how

The tool is the single `app(name, line)`. Write the spec JSON into the working folder with `write_file` and pass the line:

- `app("ml-model-tool", "data --spec data.json")` … make a dataset
- `app("ml-model-tool", "train --spec model.json --data jobs/<data job>")` … train
- `app("ml-model-tool", "evaluate --model jobs/<train job> --data jobs/<data job>")` … evaluate
- `app("ml-model-tool", "predict --model jobs/<train job> --data <dataset>")` … predict
- `app("ml-model-tool", "help train")` … meaning and defaults of the arguments

Paths are relative to the working folder. The answer's `artifacts` lists the outputs; copy those paths into the next line.
Read `summary.md` first and look at the figures (`preview.png` / `loss.png` / `compare.png`) with `read_file`.

## Steps

1. **Make data**: write `data.json`, run `data --spec data.json`. Three sources (all keys in docs/specs.md):
   - `simp-topopt`: topology optimization on a rectangle (64×32 by default); 3 support types, one line load on the boundary,
     a volume fraction range. 6 input channels (volfrac, fix_x, fix_y, load_x, load_y, sed) → density
   - `fea-elastic`: linear elasticity on a full square (64×64 by default); two fixed lines and one line load anywhere.
     5 input channels (material, fix_x, fix_y, load_x, load_y) → displacement ux, uy
   - `topodiff`: import the TopoDiff dataset (`root` = the `dataset_1_diff` folder a person placed). Same 5 inputs as TopoDiff.
     Test `levels` become `level_1.npz` / `level_2.npz`
2. **Train**: write `model.json`, run `train --spec model.json --data jobs/<data>`. `{}` runs with defaults.
   - `inputs`: use only some input channels (e.g. drop sed) — **the right way to test an input**
   - `train.n_train`: limit the training samples (to test data size). `train.epochs`, `lr`, `batch`
   - `loss`: `bce` for density, `mse` for fields (the defaults)
3. **Evaluate**: `evaluate --model jobs/<train> --data jobs/<data>` (default `--split test`). Also evaluating `--split train`
   shows the gap between training data and test (how much was memorized).
   - density: IoU (material overlap at 0.5), MAE, volume error
   - `simp-topopt` also: compliance of the predicted design by FEM / SIMP (close to 1 is good; above 2 means a broken design)
   - fields: relative L2 error (1 is the same as predicting all zeros)
   - TopoDiff tests: `--data jobs/<data>/level_1.npz` gives TopoDiff's metrics (CE / VFE / LD / FM)
4. **Improve**: decide the one thing to change next (see below). Compare against the same data job and test cases, and report
   the numbers from `summary.md` with `compare.png`.

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
`model_ablation_no_sed.json` (for the comparison without sed).
