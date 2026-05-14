# Stabilized-config training runbook

Hand-off doc for running the first real `configs/train/stabilized.yaml`
experiment on Kaggle (or any free GPU notebook host). Stays out of the
package proper because it's a one-shot operational guide, not project
documentation.

## What you'll produce

By the end of this runbook you will have, **per decode strategy**:

```
results/<run_id>/
    metrics.json        # BLEU-1..4, ROUGE-L, METEOR, CIDEr
    predictions.jsonl   # one row per validation image
    diagnostics.jsonl   # per-sample length / repetition / sentence BLEU
    run_meta.json       # decode flags, timestamp, model id
    report.md           # human-readable summary
```

Download both directories (`results/<greedy_run_id>/` and
`results/<beam_run_id>/`) and the trained `models/v1.0.0/` artefacts as
Kaggle outputs. That's the entire input for the Phase-2-stabilization
comparison.

## Caveats before you start

1. **`requirements.txt` pins `tensorflow-cpu==2.15.0`.** On Kaggle GPU you
   must install `tensorflow==2.15.0` instead — see step 3 below. The
   `Keras 3` warning in the original pin comment still applies: do not
   upgrade to 2.16+, save/load semantics for `TextVectorization` change.
2. **Kaggle session limit is 9 hours per run.** A full 10-epoch pass on
   120k captions at batch 64 fits comfortably on a T4 (well under 9h),
   but plan to checkpoint after every epoch in case the session restarts.
   `default_callbacks(...)` already writes `best.h5` on val_loss
   improvement, so this is handled.
3. **Free-tier Kaggle gives ~30 GB of attached-dataset space.** COCO 2017
   train2017 is ~19 GB. Use the public `awsaf49/coco-2017-dataset`
   mount; do not re-upload it as your own dataset.
4. **The `data.base_path` in `stabilized.yaml` points at `data/coco2017`**
   (the project-relative path). On Kaggle the COCO mount is at
   `/kaggle/input/coco-2017-dataset/coco2017`. You override at runtime
   without editing the YAML using the env-var override pattern (step 4).

## Kaggle notebook cells

Paste each block into a separate cell so you can re-run individual
steps without restarting.

### Cell 1 — Attach the COCO dataset

In the Kaggle notebook UI: **+ Add Data → Search "coco-2017" →**
`awsaf49/coco-2017-dataset`. After attaching, verify:

```python
!ls /kaggle/input/coco-2017-dataset/coco2017
# expect: annotations/  train2017/  val2017/
```

### Cell 2 — Pull the repo

```python
!git clone https://github.com/<your-user>/image-captioning-system.git
%cd image-captioning-system
```

(If your repo isn't public, upload it as a Kaggle dataset and reference
it via `/kaggle/input/<dataset-slug>/`.)

### Cell 3 — Install with the GPU TF wheel

```python
# Replace the CPU TF pin with the GPU-capable wheel. The rest of the
# pinned versions in requirements.txt are GPU/CPU-agnostic.
!pip install -q tensorflow==2.15.0
!pip install -q -r requirements-dev.txt -r requirements-eval.txt
!pip install -q -e .
```

Verify GPU is visible to TF:

```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

Expected output: at least one GPU listed (typically `T4` or `P100`).

### Cell 4 — Train with the stabilized config

The `data.base_path` field in `stabilized.yaml` is overridden via
env-var so the YAML stays unedited:

```python
import os
os.environ["CAPTIONING__DATA__BASE_PATH"] = "/kaggle/input/coco-2017-dataset/coco2017"

!python -m scripts.train \
    --config configs/train/stabilized.yaml \
    --output-dir outputs/runs/stabilized
```

Expected wall-clock on a T4: ~30-50 min per epoch, ~5-8 hours for 10
epochs. EarlyStopping (patience 3) typically fires before epoch 10.

What lands in `outputs/runs/stabilized/`:
- `best.h5`         — best val_loss checkpoint (ModelCheckpoint)
- `model.h5`        — final-epoch weights
- `vocab.pkl/json`  — fitted tokenizer
- `history.json`    — train/val loss per epoch
- `training_log.csv`— CSVLogger output

### Cell 5 — Promote the trained checkpoint

The evaluation and serving paths expect `models/v1.0.0/model.h5` +
tokenizer. Promote the best checkpoint there:

```python
!mkdir -p models/v1.0.0
!cp outputs/runs/stabilized/best.h5    models/v1.0.0/model.h5
!cp outputs/runs/stabilized/vocab.pkl  models/v1.0.0/vocab.pkl
!cp outputs/runs/stabilized/vocab.json models/v1.0.0/vocab.json
```

### Cell 6 — Greedy evaluation

```python
os.environ["CAPTIONING__DATA__BASE_PATH"] = "/kaggle/input/coco-2017-dataset/coco2017"

!python -m scripts.evaluate \
    --config configs/train/stabilized.yaml \
    --weights models/v1.0.0/model.h5 \
    --tokenizer-dir models/v1.0.0 \
    --results-root results \
    --run-id stabilized-greedy \
    --model-id inceptionv3-transformer-stabilized \
    --decode-strategy greedy \
    --max-samples 500
```

If METEOR/Java is unavailable on the Kaggle image (it usually is, but
the wheel can fail at import):

```python
!apt-get install -y openjdk-11-jre-headless  # if METEOR errors out
# or:
!python -m scripts.evaluate ... --skip-meteor
```

### Cell 7 — Beam evaluation

```python
!python -m scripts.evaluate \
    --config configs/train/stabilized.yaml \
    --weights models/v1.0.0/model.h5 \
    --tokenizer-dir models/v1.0.0 \
    --results-root results \
    --run-id stabilized-beam-w4-lp07-nrn3 \
    --model-id inceptionv3-transformer-stabilized \
    --decode-strategy beam \
    --beam-width 4 \
    --length-penalty 0.7 \
    --no-repeat-ngram-size 3 \
    --max-samples 500
```

### Cell 8 — Per-sample inspection (qualitative review)

```python
!python -m scripts.inspect_predictions \
    --config configs/train/stabilized.yaml \
    --weights models/v1.0.0/model.h5 \
    --tokenizer-dir models/v1.0.0 \
    --decode-strategy beam \
    --beam-width 4 \
    --n-samples 30 \
    --output results/stabilized-beam-w4-lp07-nrn3/qualitative.jsonl
```

### Cell 9 — Persist outputs

Kaggle persists anything under `/kaggle/working/` between sessions and
makes it downloadable as a notebook output. Move the results there:

```python
!mkdir -p /kaggle/working/handoff
!cp -r results /kaggle/working/handoff/
!cp -r models /kaggle/working/handoff/
!ls -la /kaggle/working/handoff/results
```

### Cell 10 — Pre-flight summary print

```python
import json
for run in ("stabilized-greedy", "stabilized-beam-w4-lp07-nrn3"):
    m = json.load(open(f"results/{run}/metrics.json"))
    print(run, "->", {k: m[k] for k in ("bleu1", "bleu4", "rouge_l", "meteor", "cider")})
```

This should print two lines that you copy back into the next chat
session — that's all I need to do Steps 4-6 of the original request.

## What to bring back to this conversation

When you return:
1. The two `metrics.json` files (or just the dicts from Cell 10 — same content).
2. The two `report.md` files for the human-readable view.
3. The `diagnostics.jsonl` from at least the beam run, so qualitative
   inspection isn't blind.
4. (Optional but useful) the `history.json` from the training run — lets
   us check whether warmup+cosine actually flattened the validation
   curve as predicted.

With those four artefacts I can do the full quantitative comparison,
qualitative inspection, and bottleneck analysis you originally asked
for, with real numbers and no fabrication.

## If something goes sideways

* **OOM on the T4 during training** — drop `train.batch_size` to 32
  (it's already 64; halving keeps the same effective optimization but
  doubles steps_per_epoch and doubles `warmup_steps` too — the trainer
  auto-derives `cosine_decay_steps`, so just batch_size and warmup_steps
  need adjusting).
* **METEOR fails to import** — pass `--skip-meteor`; the other four
  metrics still write to `metrics.json`.
* **Session times out mid-training** — Kaggle saves the working
  directory; re-attach the notebook, `cp` the partial weights from
  `outputs/runs/stabilized/best.h5` to `models/v1.0.0/`, and continue
  with evaluation only. EarlyStopping likely fired anyway.
* **`scripts/train.py` errors with `FileNotFoundError`** — verify
  `os.environ["CAPTIONING__DATA__BASE_PATH"]` is set in the *same cell*
  as the `!python` call, not a prior cell (Jupyter cell-level env vars
  only propagate to subprocesses started from the same cell on some
  Kaggle images).
