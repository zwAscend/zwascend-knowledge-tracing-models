# zwascend-knowledge-tracing-models

# DKT-GRU (MindSpore) — End-to-End Pipeline README

This project implements a **Deep Knowledge Tracing (DKT)** model using a **GRU** backbone in **MindSpore**. It covers the full workflow:

1. **Data preprocessing** (raw interaction logs → model-ready sequences)
2. **Model training** (DKT-GRU)
3. **Validation + checkpointing** (best model by validation AUC)
4. **Test evaluation** (AUC on held-out students)
5. **Inference** (produce a per-skill mastery probability vector)

---

## 1. Prerequisites

### System

* Ubuntu 22.04 (or similar)
* Python 3.9–3.11

### Python environment

You should already have your MindSpore environment set up (`ms-env`).

Activate:

```bash
source ~/ms-env/bin/activate
```

Recommended packages:

```bash
pip install -U fastapi uvicorn pydantic
pip install -U numpy pandas
```

Verify MindSpore:

```bash
python -c "import mindspore as ms; print('MindSpore:', ms.__version__)"
```

---

## 2. Project Structure

Expected structure:

```
dkt_ms/
  data/
    raw/
      interactions.csv
    processed/
      train.npz
      val.npz
      test.npz
      meta.json
  outputs/
    checkpoints/
      best.ckpt
    train_summary.json
  src/
    preprocess.py
    model.py
    train.py
    eval.py
    infer.py
```

---

## 3. Data Input Format (What the model expects)

The **raw input** is a CSV of learner interaction logs.

### Required columns

* `student_id` (string)
* `timestamp` (ISO 8601 recommended)
* `skill_id` (string concept tag)
* `correct` (0 or 1)

### Example row

```csv
S001,2025-12-01T08:10:00Z,SKILL_LINEAR_EQ_SOLVE,1
```

### Important concept note

The model does **not** ingest “math text” (e.g., the full question statement).
It ingests **concept tags** (`skill_id`) and correctness. Your “actual maths concepts” must be represented as a stable **skill taxonomy** (e.g., `SKILL_FRACTIONS_ADD`, `SKILL_PERCENTAGES_BASIC`, etc.).

---

## 4. Step-by-Step: Run the Pipeline

### Step 1 — Put data in place

Create (or copy) your CSV to:

```bash
mkdir -p data/raw
# Place interactions.csv here:
data/raw/interactions.csv
```

### Step 2 — Preprocess

This converts interaction logs into fixed-length sequence windows and creates train/val/test splits **by student**.

Run:

```bash
python src/preprocess.py
```

#### Expected output (example)

```text
Done.
num_skills: 13
train samples: 2 val: 1 test: 1
```

#### What this means

* `num_skills: 13` → you have 13 unique skills in your CSV (K=13).
* `train/val/test samples` → number of **sequence windows** produced for each split (not raw rows).

#### Output artifacts

`data/processed/` will contain:

* `train.npz`, `val.npz`, `test.npz`: arrays used for training/evaluation
* `meta.json`: skill mappings and preprocessing parameters

---

### Step 3 — Train the DKT-GRU model

Run:

```bash
python src/train.py
```

#### Expected output (example)

```text
Epoch 01 | train_loss=0.7032 | val_loss=0.6936 | val_auc=0.5102
  saved best checkpoint: outputs/checkpoints/best.ckpt (AUC=0.5102)
...
Epoch 10 | train_loss=0.6670 | val_loss=0.6859 | val_auc=0.5714
```

#### What this means

* `train_loss` should generally decrease over epochs.
* `val_auc` is the ROC-AUC computed from masked valid timesteps.
* The best model is saved automatically to:

  * `outputs/checkpoints/best.ckpt`

#### Output artifacts

* `outputs/checkpoints/best.ckpt` (best model)
* `outputs/train_summary.json` (summary info)

---

### Step 4 — Evaluate on test set

Run:

```bash
python src/eval.py
```

#### Expected output (example)

```text
TEST | loss=0.6955 | AUC=0.5208 | points=14
```

#### What this means

* `points` = number of valid test targets (after masking padding).
* If `points` is small (e.g., < 1,000), the AUC will be unstable and not meaningful.

---

### Step 5 — Inference (mastery update demo)

Run:

```bash
python src/infer.py
```

#### Expected output (example)

```text
Weakest skills: {'SKILL_GRAPH_GRADIENT': 0.31, 'SKILL_SIMULTANEOUS_EQ': 0.34, ...}
```

#### What inference returns

* A per-skill mastery probability vector (probability learner answers correctly if they attempt that skill next).
* The sample script prints the weakest skills for demonstration.

---

## 5. Expected Outcomes and What “Good” Looks Like

### With very small sample data

If you are using only a few students and few interactions:

* `train samples` / `val samples` / `test samples` will be tiny.
* `points` may be tens/hundreds.
* AUC may hover around 0.5 and fluctuate heavily.

This is normal and still confirms the pipeline works end-to-end.

### With real platform logs (recommended)

To get meaningful metrics:

* Use **thousands+ learners** and/or **tens of thousands+ interactions**
* Aim for `points` in the **thousands** at minimum for stable AUC

Typical KT reporting includes:

* ROC-AUC (primary)
* LogLoss (secondary)
* Inference latency per update request
* Throughput (updates/sec) on CPU

---

## 6. Common Issues and Fixes

### Issue A: “No training samples produced”

This commonly happens when:

* Your validation/test split has **0 students** (small dataset), or
* Students have fewer than 2 interactions.

Fix:

* Add more students/interactions, or
* Adjust split logic to enforce non-empty splits (recommended).

### Issue B: AUC is near 0.5

Most common cause is insufficient data (`points` too small).
Solution: export more logs or increase window generation for debugging.

### Issue C: MindSpore warning about `device_target` deprecation

Safe to ignore. It does not affect correctness.

---

## 7. Recommended Next Steps for Production Integration

1. Export real logs from your platform in the required format.
2. Train DKT-GRU on your data and report stable metrics.
3. Wrap `infer.py` logic in your backend service:

   * fetch last N events for a student
   * append the new event
   * run model forward
   * return/store mastery snapshot
4. Use mastery snapshot as input into:

   * Risk scoring module (calibrated logistic regression)
   * Recommendation ranker (knowledge-gap × exam weight × urgency)

---

## 8. Commands Summary

```bash
# Activate environment
source ~/ms-env/bin/activate

# Preprocess raw CSV → processed npz files
python src/preprocess.py

# Train DKT-GRU model
python src/train.py

# Evaluate best checkpoint on test data
python src/eval.py

# Run a simple inference demo
python src/infer.py

# Run API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Healthy check
curl -s http://localhost:8000/health
```
