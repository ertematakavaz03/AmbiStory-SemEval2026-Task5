# Group24 — SemEval-2026 Task 5 · Final Submission

**Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding**

**Members:** Taylan İrak · Ertem Ata Kavaz · Edes Rıza Çakmak · Onur Deniz Öğüncü · Emre Berk Hamarat

## What this is

A working system for **AmbiStory** graded word-sense plausibility rating: a **3-component calibrated ensemble** —

* **A** — zero-shot NLI sense-plausibility scorer (no training)
* **B** — fine-tuned gloss-informed encoder regressor
* **C** — *novel* Likert-distribution head (predicts the 5-vote distribution, reads out a continuous expected rating + a free uncertainty)

blended with weights tuned on a **homonym-disjoint train hold-out**. Open-source HuggingFace models only — **no API keys, fully reproducible**.

## Contents

| Path | What it is |
|------|------------|
| `Group24_final_report (1).pdf` | Final project report |
| `Group24_final_presentation.pptx` | 15-minute presentation slides |
| `final/Group24_final_notebook.ipynb` | Executed notebook with output logs preserved |
| `final/src/*.py` | The system: `data, metrics, nli_scorer, encoder_reg, likert_head, ensemble, run_all` |
| `final/_check_submission.py` | Validates a predictions file with the official `format_check` |
| `final/_make_figures.py` | Generates all figures from `results.json` |
| `final/figures/*.png` | Confusion matrix, per-class F1, macro P/R, PR curves, calibration, training curves, ablations, SOTA comparison, uncertainty |
| `final/predictions/*.jsonl` | Per-system dev predictions + `ensemble_test.jsonl` (CodaBench submission file) |
| `final/results.json` | Single source of truth — every reported number, written by `run_all.py` |
| `final/requirements_final.txt` | Dependencies beyond the repo root `requirements.txt` |

## Reproduce end-to-end

Run from the **repository root**:

```bash
pip install -r requirements.txt
pip install -r final/requirements_final.txt
# GPU machines: install a CUDA torch build first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# Train + evaluate everything → results.json, predictions/, scored by the official scoring.py
python final/src/run_all.py
#   GPU  → DeBERTa-v3-large, 3 seeds, 4 epochs (~2–3.5 h, canonical)
#   CPU  → RoBERTa-base fallback (weaker)
#   FAST=1 → RoBERTa-base, 2 epochs (~45 min)
```

## Results (dev set, official scorer)

| System | accuracy@std | Spearman ρ |
|--------|-------------|------------|
| Milestone best (feature Ridge) | 0.568 | +0.166 |
| A — zero-shot NLI | 0.624 | +0.381 |
| C — Likert-distribution head (novel) | 0.769 | +0.578 |
| B — gloss-informed regressor | 0.767 | +0.610 |
| **Ensemble (continuous, final system)** | **0.733** | **+0.588** |

## CodaBench submission

The competition file is **`final/predictions/ensemble_test.jsonl`** — 930 lines, continuous predictions in [1, 5].
