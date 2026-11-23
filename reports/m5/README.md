# M5: LoRA fine-tuning of Florence-2-base

Everything here comes from committed code and `configs/train.toml` (seed 0), run on a Colab T4 via
`scripts/build_colab_notebook.py --kind m5`. Files: `experiment_log.csv`, `train_summary.json`,
`val_compare.{json,md}` (validation: zero-shot vs fine-tuned), `test_final.{json,md}` (test, run once),
`failure_summary.json` (counts of every failure type on validation plus the examples drawn).

## Setup (fixed before training)
- Model `florence-community/Florence-2-base`; LoRA r 16, alpha 32, dropout 0.05 on the language model's
  q/k/v/out/fc1/fc2 (3.24 M trainable of 234.7 M parameters, 1.38%). The vision tower is frozen, so its
  features were cached once: **no image augmentation** was possible.
- Two tasks mixed: region captioning (`<REGION_TO_DESCRIPTION>`) and phrase grounding
  (`<CAPTION_TO_PHRASE_GROUNDING>`). lr 1e-4, cosine, 5% warmup, batch 16, 5 epochs, grad clip 1.0.
- Epoch picked by lowest validation loss (rule fixed in the config): epoch 4 (1.275; epoch 5 was 1.277).
  Training took 1,763 s on a T4.

| epoch | train loss | val loss | val caption loss | val grounding loss |
|---|---|---|---|---|
| 0 (pretrained) | | 5.314 | 6.305 | 3.934 |
| 1 | 1.656 | 1.365 | 0.469 | 2.614 |
| 2 | 1.125 | 1.315 | 0.411 | 2.575 |
| 3 | 1.039 | 1.281 | 0.375 | 2.544 |
| 4 | 0.988 | 1.275 | 0.363 | 2.547 |
| 5 | 0.964 | 1.277 | 0.363 | 2.549 |

## Results
Grounding, protocol `given_phrases` (the model is told which classes are present, so it is **not**
comparable with the `detection` protocol).

| split | model | mAP@0.5 | mAP@0.75 | recall@0.5 | ECE |
|---|---|---|---|---|---|
| val (215 img) | Florence-2-base zero-shot | 16.3 | | 27.5 | 0.101 |
| val | Florence-2-large zero-shot | 21.5 | | 33.8 | |
| val | OWLv2-base zero-shot | 24.1 | | 50.8 | |
| val | **Florence-2-base + LoRA** | **39.2** | 15.9 | 46.6 | 0.432 |
| test (214 img, once) | Florence-2-base zero-shot | 18.8 | 10.6 | 30.9 | 0.131 |
| test | OWLv2-base zero-shot | 30.2 | | 57.5 | |
| test | **Florence-2-base + LoRA** | **34.0** | 19.4 | 46.5 | 0.459 |

Region captioning (1,102 validation regions, 1,186 test regions):

| split | model | resource acc | colour acc | count acc | template parse |
|---|---|---|---|---|---|
| val | zero-shot base | 18.1% | 23.9% | 0.1% | 0% |
| val | LoRA | 74.3% | 80.4% | 74.7% | 99.8% |
| test | zero-shot base | 14.8% | 22.5% | 0.1% | 0% |
| test | LoRA | 71.8% | 79.3% | 77.6% | 99.7% |

BLEU-4 / CIDEr-D / METEOR(exact) on test: 64.9 / 5.55 / 77.7. Those are inflated by the dataset's
rigid caption template (a model that only learns the template scores high); the per-field accuracies
above are the numbers to quote.

## Failure analysis (validation, fixed rules in `sitesight/eval/gallery.py`)
Grounding: 589 of 1,102 ground-truth boxes were missed at IoU 0.5: no_overlap 273, small_missed 156,
poor_localisation 148, group_split 12; plus 406 spurious and 20 duplicate unmatched predictions.
Captioning: 652 of 1,102 captions differ from the reference: resource 281, count 145, state 113,
target 64, colour 47, unparsed 2. Images (about 30 MB) are not committed; regenerate with
`python -m sitesight.eval.gallery` (seeded). Diagnoses below are hypotheses from looking at three random
examples per type, not tested causes.

- **no_overlap**: in the h_beam example the model boxed a real pile of steel beams while the label sits
  on a timber/steel stack elsewhere, so some "misses" are ambiguous or incomplete labels rather than model errors.
- **group_split** (only 12): the model boxed single excavators inside a label that spans two; group-label
  inconsistency, rare in count but it caps recall on group boxes.
- **small_missed**: ground truth under 1% of the image (rebar, workers, containers); likely the 768-px
  Florence input resolution and the 1000-bin location tokens. Untested.
- **colour / state / resource captions**: colour errors are almost all "orange excavator" for black or
  yellow ones and "white" for gray, a prior towards the most common colour. State errors collapse
  "digging" and "located" to the majority class. Resource errors confuse visually similar piles
  (rebar vs mold, fence vs temporary building).

## Caveats
- **Calibration got worse**: ECE rose from 0.10-0.13 (zero-shot) to 0.43-0.46. The token-probability
  confidence is overconfident after fine-tuning; mAP is unaffected, but scores should not be read as
  probabilities until recalibrated (candidate M6/M7 work: temperature scaling on validation).
- OWLv2 still has higher recall@0.5 (50.8 val, 57.5 test) than the fine-tuned model (46.6 / 46.5);
  fine-tuning wins on mAP@0.5 because of its better precision.
- Single run, deterministic, no confidence intervals; differences of a few points are not established.
- Test numbers were produced once, after the epoch rule was fixed; validation was used only for that rule.
- Residual same-site leakage between splits (M2: 297 train-test pairs under looser similarity) means
  test scores may be optimistic; they are not "unseen sites" results.
- Labels are incomplete and use group boxes (see M4 README); the 100-image label audit is still pending.
