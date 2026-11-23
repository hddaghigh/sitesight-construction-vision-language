# M6: active-learning simulation

Question: if only a fraction of the training images can be captioned by a person, does choosing
them by model uncertainty (with or without diversity) beat choosing at random?

## Setup (fixed in `configs/active.toml` before the run)
- Pool: 990 training images (6,096 regions). Labelling cost is counted in **images**; all regions of
  an image are labelled together. Region boxes are assumed known; only captions are "bought".
- Budgets: 50, 99, 198, 396 images (5/10/20/40% of the pool). The first 50 are a random set shared
  by all strategies within a seed; strategies diverge from round 1.
- Every round trains a fresh LoRA (r 16, alpha 32) on Florence-2-base for 3 epochs, scored on the
  validation split (1,102 regions). The test split is never used.
- Score = mean of strict resource, colour and count accuracy. Seeds 0, 1, 2. Tesla T4.
- Strategies: random; uncertainty (1 minus mean geometric-mean token probability of generated
  captions); uncertainty_diversity (top 3k uncertain, then greedy k-center on cached vision features).
- Target: labels needed to reach 90% of the same seed's full-data score (linear interpolation, no
  extrapolation). Full-data scores: 0.7432, 0.7424, 0.7409 (target about 0.668).

## Results (mean ± sample std over 3 seeds)
| Labelled images | random | uncertainty | uncertainty + diversity |
|---|---|---|---|
| 50 | 0.476 ± 0.006 | 0.476 (shared) | 0.476 (shared) |
| 99 | 0.591 ± 0.008 | 0.602 ± 0.012 | 0.596 ± 0.002 |
| 198 | 0.648 ± 0.005 | 0.658 ± 0.004 | 0.659 ± 0.001 |
| 396 | 0.694 ± 0.002 | 0.707 ± 0.004 | 0.705 ± 0.007 |

Labelled images to reach 90% of full-data score: random 283.0 ± 9.8; uncertainty 238.2 ± 10.4;
uncertainty + diversity 238.5 ± 3.7 (per-seed values: random 293.6, 274.2, 281.2; uncertainty
226.5, 241.8, 246.4; diversity 239.5, 241.5, 234.4).

## Reading
- Both active strategies beat random at every budget, by about 1 to 1.3 points, and need about
  45 fewer images (16%) to reach the target. The gap is several seed standard deviations.
- Diversity did **not** help over plain uncertainty here (negative result); it only reduced
  variance. We did not test why. A possible reason is that 3x-candidate pools of a near-duplicate-heavy
  dataset are already varied enough, but this is a hypothesis.

## Caveats
Boxes assumed known; cost in images not regions; captioning only (no grounding); 3 epochs per
round; one dataset with possible residual same-site leakage; 3 seeds only; validation split used
for both scoring and reporting (no test claim). Per-round rows (`results.jsonl`) stayed in Colab and
are not committed; the numbers above come from its `summary.json` printout.
