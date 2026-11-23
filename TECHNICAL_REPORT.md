# SiteSight: technical report

Grounded understanding of construction-site drone images: a caption for every region, a box for
every phrase, a structured site summary, a CPU service, and monitoring. Everything here was built on
one public dataset and one laptop plus a free Colab runtime. Numbers come from committed files
(`reports/`, checked by `tests/test_headline_numbers.py`); where something was not measured, it says so.

## 1. Problem and scope
Input: a UAV image of a construction site. Outputs: (a) a templated English caption for a given
region box, (b) boxes for a given phrase, (c) a per-image tally of described resources, with a
confidence and a "needs review" flag. The dataset supplies region captions, not per-object labels,
which shapes every decision below: one box often covers a group ("Ten foundations are installed on soil"),
annotators did not label everything, and counts are partial annotations rather than site totals.
The tally therefore counts described *regions*, never objects.

## 2. Data
Bang & Kim, Mendeley Data v1 (CC BY 4.0): 1,431 JPEGs (all 960 x 540) and 8,428 region annotations
(the page states 8,601; the 173 difference is unexplained). See [DATASHEET.md](DATASHEET.md) for
composition, cleaning and limits.
- **Caption fields.** A deterministic parser turns captions into resource, count, colour and state:
  8,396 of 8,427 non-empty captions parse (99.63%); the 31 failures are listed with reasons.
- **Leakage-aware split.** Two images are linked if their pHash distance is at most 12 or the cosine
  similarity of their DINOv2-small embeddings is at least 0.90; connected components form 428 groups;
  the split is by group (train 1,002 / val 215 / test 214 images). A random split with the same
  sizes would have put a near-duplicate of 76.6% of test images in train (mean of 20 seeds, range
  70.6% to 81.8%); the grouped split has none under the same rule. Under a looser rule
  (pHash <= 16 or cosine >= 0.85) 44.9% of test images still have a train neighbour, so held-out
  scores measure generalisation to *different scenes of similar sites*, not to unseen sites.
- **Label quality is unmeasured.** The 100-image re-annotation (M3) needs the author's own labelling
  and has not been done; the review tool, sample (`reports/m3/audit_sample.json`) and agreement
  analysis (IoU, label kappa) are built and tested.

## 3. Models and protocol
- **Zero-shot baselines (M4)**: OWLv2-base (detector), Florence-2-base and -large (generative; boxes are
  `<loc_N>` tokens on a 1000-bin grid). Two grounding protocols: `detection` (all 31 classes asked
  everywhere) and `given_phrases` (only classes present). They are not comparable; Florence-2 was only
  run with `given_phrases`.
- **Fine-tuning (M5)**: LoRA (r 16, alpha 32) on Florence-2-base's language model, 3.24 M trainable of
  234.7 M parameters; mixed region-captioning and phrase-grounding objectives; vision tower frozen
  (features cached, so no image augmentation); epoch chosen by validation loss (rule fixed in the
  config); test split evaluated once.
- **Metrics (M4)**: AP/mAP with COCO-style matching, recall at IoU 0.5/0.75, ECE (10 bins), BLEU,
  CIDEr-D, METEOR, and strict/loose field accuracy. Each is unit-tested against a hand-computed
  example. Because 735 distinct captions cover the dataset, n-gram metrics mostly reward learning the
  template; field accuracy is the caption metric.

## 4. Results
All grounding numbers are mAP@0.5 in percent under `given_phrases`; one deterministic run each.

| | val (215 img) | test (214 img, run once) |
|---|---|---|
| OWLv2-base zero-shot | 24.1 | 30.2 |
| Florence-2-large zero-shot | 21.5 | |
| Florence-2-base zero-shot | 16.3 | 18.8 |
| Florence-2-base + LoRA | 39.2 | 34.0 |

Region captions on test (1,186 regions), loose field accuracy: resource 71.8%, colour 79.3%, count
77.6% after fine-tuning, against 14.8%, 22.5%, 0.1% zero-shot. Recall@0.5 on test is 46.5% for the
fine-tuned model against 57.5% for OWLv2: the fine-tuned model has the higher mAP and the lower recall. Fine-tuning also made the model's confidence much worse calibrated (ECE 0.459 on
test; OWLv2 0.048). On validation, recall on group boxes is far below single-object boxes for every
model (e.g. OWLv2 36.7% vs 66.4%), consistent with the group-box labelling convention being a major
cause; this was not isolated by a separate experiment. The original paper reports 43.52% mAP for its own
dense-captioning method; that is a different task and protocol and is not a comparison.

Failure analysis on validation (fixed rules in `sitesight/eval/gallery.py`): of 1,102 ground-truth
boxes, 589 were missed at IoU 0.5 (273 no overlap, 156 small, 148 poor localisation, 12 group split);
652 of 1,102 captions differ from the reference (281 resource, 145 count, 113 state, 64 target, 47
colour). The hypotheses attached to these types come from three random examples each and are not
tested causes.

## 5. Beyond accuracy
**Active learning (M6).** Simulated labelling of training images (cost counted in images, boxes
assumed known, fresh LoRA each round, 3 seeds). To reach 90% of the full-data score: random 283.0,
uncertainty 238.2, uncertainty + diversity 238.5 images (about 16% fewer). Diversity did not help over
plain uncertainty; the likely reason (candidate pools of a near-duplicate-heavy dataset are already
varied) was not tested. The per-round rows stayed in Colab and are not committed.

**Serving (M7).** Export to three ONNX graphs (vision, encoder, decoder) with a numpy decoding loop
that reproduces forced BOS, 3-gram blocking and log-softmax confidence; int8 by dynamic quantisation
of MatMul/Gemm weights only. On 300 validation regions on a 2-vCPU Colab CPU, every request cold:
PyTorch p50 85.1 s, ONNX fp32 12.1 s, int8 9.5 s; 1,242 MB vs 574 MB. ONNX fp32 gives the identical
caption for 99.7% of regions; int8 for 76%, with strict field accuracy about 2 regions of 300 lower.
A first export silently stored int8 at fp32 size (weights had been baked in as constants); it was
caught by comparing file sizes on real weights and is now covered by a test. The decoder has no KV
cache, so 9 to 12 s per cold request is still slow. The FastAPI service rejects non-images,
oversized uploads and invalid boxes with a reason, clamps and reports boxes that overhang the
image, and returns `review` when a caption does not parse or confidence is under 0.5 (an untuned
default). The Docker image is built and import-checked in CI, never run with real weights.

**Monitoring (M8).** A request log without images or captions; per-image quality statistics
(brightness, contrast, Laplacian-variance sharpness, clipped shadows/highlights, size); a drift test
combining per-feature Kolmogorov-Smirnov (Bonferroni) and a multivariate MMD permutation test with an
overall budget of alpha = 0.01; an HTML report; exit codes 0/1/2 (ok/alert/too small). Measured on real
images: same-site batches alert 1%/0%/0% (n = 20/40/80); whole-batch blur (radius 2 and 5), darkening,
brightening and rescaling are caught at every size from 20 to 80 (the mildest blur, radius 0.5, needs 40); with 25% of a batch degraded,
n = 80 is needed; mild JPEG and noise are mostly missed; validation images (new sites) alert against
training (16%/50%/95%), a real brightness difference between sites. Output drift is implemented and
tested on synthetic logs only, and no embedding-based drift test was run.

**Geometry (M9, optional).** RGB-D visual odometry (ORB, depth back-projection, PnP + RANSAC) on TUM
fr1/desk: ATE 8.9 cm frame-to-frame and 9.1 cm with keyframes over 573 frames, against 87 cm for a
camera that never moved; `evo` agrees with the project's own ATE to 1e-6. Keyframes did not help ATE
(they lowered one-second drift but were noisier per frame). No map, bundle adjustment or loop closure.

## 6. What I would not claim
- Anything about unseen sites, other sensors or altitudes (one dataset, similar-site leakage possible).
- Label accuracy (unmeasured), or that the 71.8% resource accuracy would hold against re-annotated labels.
- That `confidence` is a probability (ECE 0.459); it ranks and drives a review flag only.
- That int8 is loss-free (24% of captions change; field accuracy differences are within sample noise).
- That the monitor detects mild degradations or partial faults in small batches (it does not).
- Generality of the odometry result (one sequence, one run per mode).

## 7. Engineering practice
- Tests use real files (PNG/JPEG uploads, VIA JSON, TUM-format sequences); only external engines are
  replaced at their own boundary. Hand-derived expectations for the maths; mutation checks per milestone
  (e.g. the Bonferroni split, quaternion conversion, depth scale and pose composition in the odometry).
- CI (`.github/workflows/ci.yml`): ruff, ruff format, mypy, pytest; a GUI job on offscreen Qt; a Docker
  build with an import check of the installed serving package.
- Every limit is reported, not silent: warnings for clamped boxes, `too_small` drift batches, tracking
  failures counted in the odometry, exclusions counted in the evaluation harness.
- Heavy runs are token-free Colab notebooks that embed the code and call the same CLIs.

## 8. Next steps (not done)
1. Run the 100-image audit and re-score against corrected labels.
2. Calibrate confidence (temperature scaling) and re-pick the review threshold on validation.
3. A KV-cached decoder and static int8 for serving; benchmark warm requests.
4. Embedding-based drift and a real prediction log for output drift.
5. A split by site if site metadata can be recovered; otherwise a stricter similarity rule.
