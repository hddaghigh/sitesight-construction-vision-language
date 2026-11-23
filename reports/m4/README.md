# M4: zero-shot baselines on the validation split

Zero-shot means: no training, prompts fixed in advance (`sitesight/baselines/prompts.py`), scored on
the **validation** split only (215 images, 1,102 evaluated regions). The test split was not touched.
Full tables (size/crowding breakdowns, per-class AP, image-level presence): [`zero_shot_val.md`](zero_shot_val.md);
machine-readable metrics: [`zero_shot_val.json`](zero_shot_val.json).

## Headline numbers

| Model | Protocol | mAP@0.5 | Recall@0.5 | Recall@0.75 | ECE |
|---|---|---|---|---|---|
| OWLv2-base | `detection` (all 31 classes queried everywhere) | 7.9 | 50.8 | 22.3 | 0.047 |
| OWLv2-base | `given_phrases` | 24.1 | 50.8 | 22.3 | 0.049 |
| Florence-2-base | `given_phrases` | 16.3 | 27.5 | 16.6 | 0.101 |
| Florence-2-large | `given_phrases` | 21.5 | 33.8 | 19.0 | 0.088 |

Region captioning, given the ground-truth box (Florence-2, free-text output vs the dataset's templated captions):

| Model | BLEU-4 | CIDEr-D | Resource acc | Colour acc | Count acc | Fits the template |
|---|---|---|---|---|---|---|
| Florence-2-base | 0.0 | 0.16 | 18.1% | 23.9% | 0.1% | 0.0% |
| Florence-2-large | 0.0 | 0.15 | 14.4% | 14.9% | 0.0% | 0.0% |

## How to read these numbers (important)

1. **The two protocols are not comparable.** `detection` asks about every class in every image, so a
   confident "worker" box in an image with no worker is a false positive (mAP 7.9). `given_phrases`
   only scores classes that really are in the image (mAP 24.1). Florence-2 was only ever queried in the
   second way (it was told what to look for), so its numbers are the *easier* setting and must be compared
   with OWLv2's `given_phrases` row, never its `detection` row.
2. **Ground truth is incomplete and uses group boxes.** Annotators did not label every object, and one box
   often covers a group ("Ten foundations are installed on soil"). Both lower AP/recall for any
   per-object detector. In the data: recall@0.5 on single-object boxes is 66.4% (OWLv2), 46.6% / 51.3%
   (Florence base / large) versus 36.7% / 10.2% / 18.0% on group boxes. That gap is consistent with the
   labelling convention being a major cause (a hypothesis; I did not test it separately).
3. **Strict IoU is hard against these labels.** Recall falls from 50.8% at IoU 0.5 to 22.3% at 0.75 for
   OWLv2. A local check on a 2-image sample showed correctly placed OWLv2 boxes typically at IoU about
   0.7 with the annotators' boxes (single objects, median best IoU 0.71), so this is a box-convention
   effect, not a coordinate bug. mAP@0.5 is the meaningful headline.
4. **Zero-shot captions do not use the template at all** (0% parse), so BLEU/CIDEr are near zero and say
   little. The loose field accuracies (what resource/colour/count the text mentions) are the fair
   measure, and they are low: Florence-2 writes things like "orange telescopic boom lift" or
   "construction site", and almost never gives a count. The larger model is not better here (14.4% vs 18.1%
   resource accuracy). Fine-tuning (M5) is what is expected to fix this; that is a prediction, not a result.
5. **Calibration:** OWLv2's scores are well calibrated for this task (ECE 0.047 to 0.049), Florence-2's
   token-probability confidences less so (0.088 to 0.101; ECE with 10 equal-width bins, IoU 0.5).
6. **Image-level presence F1** (which resource types are in the image) was computed for OWLv2 only
   (29.4% at score >= 0.3). It is meaningless for models queried only where a class is present; an
   earlier version of the code printed 99% for Florence-2 and that bug was fixed (regression test added).
7. **Class imbalance matters:** per-class AP ranges from about 50% (excavator, crane truck, structure,
   tarpaulin, truck for OWLv2) to under 5% (worker, rebar, H-beam, mold, wood bar). Small or numerous
   objects are poorly found: workers are tiny (see the size breakdown) and `forklift` (1 region) and
   `concrete_structure` (1 region) have too little support to read anything into their AP.
8. **Florence-2 vs OWLv2:** OWLv2-base beats both Florence-2 models on recall and on mAP@0.5 in the
   `given_phrases` protocol, even though Florence-2 was told which classes to look for. Florence-2-large
   beats Florence-2-base by 5.2 mAP points and 6.3 recall points; mAP@0.75 is about equal (9.6 vs 11.0).

## What was run (provenance)

- Data: validation split of the curated dataset (`reports/m2/splits.json`, seed 0), 215 images; 1,102 of
  the 1,111 validation regions are evaluated (9 excluded: duplicates, empty or unparsed captions).
- Hardware: Google Colab, NVIDIA T4 GPU. Software: torch 2.14.1+cu130, transformers 5.22.1.
- Models: `google/owlv2-base-patch16-ensemble` (Apache-2.0), `florence-community/Florence-2-base` and
  `-large` (MIT), native `transformers` code (no remote code), fp32, greedy decoding for Florence-2.
- Speed on the T4 (this session, includes all queries): OWLv2 1.1 s per image for all 31 classes,
  Florence-2-base 1.1 s, Florence-2-large 1.85 s per image for both its tasks. On the development laptop CPU
  Florence-2-base needed about 140 s per image, which is why the run moved to Colab.
- No randomness is involved (deterministic inference, fixed prompts), so there are no seeds to vary and no
  confidence intervals; with 215 images and rare classes, small differences between models are within
  noise (per-class APs on classes with a handful of regions especially).
- Reproduce: `python scripts/build_colab_notebook.py --out colab.ipynb`, upload it to Colab, Run all
  (no token needed). `python -m sitesight.eval.run --from-metrics ... --out ...` re-renders the tables.
  The raw prediction files stay in Colab/the laptop data folder (OWLv2's are hundreds of MB).
- Code version: the run used commit `0a54636` (downloader and `splits` fixes). The evaluation code was
  changed afterwards only to stop emitting the invalid presence metric for Florence-2; the metrics file
  committed here was passed through that fix (`--from-metrics`), and no other number changed.

## Limits

- One run, validation split, no confidence intervals (see above).
- Prompts are plain class names (and OWLv2's standard "a photo of a ..." template); a different wording
  would change the numbers. I did not tune them, deliberately.
- Florence-2 was given the oracle list of classes present, which flatters it.
- Region captioning used the ground-truth boxes, so it measures description quality only, not localisation.
- Everything here is zero-shot; the point is the baseline fine-tuning (M5) has to beat.
