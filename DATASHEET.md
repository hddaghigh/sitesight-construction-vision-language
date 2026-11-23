# Datasheet: UAV construction-site region captions (as curated by SiteSight)

Format follows "Datasheets for Datasets" (Gebru et al.). Every number here comes from a committed script and config; the machine-readable evidence is in [`reports/m2/`](reports/m2/). Where something is **unknown**, it says so.

## 1. Motivation
- **Original purpose** (from the dataset page and paper): generate text about construction resources (position, status, movement, colour, quantity) from UAV images with image captioning. Source: Bang & Kim, Mendeley Data v1, [10.17632/4h68fmktwh.1](https://doi.org/10.17632/4h68fmktwh.1), CC BY 4.0 (see [DATA_LICENSES.md](DATA_LICENSES.md)).
- **SiteSight's purpose:** grounded region captioning and phrase grounding on construction sites, with honest evaluation. This datasheet documents the *curated derivative*, not the original release.

## 2. Composition
| Item | Value | Source |
|---|---|---|
| Images | 1,431 JPEG, all 960 x 540 | `reports/m2/validation_report.json` |
| Region annotations | **8,428** (the dataset page states 8,601; the 173 difference is unexplained) | same |
| Annotation form | one axis-aligned box + one English phrase per region | VIA 1.x export |
| Regions per image | min 0, median 5, mean 5.89, max 17; 19 images have none | M1 measurement |
| Distinct captions | 795 raw / 746 stripped / 735 whitespace-normalised | M1 measurement |
| Boxes under 1% of image area | 2,103 of 8,428 (25%) | M1 measurement |

Caption grammar: `[count] [pile of] [colour] RESOURCE [is/are] PREDICATE`, e.g. "Two orange excavators are digging soil". The deterministic parser (`sitesight/data/caption_fields.py`) extracts **resource, count, colour, state, preposition, target, destination** per region.

**Parser coverage** (`reports/m2/coverage_report.json`): 8,396 of 8,427 non-empty captions parse (**99.63%**); 31 do not, and one caption is empty. Failure reasons: 15 two captions glued without a space ("…soilA black dust…"), 9 two clauses, 3 no resource (e.g. truncated "An orange exca"), 2 a repeated verb ("located are located"), 2 unknown predicate word. 42 parsed captions needed a recorded typo/truncation repair (e.g. "sturcture", "wo workers"). 7,239 parsed captions carry a count; 1,147 (plurals/mass nouns such as "Fences are installed") have **no** count and are kept as unknown rather than guessed. 1,113 of the counts refer to *piles* of objects, not to objects (`count_unit = "pile"`; a further 10 "Piles of X" captions have no count).

**Classes** (31 resource categories; ids are alphabetical and stable). Largest: excavator 1,558, worker 884, fence 821, dump_truck 703, h_beam 616, tarpaulin 481, rebar 414. Smallest: safety_fence 2, crane 8, forklift 11, concrete_structure 14. "Class" here means the resource noun the annotator wrote; there is no independent class taxonomy in the source.

## 3. Collection process
- **Known:** UAV-acquired construction-site images with human-written region captions on the VIA tool.
- **Unknown (not stated on the dataset page):** number of sites and flights, acquisition dates, altitude / ground sampling distance, camera model, number of annotators, annotation guidelines, quality control, whether images were resized to 960 x 540. A visual check of ~30 sampled image pairs shows both oblique and top-down views over several sites in different seasons; this is an observation, not documentation.

## 4. Preprocessing, cleaning and labelling done by SiteSight
Nothing is silently dropped. Findings are flagged (`flags` on each annotation):
| Finding | Count | Treatment |
|---|---|---|
| Boxes extending past the image (1 to 41 px) | 6 | clipped; original kept in `bbox_raw`; flag `clipped` |
| Exact duplicate (same box and caption) in one image | 12 | kept, flag `duplicate` |
| Empty caption | 1 | kept, flag `empty_caption` |
| Caption not parseable | 31 | kept, flag `unparsed`, no category |
| Trailing / doubled whitespace in captions | 665 regions differ from their normalised form (13 beyond trailing spaces) | `caption` normalised, `caption_raw` preserved |
| Two captions run together in one region | 15 glued with no space ("…soilA black dust…") plus 9 with two clauses (mostly two captions; one is a repeated-verb stutter) | reported as `unparsed`; flagged for the label audit (M3) |

**Near-duplicate grouping and splits** (`configs/data.toml`, seed 0):
- Two images are linked if their pHash distance is at most **12** (of 64 bits) **or** the cosine similarity of their DINOv2-small CLS embeddings (whole 16:9 frame, 224 x 392) is at least **0.90**. A *group* is a connected component (single linkage).
- Result: **428 groups** (230 singletons, largest 74 images = 5.2%, median 1). 5,901 linked pairs.
- Split by group, 70/15/15: **train 1,002 / val 215 / test 214 images** (6,128 / 1,111 / 1,189 regions).
- **Leakage a naive random split would have had** (20 seeds, same rule, same sizes): mean **1,210** near-duplicate train-test pairs (range 1,076 to 1,360) and **76.6%** of test images (70.6% to 81.8%) have a near-duplicate in train. The grouped split has **0** pairs and 0% under the same rule.

How the thresholds were chosen: from the similarity distributions and a visual check of 15 randomly sampled pairs per signal (contact sheets were inspected but are not committed). pHash up to 12 and DINOv2 cosine from 0.95 up looked like the same scene; 0.85 to 0.90 was mixed; 0.80 and below was mostly different sites. At cosine 0.85 single linkage chains into one group holding 64% of all images, so 0.90 is the loosest rule that does not chain. There is **no ground-truth site label**, so this is a judgement, not a measured purity.

## 5. Uses
- **Intended:** research and portfolio evaluation of region captioning, phrase grounding, structured site summaries, active learning and data-quality tooling.
- **Not suitable for:** safety-critical decisions; counting people or machines on a real site (counts are partial annotations of regions, not site totals); claims about generalisation to *new sites* (see limitations).

## 6. Limitations and known risks (please read)
1. **Residual same-site leakage.** The grouped split removes near-duplicates under the chosen rule only. If "similar" is defined more loosely (pHash at most 16 **or** cosine at least 0.85) the grouped split still has **297** train-test pairs and **44.9%** of test images have such a neighbour in train; at (20, 0.80) it is 98.1%. Held-out scores are therefore a measure of generalisation to *unseen frames/scenes*, **not necessarily unseen sites**. Fixing this properly needs site labels or a clustering that avoids chaining (planned follow-up).
2. **Templated captions.** About 735 distinct strings cover 8,428 regions; n-gram metrics (CIDEr/BLEU) reward template learning. Field-level accuracy is the primary caption metric.
3. **Class imbalance and missing classes in test.** `safety_fence` (2 regions, all in val) and `water_tank` (29, all in train) do not appear in test; `crane`, `forklift`, `tower_crane`, `concrete_structure`, `dust_protection`, `structure` have under 25 train regions. Per-class metrics must be reported with their support, and absent classes excluded explicitly.
4. **Label noise.** Typos, grammar slips, glued captions, ambiguous nouns ("drilling" for a drilling rig, "crane" vs "crane truck", "structure"). The planned 100-image re-annotation (M3) will measure agreement and produce an error taxonomy; **until then label accuracy is unmeasured.**
5. **Unexplained count gap** (8,428 vs 8,601) and unknown acquisition details (section 3).
6. **Small objects:** 25% of boxes are under 1% of the image area; some may be unresolvable at 960 x 540.

## 7. Distribution and licence
Original data: CC BY 4.0, cite as in [DATA_LICENSES.md](DATA_LICENSES.md). Changes made: format conversion to COCO-style JSON, box clipping, caption normalisation, parsed fields, groups and splits. The images and original annotations are **not** redistributed in this repository; derived reports and `reports/m2/splits.json` (file name to split/group) are.

## 8. Maintenance
Reproduce with `make data`, `make fields`, `make embed`, `make curate` (see the Makefile). Parser, grouping and split code are unit-tested; thresholds and seeds live in `configs/data.toml`. Changing a threshold changes every number in sections 4 and 6; regenerate `reports/m2/` in the same commit.
