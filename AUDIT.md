# Label-quality audit protocol (milestone M3)

**Goal:** measure how reliable the original labels are by re-annotating a random sample of 100 images yourself, then computing agreement and an error taxonomy. **Status: the tool, sample and analysis code are ready and tested; no audit has been performed yet, so there are no audit results.** The numbers come only from your labels.

## The sample
`reports/m3/audit_sample.json`: 100 images drawn uniformly at random without replacement from all 1,431 (seed 0, `python -m sitesight.data.audit sample`). It holds 67 train / 13 val / 20 test images, 600 original regions (mean 6.0 per image, max 14) and 1 image with no original regions. Sampling from every split estimates the noise of the *whole* dataset.

## Two modes (pick one; you can run `review` first as a warm-up on a copy)
| Mode | What you see | What you do | Time (rough estimate, not measured) | What the numbers mean |
|---|---|---|---|---|
| `blind` (recommended) | The original labels are **hidden** until you finish an image | Draw every box and write every caption from scratch | about 2 to 3 hours (about 600 boxes) | An independent second annotation: genuine inter-annotator agreement |
| `review` | The original labels are shown as the starting point | Fix, delete and add | about 45 to 90 minutes | An **upper bound** (you are anchored on what you see) |

Reports state the mode. Do not mix modes in one audit file.

## Run it
```powershell
.\.venv\Scripts\python.exe -m sitesight.viewer.app `
  --annotations sitesight_data/derived/curated/annotations_curated.json `
  --images sitesight_data/extracted/UAVdata `
  --audit-sample reports/m3/audit_sample.json `
  --audit-file sitesight_data/audit/audit.json --audit-mode blind
```
The audit file is created if missing and **resumed** if it exists (progress shows in the status bar and as ✓ / - / • in the image list). It is saved every time you finish an image and when you close the window.

Install the viewer first if needed: `uv sync --group viewer` (PyQt6, about 90 MB).

## Per image
1. The **edit** panel is on the left. In blind mode the original is hidden, so the right panel (tick *Compare*) is empty until you finish the image.
2. Type or pick the caption in the **Caption** box (it autocompletes from the dataset's existing phrases), then **drag on the image** to draw a box with that caption. Drag inside a box to move it, drag an edge or corner to resize, click to select; **Apply caption** edits the selected box; **Delete** removes it. `Ctrl+Z` / `Ctrl+Y` undo and redo. Wheel zooms, middle-button drag pans, `Ctrl+0` fits.
3. Label what you can see following the dataset's style: one box per object, or one box for a group where the originals do (for example "Fences are installed in a line", "Three concrete pipes are located in a line"). Use the autocomplete to match phrasing, but choose what you actually see, not what you expect the original says.
4. **`Ctrl+Return`** = done and go to the next image; **Skip** if the image is unusable (note why in the notes box). Use the notes field for anything odd (blurry, ambiguous object, etc.).

Do not look at the original labels (other windows, earlier exports) before finishing an image in blind mode.

## Analysis
```powershell
.\.venv\Scripts\python.exe -m sitesight.data.audit report `
  --annotations sitesight_data/derived/curated/annotations_curated.json `
  --audit sitesight_data/audit/audit.json --out reports/m3/audit_report.json
```
writes `audit_report.json` and `audit_report.md`. Optionally export your labels in the SiteSight schema from the viewer (*File > Export audit dataset*).

**Method** (see the docstring in `sitesight/data/audit.py`): regions are matched one-to-one by box IoU at least 0.5 (greedy, highest IoU first). Then:
- *precision* = matched / original regions (how many original labels you confirm); *recall* = matched / your regions (how many of your objects the original has);
- field agreement on matched pairs, using the same caption parser as the dataset: resource, count, colour, state, target; plus exact caption match;
- Cohen's kappa for the resource label (chance-corrected agreement);
- 95% confidence intervals by bootstrapping over images.

## Error taxonomy (automatic, counted per matched pair or unmatched region)
| Code | Meaning |
|---|---|
| `missed_by_original` | You labelled an object the original has no box for (missing label) |
| `not_confirmed_by_audit` | An original box you have no matching box for (spurious, merged or ambiguous label) |
| `box_too_large` / `box_too_small` / `box_shifted` | Matched, but IoU is below 0.75 and the original box is more than 25% larger / smaller, or about the same size but offset |
| `resource_mismatch` | Different object noun (e.g. truck vs dump truck) |
| `count_mismatch` | Different count or count unit (object vs pile) |
| `colour_mismatch`, `state_mismatch`, `target_mismatch` | Colour, verb (located / installed / digging …), or what/where after the verb differs |
| `original_caption_unparsed` / `audit_caption_unparsed` | A caption does not fit the grammar (typo, glued captions, unknown word) |

## Limits to state when quoting results
One auditor (not an independent domain expert); 100 images give wide confidence intervals for rare classes; the same caption parser defines the fields on both sides; matching is greedy; `review` mode is an upper bound; the original annotation guideline is unknown, so a disagreement can be a different (reasonable) convention rather than an error.
