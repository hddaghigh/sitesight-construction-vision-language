# Data licences and required citation

## UAV construction-site captioning dataset

- **Title:** Data for: Context-based information generation from construction site images using unmanned aerial vehicle (UAV)-acquired data and image captioning
- **Authors:** Hyoungkwan Kim, Seongdeok Bang
- **Publisher / version:** Mendeley Data, V1, published 7 February 2020
- **DOI:** [10.17632/4h68fmktwh.1](https://doi.org/10.17632/4h68fmktwh.1)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the dataset page. Attribution is required; changes must be indicated (this repo derives a COCO-style JSON, cleaned/flagged annotations and group splits from the original VIA export).
- **Associated paper:** Automation in Construction, 2020, DOI [10.1016/j.autcon.2020.103116](https://doi.org/10.1016/j.autcon.2020.103116). Complete bibliographic details (title, volume) still to be verified from the publisher page before the README cites it.

Suggested dataset citation (the Mendeley page does not give a ready-made string):

> Kim, H.; Bang, S. (2020). *Data for: Context-based information generation from construction site images using unmanned aerial vehicle (UAV)-acquired data and image captioning*. Mendeley Data, V1. https://doi.org/10.17632/4h68fmktwh.1

The data itself is **not** redistributed in this repository; it lives outside the repo and is fetched by `scripts/download_data.py` (to be written in M1). The file `image_with_captions.zip` (593,575,014 bytes) has SHA-256 `d1860ae77b885ad87dfee0f530606a2b227dfc776a50dc55a20024e9a40f7333`.

## Models (to be completed per milestone)

| Model | Licence (as listed on its Hugging Face API entry, 2026-11-22) |
|---|---|
| microsoft/Florence-2-base, -large | MIT |
| google/owlv2-base-patch16-ensemble | Apache-2.0 |
| IDEA-Research/grounding-dino-tiny | Apache-2.0 |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | Apache-2.0 |
