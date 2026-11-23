## Grounding on `val` (percent; IoU 0.5 unless stated)

| Model | Protocol | mAP@0.5 | mAP@0.75 | Recall@0.5 | Recall@0.75 | ECE | Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| owlv2-base-patch16-ensemble | detection | 7.9 | 2.7 | 50.8 | 22.3 | 0.047 | 215 |
| owlv2-base-patch16-ensemble | given_phrases | 24.1 | 7.1 | 50.8 | 22.3 | 0.049 | 215 |
| florence-2-base-grounding | given_phrases | 16.3 | 11.0 | 27.5 | 16.6 | 0.101 | 215 |
| florence-2-large-grounding | given_phrases | 21.5 | 9.6 | 33.8 | 19.0 | 0.088 | 215 |
| florence-2-base-lora-epoch_4-grounding | given_phrases | 39.2 | 15.9 | 46.6 | 28.6 | 0.432 | 215 |

### Recall@0.5 by object size, crowding and region kind

| Model | small | medium | large | sparse | mid-crowd | crowded | single | group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| owlv2-base-patch16-ensemble | 45.2 (n=241) | 49.4 (n=437) | 55.4 (n=424) | 53.5 (n=258) | 52.1 (n=641) | 43.3 (n=203) | 66.4 (n=524) | 36.7 (n=578) |
| florence-2-base-grounding | 12.9 (n=241) | 30.9 (n=437) | 32.3 (n=424) | 38.8 (n=258) | 26.8 (n=641) | 15.3 (n=203) | 46.6 (n=524) | 10.2 (n=578) |
| florence-2-large-grounding | 17.8 (n=241) | 34.8 (n=437) | 42.0 (n=424) | 46.5 (n=258) | 32.6 (n=641) | 21.7 (n=203) | 51.3 (n=524) | 18.0 (n=578) |
| florence-2-base-lora-epoch_4-grounding | 35.3 (n=241) | 51.3 (n=437) | 48.1 (n=424) | 54.3 (n=258) | 47.4 (n=641) | 34.0 (n=203) | 59.9 (n=524) | 34.4 (n=578) |

### Image-level resource presence (F1 at fixed score thresholds)

| Model | F1 @ score>=0.1 | F1 @ score>=0.3 | F1 @ score>=0.5 |
| --- | --- | --- | --- |
| owlv2-base-patch16-ensemble | 23.9 | 29.4 | 25.0 |

Not computed for florence-2-base-grounding, florence-2-large-grounding, florence-2-base-lora-epoch_4-grounding: those models were only queried for classes already present, so their predictions cannot be false alarms.

### Per-class AP@0.5 (given_phrases protocol)

| Class | n GT | owlv2-base-patch16-ensemble | florence-2-base-grounding | florence-2-large-grounding | florence-2-base-lora-epoch_4-grounding |
| --- | --- | --- | --- | --- | --- |
| concrete | 30 | 36.1 | 3.9 | 32.6 | 55.0 |
| concrete_pillar | 20 | 10.5 | 3.3 | 10.0 | 11.6 |
| concrete_pipe | 14 | 16.6 | 9.8 | 12.0 | 30.8 |
| concrete_structure | 1 | 16.7 | 0.0 | 0.0 | 100.0 |
| concrete_truck | 9 | 45.7 | 35.6 | 40.0 | 36.6 |
| container | 23 | 1.4 | 0.9 | 3.0 | 10.5 |
| crane_truck | 26 | 56.2 | 55.9 | 58.6 | 60.3 |
| drilling_rig | 15 | 49.8 | 36.2 | 21.7 | 13.2 |
| dump_truck | 96 | 23.1 | 28.5 | 29.1 | 61.9 |
| dust_protection | 12 | 5.2 | 0.5 | 2.1 | 12.5 |
| excavator | 167 | 51.7 | 34.6 | 36.9 | 41.1 |
| excavator_bucket | 28 | 8.7 | 0.0 | 0.0 | 49.3 |
| fence | 67 | 27.8 | 8.5 | 28.3 | 36.7 |
| forklift | 1 | 100.0 | 100.0 | 100.0 | 100.0 |
| foundation | 18 | 0.5 | 1.4 | 1.9 | 16.5 |
| h_beam | 76 | 3.3 | 0.1 | 0.8 | 26.7 |
| material | 9 | 3.0 | 0.0 | 2.8 | 65.1 |
| mold | 60 | 1.5 | 0.2 | 0.3 | 2.6 |
| rebar | 98 | 4.4 | 2.6 | 0.8 | 11.3 |
| safety_cone | 7 | 0.4 | 0.0 | 0.0 | 7.1 |
| safety_fence | 2 | 4.7 | 0.0 | 50.0 | 50.0 |
| structure | 18 | 55.3 | 38.6 | 42.6 | 32.1 |
| tarpaulin | 67 | 51.4 | 19.8 | 35.7 | 52.0 |
| temporary_building | 17 | 26.7 | 21.0 | 24.7 | 70.6 |
| truck | 33 | 54.2 | 30.9 | 29.2 | 62.1 |
| wood_bar | 42 | 3.4 | 0.1 | 1.2 | 28.2 |
| wood_fence | 33 | 11.7 | 21.2 | 36.9 | 42.6 |
| worker | 113 | 4.1 | 1.8 | 1.8 | 11.1 |

## Region captioning (given the ground-truth box)

| Model | Regions | BLEU-4 | CIDEr-D | METEOR (exact) | Resource acc | Colour acc | Count acc | Template parse rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| florence-2-base-regions | 1102 | 0.0 | 0.16 | 3.1 | 18.1 | 23.9 | 0.1 | 0.0 |
| florence-2-large-regions | 1102 | 0.0 | 0.15 | 2.7 | 14.4 | 14.9 | 0.0 | 0.0 |
| florence-2-base-lora-epoch_4-regions | 1102 | 64.4 | 5.53 | 76.6 | 74.3 | 80.4 | 74.7 | 99.8 |

