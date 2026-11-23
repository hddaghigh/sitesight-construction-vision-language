## Grounding on `test` (percent; IoU 0.5 unless stated)

| Model | Protocol | mAP@0.5 | mAP@0.75 | Recall@0.5 | Recall@0.75 | ECE | Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| florence-2-base-lora-epoch_4-grounding | given_phrases | 34.0 | 19.4 | 46.5 | 28.2 | 0.459 | 214 |
| florence-2-base-grounding | given_phrases | 18.8 | 10.6 | 30.9 | 18.3 | 0.131 | 214 |
| owlv2-base-patch16-ensemble | detection | 9.5 | 4.1 | 57.5 | 26.8 | 0.048 | 214 |
| owlv2-base-patch16-ensemble | given_phrases | 30.2 | 11.3 | 57.5 | 26.8 | 0.048 | 214 |

### Recall@0.5 by object size, crowding and region kind

| Model | small | medium | large | sparse | mid-crowd | crowded | single | group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| florence-2-base-lora-epoch_4-grounding | 36.0 (n=253) | 49.2 (n=496) | 49.7 (n=437) | 58.7 (n=259) | 50.8 (n=530) | 33.0 (n=397) | 57.9 (n=589) | 35.3 (n=597) |
| florence-2-base-grounding | 20.6 (n=253) | 30.0 (n=496) | 37.8 (n=437) | 44.4 (n=259) | 32.8 (n=530) | 19.4 (n=397) | 50.6 (n=589) | 11.4 (n=597) |
| owlv2-base-patch16-ensemble | 49.4 (n=253) | 56.5 (n=496) | 63.4 (n=437) | 67.6 (n=259) | 59.8 (n=530) | 47.9 (n=397) | 75.2 (n=589) | 40.0 (n=597) |

### Image-level resource presence (F1 at fixed score thresholds)

| Model | F1 @ score>=0.1 | F1 @ score>=0.3 | F1 @ score>=0.5 |
| --- | --- | --- | --- |
| owlv2-base-patch16-ensemble | 25.6 | 31.0 | 23.3 |

Not computed for florence-2-base-lora-epoch_4-grounding, florence-2-base-grounding: those models were only queried for classes already present, so their predictions cannot be false alarms.

### Per-class AP@0.5 (given_phrases protocol)

| Class | n GT | florence-2-base-lora-epoch_4-grounding | florence-2-base-grounding | owlv2-base-patch16-ensemble |
| --- | --- | --- | --- | --- |
| concrete | 37 | 54.5 | 3.1 | 32.1 |
| concrete_pillar | 18 | 23.9 | 15.4 | 25.2 |
| concrete_pipe | 17 | 31.2 | 3.5 | 18.9 |
| concrete_structure | 2 | 0.0 | 50.0 | 4.4 |
| concrete_truck | 14 | 89.2 | 35.2 | 68.4 |
| container | 22 | 17.9 | 0.9 | 3.7 |
| crane | 2 | 50.0 | 50.0 | 83.3 |
| crane_truck | 51 | 37.0 | 33.8 | 37.8 |
| drilling_rig | 31 | 26.3 | 47.2 | 56.6 |
| dump_truck | 103 | 47.3 | 24.6 | 27.9 |
| dust_protection | 13 | 14.8 | 5.6 | 54.8 |
| excavator | 150 | 48.6 | 31.5 | 58.8 |
| excavator_bucket | 14 | 13.1 | 0.0 | 6.6 |
| fence | 88 | 48.0 | 8.7 | 24.2 |
| forklift | 4 | 100.0 | 50.0 | 68.8 |
| foundation | 23 | 10.8 | 1.3 | 12.1 |
| h_beam | 117 | 17.7 | 0.2 | 5.1 |
| material | 14 | 26.4 | 0.0 | 0.3 |
| mold | 66 | 13.0 | 0.3 | 1.0 |
| rebar | 64 | 14.5 | 1.0 | 8.7 |
| safety_cone | 10 | 20.0 | 0.0 | 0.4 |
| structure | 14 | 12.6 | 2.2 | 14.2 |
| tarpaulin | 82 | 59.6 | 16.9 | 47.0 |
| temporary_building | 24 | 21.4 | 10.0 | 26.5 |
| tower_crane | 2 | 100.0 | 100.0 | 100.0 |
| truck | 40 | 33.1 | 34.9 | 48.2 |
| wood_bar | 23 | 21.7 | 1.2 | 1.4 |
| wood_fence | 22 | 17.8 | 13.2 | 31.1 |
| worker | 119 | 15.9 | 4.8 | 7.8 |

## Region captioning (given the ground-truth box)

| Model | Regions | BLEU-4 | CIDEr-D | METEOR (exact) | Resource acc | Colour acc | Count acc | Template parse rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| florence-2-base-lora-epoch_4-regions | 1186 | 64.9 | 5.55 | 77.7 | 71.8 | 79.3 | 77.6 | 99.7 |
| florence-2-base-regions | 1186 | 0.0 | 0.16 | 3.1 | 14.8 | 22.5 | 0.1 | 0.0 |

