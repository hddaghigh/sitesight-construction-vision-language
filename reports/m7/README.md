# M7: ONNX / int8 export, API, batch CLI, latency

Question: can the fine-tuned Florence-2-base (M5 adapter, merged) be served on a CPU, how much
faster and smaller does ONNX / int8 make it, and what does int8 cost in the project's own metric?

## Setup
- Three ONNX graphs (vision, encoder, decoder; opset 17, no KV cache) and a numpy greedy loop that
  reproduces the model's generation settings (forced BOS, no-repeat 3-gram, log-softmax confidence).
- int8 = `onnxruntime` dynamic quantisation, weights of MatMul/Gemm only. Convolutions and the
  embedding table stay fp32.
- 300 validation regions (seeded sample, seed 0; never the test split), one process, **cold**
  requests: the vision features are recomputed for every region, as for a new image.
  The first 3 calls per engine are warm-up and not timed (297 timed).
- Hardware (logged by the script): Colab CPU runtime, Intel Xeon @ 2.20 GHz, **2 logical CPUs**,
  Python 3.13, torch 2.14.1+cpu, onnxruntime 1.32.0. These are that machine's numbers, not a laptop's.
- The adapter was rounded to fp16 for upload to Colab (size limit of the browser upload), so scores
  can differ slightly from M5's fp32 adapter.

## Results (`bench.json`)
| Engine | Size | Cold p50 | Cold p95 | Strict resource | colour | count | state |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | n/a | 85.06 s | 88.10 s | 0.753 | 0.833 | 0.677 | 0.790 |
| ONNX fp32 | 1,242 MB | 12.07 s | 12.68 s | 0.753 | 0.833 | 0.677 | 0.790 |
| ONNX int8 | 574 MB | 9.45 s | 10.10 s | 0.747 | 0.833 | 0.669 | 0.783 |

Agreement with PyTorch: ONNX fp32 gives the identical caption for 99.7% of regions (mean absolute
confidence difference 0.0005). ONNX int8 gives the identical caption for **76%** (mean absolute
confidence difference 0.028).

Preprocessing parity (20 regions): the numpy image code differs from `Florence2Processor` by at
most 0.035 in normalised pixel units (resampling rounding), and the prompt ids are identical for 20
of 20.

## What this says
- ONNX fp32 is about 7 times faster than PyTorch on this CPU with the same field accuracy.
- int8 is 46% of the size and a further 1.28 times faster than ONNX fp32 (9.45 s vs 12.07 s), so
  about 9 times faster than PyTorch. Field accuracy drops by 2 regions of 300 on resource and
  state, and by one counted region on count. That is inside what a sample of 300 can resolve, so
  "no measurable accuracy loss" is not claimed; the loss is small and not shown to be zero.
- 24% of int8 captions differ in wording from PyTorch while the parsed fields mostly agree.
  Caption-text metrics (CIDEr etc.) were not run on the int8 model.
- Confidence moves more under int8 (0.028), and confidence was already not calibrated (M5 ECE 0.43).

## Caveats
- 9 to 12 s per region is still slow for interactive use: every request here is cold and the decoder
  has no KV cache. A cached image (several regions of one image) is faster; not benchmarked.
- 2 vCPUs only, one stream, threads not tuned (`threads: null`). No GPU and no batching.
- Only 300 validation regions; PyTorch/ONNX identical accuracy is on that sample only.
- The Docker image is built in CI (build and import check) but was **not run locally** (no Docker
  here), and the served model weights are not in the image; `SITESIGHT_MODEL_DIR` must mount them.
- A first export on real weights stored int8 at the same size as fp32 (weights were baked in as
  constants, so dynamic quantisation did nothing); caught by comparing sizes on Colab, fixed and
  covered by `test_weights_are_initializers_and_int8_really_stores_int8`.

## Reproduce
`python -m sitesight.serve.export --adapter ... --out export/` then
`python -m sitesight.serve.bench --annotations ... --images ... --adapter ... --onnx-dir export/
--out-dir bench/ --n-regions 300` (the Colab notebook is built by
`scripts/build_colab_notebook.py --kind m7`). The full run took about 9 hours, almost all of it the PyTorch engine.
