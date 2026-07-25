# GPU selection for quantization + evaluation

Quantization and evaluation are **inference-shaped** workloads - forward passes
only, no optimizer state, no gradients. The binding constraint is therefore
**VRAM (does the model fit), not FLOPS.** Memory budget ~= weights + activations
+ KV cache.

## The rule

1. **Fit first.** Pick the cheapest card whose VRAM holds the model with headroom.
   - The *quantization* step can go sequential (llm-compressor offloads
     layer-by-layer to CPU), letting a small card quantize a big model - trades
     wall-clock for memory.
   - The *evaluation* step has no such escape hatch: the model must fit to run.
2. **Then fastest-per-dollar** among the cards that fit.

## When a bigger card matters

- The model stops fitting: a 70B in bf16 is ~140GB -> an 80GB A100/H100 or
  multi-GPU (shard the model), or the sequential-offload trick for the quant step.
- KV cache / long context dwarfs the weights (big batch or long sequences).
- You are compute-bound, not memory-bound - i.e. *training* (QAT/distillation),
  not this workload.

## The NVFP4 hard gate

The Blackwell requirement depends on *which* NVFP4 you produce:

- **Weight-only (NVFP4A16)** - the scheme this project certifies - quantizes
  only the weights and runs via vLLM's FP4 Marlin kernel, which dequantizes to
  16-bit for the GEMM. It does **not** require Blackwell: validated on Ada
  (sm_89, e.g. L4/RTX 4090); Ampere/Hopper (>= sm_80) are expected to work the
  same way but are not yet independently validated here; Turing (sm_75) is
  excluded due to a known vLLM NVFP4-Marlin correctness bug. Producing AND
  serving a weight-only checkpoint can both run on any CUDA GPU that fits the
  model - no Blackwell needed either side.
- **Activation-quantized (W4A4)** - a documented escape hatch, not the
  certified default - quantizes activations too, so it needs native FP4
  tensor cores at inference time. Producing the weights can still run on any
  CUDA GPU, but serving requires a Blackwell GPU regardless of model size.

For a checkpoint you intend to serve, validate on the same silicon (or at
least the same weight-only-vs-Blackwell category) that serves it.

## This project's instance

`Qwen2.5-7B-Instruct` (bf16 ~15GB) fits a 24GB or 32GB card trivially, and the
certified NVFP4A16 scheme this project produces does not require Blackwell to
serve. The RTX 5090 (Blackwell, sm_120) was chosen because it was the card
RunPod stocks in EUR-IS-1 where the weights volume lives - it is a superset
of what weight-only NVFP4A16 needs, not a requirement of the scheme itself.
