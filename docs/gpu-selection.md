# GPU selection for quantization + evaluation

Quantization and evaluation are **inference-shaped** workloads -- forward passes
only, no optimizer state, no gradients. The binding constraint is therefore
**VRAM (does the model fit), not FLOPS.** Memory budget ~= weights + activations
+ KV cache.

## The rule

1. **Fit first.** Pick the cheapest card whose VRAM holds the model with headroom.
   - The *quantization* step can go sequential (llm-compressor offloads
     layer-by-layer to CPU), letting a small card quantize a big model -- trades
     wall-clock for memory.
   - The *evaluation* step has no such escape hatch: the model must fit to run.
2. **Then fastest-per-dollar** among the cards that fit.

## When a bigger card matters

- The model stops fitting: a 70B in bf16 is ~140GB -> an 80GB A100/H100 or
  multi-GPU (shard the model), or the sequential-offload trick for the quant step.
- KV cache / long context dwarfs the weights (big batch or long sequences).
- You are compute-bound, not memory-bound -- i.e. *training* (QAT/distillation),
  not this workload.

## The NVFP4 hard gate

Running or validating NVFP4 requires a Blackwell GPU regardless of model size --
FP4 tensor cores exist only there. Producing the weights can run on any CUDA GPU,
but you cannot load/serve the result off Blackwell. For a checkpoint you intend to
serve, validate on the same silicon that serves it.

## This project's instance

`Qwen2.5-7B-Instruct` (bf16 ~15GB) fits a 24GB or 32GB card trivially -- so the
choice was *forced* by the NVFP4 gate (need Blackwell) to the RTX 5090, the only
Blackwell card RunPod stocks in EUR-IS-1 where the weights volume lives. It is
the same Blackwell silicon (sm_120) a production vLLM deployment would serve on,
so we validate on serving-representative hardware.
