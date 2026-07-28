# assay batch image

Dedicated batch image for the NVFP4 quantize + lm-eval run.

Latest-stable coherent stack, pinned exactly and bumped as a unit:

| component | version | role |
|-----------|---------|------|
| vLLM | 0.25.0 (cu129 wheel, torch 2.11) | reader / eval engine (Blackwell sm_120, no cu130 driver floor) |
| torch | 2.11.0+cu129 | - |
| llmcompressor | 0.12.0 | writer (NVFP4 quantization) |
| compressed-tensors | 0.17.1 | NVFP4 on-disk format contract (shared write + read) |

**Deliberate ct pin:** vLLM 0.25 pins `compressed-tensors==0.17.0`, but no released
llmcompressor pins 0.17.0 (it skips 0.16.0 -> 0.17.1). So 0.17.1 is the closest coherent
pairing; its diff over 0.17.0 is perf/validation, not the NVFP4 layout. We run 0.17.1 on
both sides and validate load-correctness on metal (the build gate documents this).

Build + push:

    podman build -t ghcr.io/uist-labs/assay:0.6.0 -f deploy/Dockerfile .
    podman push ghcr.io/uist-labs/assay:0.6.0

Weights are NOT baked - they mount from a pre-staged RunPod network volume at runtime.
