# assay -- design notes

Why the pipeline is shaped the way it is. The README covers how to run it; this
covers the reasoning a reviewer or contributor would want.

## The certification thesis (why a gate)

Quantization is a lossy transform. The common failure mode in published quantized
checkpoints is a silent one: the weights load, generation looks plausible, and the
real accuracy loss only shows up later on a benchmark nobody ran before shipping.

`assay` inverts that. It treats a quantized checkpoint as *unproven until measured*
and refuses to publish one that has not cleared a hard, stated accuracy bar against
its own bf16 baseline, on the identical evaluation harness. The name is the point:
an assay is the test that certifies a refined metal's purity. Publishing is a
side effect of passing; a failing checkpoint leaves its measured deltas on disk and
is never uploaded.

The gate (defaults, all operator-tunable in `config.py`):
- mean accuracy retention across the task battery >= 99%,
- no single accuracy task down more than 2.0 absolute points,
- wikitext perplexity increase <= 3%.

The perplexity bar is 3%, not the tighter 1% we first tried, and that number was
earned on metal: a fully-4-bit (W4A4) checkpoint failed it at +12.55% (correctly
rejected -- activation quantization noise compounds across layers and hits the
token distribution far harder than argmax accuracy), while an excellent weight-only
(W4A16) checkpoint -- ~100% mean accuracy retention, no task down more than a
fraction of a point -- was nicked at +1.69%. A <=1% ppl bar is tighter than even a
near-lossless 4-bit weight quant achieves. 3% is a defensible "perplexity within 3%
of baseline" floor that clears good weight-only quant with margin and still
decisively rejects W4A4-class degradation. The accuracy bars stay strict on purpose.

## Secrets / threat model

The pipeline runs on a rented GPU pod. Its trust model:

- **No secret ever touches disk.** The two secrets (a RunPod API key and an HF write
  token) are injected into the pod's environment at launch and read from there.
  Nothing is written to a file, baked into the image, or committed. The launcher
  disables shell xtrace before any secret-bearing line so `bash -x` cannot leak them
  to a trace; the heartbeat log redacts any known secret substring before writing;
  the durable failure traceback is routed through the same redaction.
- **Least-privilege, rotatable keys.** The RunPod key is dedicated and pod-scoped;
  the HF token is write-scoped to the target repo. Both are meant to be rotated
  freely -- the pipeline reads them fresh each run, so rotation costs nothing.
- **Bounded blast radius.** The pod self-terminates on every exit path (success or
  failure) and enforces a wall-clock backstop, so a crash or an unattended run can
  never leave a money-burning GPU alive. Ops artifacts (heartbeat, raw eval JSONs,
  delta table) are written to a directory that is structurally outside the published
  checkpoint directory, and `config.py` refuses a layout that would nest them inside
  it -- so an operator log can never be swept into a public model upload.

## Escalation ladder (measured-need only)

Quantization quality work escalates only when the cheaper rung actually fails a bar,
never speculatively:

1. **PTQ (post-training quantization)** -- what this pipeline does. One calibration
   pass, no training. Cheapest; sufficient for weight-only 4-bit on a 7B instruct
   model (proven).
2. **QAT (quantization-aware training)** -- only if PTQ cannot clear the gate. More
   compute, recovers more quality by training with the quantization in the loop.
3. **Distillation** -- only if a *smaller* student model is the actual goal, a
   different objective from "quantize this model faithfully."

## Calibration

PTQ sets activation *scales*, not weights, so a single general-purpose calibration
set (a few hundred chat samples) covers the model's behavior; a domain-specific
calibration set is warranted only if a measured domain gap appears, not by default.
The default is ~512 samples from a general instruct dataset, streamed so a large
source dataset never has to be materialized.

## GPU selection

See `gpu-selection.md`. Short version: quantize and eval are inference-shaped
(forward pass only), so VRAM-fit is the binding constraint, not FLOPS -- pick the
cheapest card the model fits on, then fastest-per-dollar. NVFP4 adds a hard
Blackwell requirement regardless of model size.
