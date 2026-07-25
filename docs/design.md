# assay - design notes

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
rejected - activation quantization noise compounds across layers and hits the
token distribution far harder than argmax accuracy), while an excellent weight-only
(W4A16) checkpoint - ~100% mean accuracy retention, no task down more than a
fraction of a point - was nicked at +1.69%. A <=1% ppl bar is tighter than even a
near-lossless 4-bit weight quant achieves. 3% is a defensible "perplexity within 3%
of baseline" floor that clears good weight-only quant with margin and still
decisively rejects W4A4-class degradation. The accuracy bars stay strict on purpose.

## The recipe model

The certification science - which model, which quantization scheme, which
calibration set, which eval battery, which gate - lives in a `Recipe`: a frozen
dataclass in `src/assay/recipes.py`, keyed by slug in a `RECIPES` registry
(select with `ASSAY_RECIPE=<slug>`, default `qwen2_5_7b_instruct`). `RunConfig`
(in `config.py`) wraps a *resolved* Recipe and carries only the run's infra
plumbing - paths, GPU memory fraction, checkpoint repo. The split matters
because those two things change for different reasons and at different rates:
infra config is the same knob turned differently per run (a new output dir, a
different GPU headroom); the recipe is a reviewable, versioned claim about what
was actually certified. A recipe can also override the gate itself (the `gate`
field; `None` inherits `DEFAULT_GATE` via `gate_or_default`) - both shipped
recipes currently inherit the default.

Recipes are code-as-config, deliberately - consistent with the workspace-wide
"no on-disk runtime config" convention. Adding a model is a git diff (copy the
`template_example` Recipe, edit base_model/calib/eval/tags, give it a slug), not
a new config file format or a schema migration, and it gets the same review a
code change gets. `validate_recipe` runs inside `load_config` at startup - an
empty task list, an unqualified metric key, a bad `mode`, or a missing tag set
fails in milliseconds, before any paid GPU work, rather than as a mid-run crash
or, worse, a silently wrong gate.

A handful of scalar env vars (`ASSAY_NUM_CALIB`, `ASSAY_MAX_SEQ_LEN`,
`ASSAY_QUANT_SCHEME`, `ASSAY_BASE_MODEL`, `ASSAY_CALIB_DATASET`,
`ASSAY_CALIB_SPLIT`) still override individual recipe fields for a one-off run -
the 2 AM escape hatch - but the task battery, eval mode, and sampling are
structured science and are changed only by editing the recipe in git.

### Chat-mode eval is gate-blocking

Certifying an instruct or reasoning model on raw-completion measures the wrong
distribution - nobody prompts Qwen2.5-7B-Instruct without its chat template.
`Eval.mode` (`"chat"` | `"completion"`) is a recipe field, not a global switch:
for `mode == "chat"`, `run_job` applies the model's chat template and
few-shot-as-multiturn, and threads the recipe's own sampling (`gen_kwargs`,
`system_prompt`) through to the harness - and the gate's PASS/FAIL is computed
on those chat-mode numbers, the mode the model is actually used in, not on raw
completion. Baseline and quantized are built from the exact same `eval_kwargs`
dict, constructed once, so a gate delta reflects the quantization, not a
settings drift between the two runs. Both shipped recipes
(`qwen2_5_7b_instruct`, `r1_distill_qwen_7b`) use `mode="chat"`.

One recipe field is intentionally further along than its wiring: `prompt_prefix`
(a forced assistant-turn prefix - the R1 recipe sets `"<think>\n"`, matching
DeepSeek's documented usage) is validated (`validate_recipe` requires
`mode="chat"` to set it) but not yet threaded into the harness call -
rendering a forced prefix through the chat template correctly needs the actual
checkpoint's `chat_template` at hand, which is a metal-time, R1-cert-run
concern. This is a deliberate, documented gap, not a silent stub: the R1
recipe's numbers won't reflect the forced-prefix behavior until that wiring
lands at cert time.

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
  freely - the pipeline reads them fresh each run, so rotation costs nothing.
- **Bounded blast radius.** The pod self-terminates on every exit path (success or
  failure) and enforces a wall-clock backstop, so a crash or an unattended run can
  never leave a money-burning GPU alive. Ops artifacts (heartbeat, raw eval JSONs,
  delta table) are written to a directory that is structurally outside the published
  checkpoint directory, and `config.py` refuses a layout that would nest them inside
  it - so an operator log can never be swept into a public model upload.

## Escalation ladder (measured-need only)

Quantization quality work escalates only when the cheaper rung actually fails a bar,
never speculatively:

1. **PTQ (post-training quantization)** - what this pipeline does. One calibration
   pass, no training. Cheapest; sufficient for weight-only 4-bit on a 7B instruct
   model (proven).
2. **QAT (quantization-aware training)** - only if PTQ cannot clear the gate. More
   compute, recovers more quality by training with the quantization in the loop.
3. **Distillation** - only if a *smaller* student model is the actual goal, a
   different objective from "quantize this model faithfully."

## Calibration

PTQ sets activation *scales*, not weights, so a single general-purpose calibration
set (a few hundred chat samples) covers the model's behavior; a domain-specific
calibration set is warranted only if a measured domain gap appears, not by default.
The default is ~512 samples from a general instruct dataset, streamed so a large
source dataset never has to be materialized.

## Hardware support (what the checkpoint needs to serve)

Distinct from "GPU selection" below, which is about renting a GPU to *run assay
itself* - this is about what the *published checkpoint* needs to be served.

`assay` ships **weight-only** NVFP4 (NVFP4A16): 4-bit weights, 16-bit
activations. That scheme runs via vLLM's **FP4 Marlin** kernel and does **not**
require Blackwell GPUs or native FP4 tensor cores - on non-Blackwell hardware
the 4-bit weights are dequantized to 16-bit for the GEMM, trading some compute
throughput for the smaller memory footprint (the KV cache and activations are
16-bit either way, so the saving is on the linear weights). Only *activation*
FP4 (W4A4, which this pipeline does not produce) needs native Blackwell tensor
cores.

**Validated (measured through assay's own gate): Ada, sm_89 (e.g. L4,
RTX 4090).** Ampere/Hopper (>= sm_80) are expected to work by the same
weight-only Marlin path but are **not yet independently validated here**.
**Turing (sm_75) is excluded**: vLLM's source gate for FP4 Marlin support is
nominally `capability >= 75` (`is_fp4_marlin_supported()`), but that is the
kernel's *nominal* gate, not an empirical guarantee - there is a known vLLM
correctness bug where NVFP4 Marlin emits incorrect output on Turing, so Turing
is excluded until that is fixed and independently measured.

This is the `assay` measure-don't-vendor-claim posture applied to hardware
support, not just accuracy: the published model card
(`src/assay/publish.py::_hardware_section`) states exactly this measured floor,
generated automatically for every weight-only release rather than hand-edited
per checkpoint.

## GPU selection

See `gpu-selection.md`. Short version: quantize and eval are inference-shaped
(forward pass only), so VRAM-fit is the binding constraint, not FLOPS - pick the
cheapest card the model fits on, then fastest-per-dollar. This pipeline's RunPod
launcher defaults to Blackwell (RTX 5090, `EUR-IS-1`) because that is the
combination validated end-to-end for the full quantize -> eval -> gate run - an
infra default (`ASSAY_GPU_TYPE` / `ASSAY_REGION` override it), not an
architectural requirement of the weight-only NVFP4 this pipeline produces. See
"Hardware support" above for what the published checkpoint itself needs.

## Deferred (measured-need only)

Two things are deliberately not built yet, consistent with the escalation
ladder above - add scope only when a measured need shows up, not
speculatively:

- **CI-aware statistical gate.** The current gate compares point estimates
  (mean retention, per-task drop, perplexity increase) against fixed
  thresholds. A future upgrade would fold in each task's confidence interval
  (stderr) and seed count so a pass/fail decision accounts for benchmark noise,
  not just the raw number. `GateThresholds` (in `config.py`) is already shaped
  as a struct rather than bare floats specifically so that upgrade - e.g.
  `k_stderr`, `n_seeds` - slots in as new fields without a call-site signature
  change.
- **Generation-quality coverage.** The eval battery is accuracy + perplexity;
  it does not yet cover instruction-following quality (IFEval) or
  domain-shifted perplexity (how the checkpoint holds up on text unlike the
  calibration/eval sets). Both are real gaps for a certification pipeline, not
  addressed here because neither has failed a real release yet - the
  escalation ladder's measured-need principle applies to eval coverage too, not
  only to quantization technique.
