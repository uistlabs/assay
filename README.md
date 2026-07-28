# assay

**Quantize a language model, certify it against a benchmark gate, and publish it
only if it passes.**

`assay` is an NVFP4 quantization + *benchmark-gated publishing* pipeline. It
quantizes a Hugging Face model with [llm-compressor](https://github.com/vllm-project/llm-compressor),
evaluates the bf16 baseline and the quantized checkpoint side by side with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and
uploads the result to the Hugging Face Hub **only if** it clears a hard, stated
accuracy bar. The name is the point: an assay is the test that certifies a refined
metal's purity - here, the benchmark gate is what earns a checkpoint the right to
ship.

> **What this is.** A working reference implementation, proven end-to-end on real
> Blackwell hardware. It is not a turnkey product: it targets one cloud (RunPod)
> and its build pipeline defaults to one GPU class (Blackwell / RTX 5090), and it
> assumes you can stage weights and build a container image. (The *checkpoints it
> produces* are weight-only NVFP4 and do not require Blackwell to serve - see
> [Hardware support](#hardware-support).) A motivated engineer can reproduce a
> certified checkpoint from this repo in an afternoon; it will not one-click for
> you. The prerequisites below are honest about the walls.

## Proof

The pipeline shipped a real certified checkpoint:
**[uist-labs/Qwen2.5-7B-Instruct-NVFP4A16](https://huggingface.co/uist-labs/Qwen2.5-7B-Instruct-NVFP4A16)**
(weight-only NVFP4). Measured deltas vs the bf16 baseline (identical harness):

| task | retention |
|------|-----------|
| gsm8k, arc_challenge | improved / within noise |
| hellaswag, winogrande, mmlu | ~99% (worst task down < 1 point) |
| mean accuracy retention | ~100% |
| wikitext perplexity | +1.7% (bar: <= 3%) |

It passed the gate; that is why it exists on the Hub.

## How it works

```
quantize (llm-compressor, NVFP4)
   -> eval baseline (bf16)  -\
   -> eval quantized         -> gate (accuracy + perplexity) -> publish IFF pass
```

The whole run is one self-terminating batch job on a rented GPU pod: it asserts a
usable GPU up front (never silently falls back to CPU), quantizes, runs both evals
in isolated subprocesses, applies the gate, and - only on a pass - writes a model
card and uploads. It tears the pod down on every exit path and enforces a wall-clock
backstop, so an unattended run can never burn a GPU indefinitely.

## Recipes

Every run is driven by a `Recipe` - the certification science (base model,
quantization scheme, calibration set, eval battery, gate) - as a frozen dataclass
in [`src/assay/recipes.py`](src/assay/recipes.py), keyed by slug in a `RECIPES`
registry. Infra plumbing (paths, GPU memory fraction, checkpoint repo) is a
separate `RunConfig` that wraps the resolved recipe; see
[`docs/design.md`](docs/design.md) for why the two are split.

Select a recipe with `ASSAY_RECIPE=<slug>` (default `qwen2_5_7b_instruct`).
Shipped slugs:

- `qwen2_5_7b_instruct` - Qwen2.5-7B-Instruct, chat-mode eval (gsm8k,
  arc_challenge, hellaswag, winogrande, mmlu + wikitext perplexity). This is the
  recipe behind the [Proof](#proof) checkpoint above.
- `r1_distill_qwen_7b` - DeepSeek-R1-Distill-Qwen-7B, chat-mode reasoning eval
  (aime24, aime25, minerva_math500, gpqa_diamond_cot_zeroshot) with the model's
  documented sampling (temperature 0.6, top_p 0.95, forced `<think>\n` prefix).
- `template_example` - not a real model; copy this block to add one.

**Adding a model = copying `template_example`, editing
base_model/calib/eval/tags, and giving it a new slug.** `load_config` runs
`validate_recipe` on the resolved recipe at startup - an empty task list, a bad
`mode`, an unqualified metric key, or a missing tag set fails in milliseconds,
before any paid GPU work.

For a one-off without editing code, a handful of scalar env vars still override
individual recipe fields: `ASSAY_NUM_CALIB`, `ASSAY_MAX_SEQ_LEN`,
`ASSAY_QUANT_SCHEME`, `ASSAY_BASE_MODEL`, `ASSAY_CALIB_DATASET`,
`ASSAY_CALIB_SPLIT`. The task battery, eval mode, and sampling are structured
science and are changed by editing the recipe in git, not by env var.

## Prerequisites (the honest five)

1. **A RunPod account + API key.** The launcher creates a GPU pod via the RunPod
   SDK. Key is injected at launch, never stored.
2. **A network volume, in your chosen region, with the base model's bf16 weights
   pre-staged on it.** assay mounts the volume and reads the weights from it. **No
   staging tool is provided** - you upload the weights yourself, once. This is the
   prerequisite most likely to strand you.
3. **Your own PUBLIC container image built from `deploy/Dockerfile`** (~15-18 GB),
   pushed to a registry the pod can pull anonymously (e.g. `ghcr.io/<you>/assay`).
   The default create path passes no registry credentials, so the image must be
   public. The image carries no secrets by construction.
4. **A Hugging Face org (or user) + a write-scoped token** for the target repo.
5. **`ASSAY_CHECKPOINT_REPO` set** to your target repo id (e.g.
   `yourorg/Qwen2.5-7B-Instruct-NVFP4A16`). Required - there is no default org.

Build pipeline defaults to Blackwell: the launcher's defaults target an RTX 5090
in `EUR-IS-1`; override with `ASSAY_GPU_TYPE` / `ASSAY_REGION`. That is an infra
default, not an architectural requirement of the weight-only NVFP4 this pipeline
produces - see [Hardware support](#hardware-support) for what the *published
checkpoint* actually needs to serve.

## Hardware support

The **build pipeline** (this repo, RunPod-launched) currently defaults to
Blackwell (RTX 5090) - the environment validated end-to-end for
quantize -> eval -> gate. **The checkpoints it produces are a different story:**
this project ships **weight-only** NVFP4 (NVFP4A16), which runs via vLLM's
**FP4 Marlin** kernel and does **not** require Blackwell GPUs or native FP4
tensor cores - non-Blackwell hardware dequantizes the 4-bit weights to 16-bit
for the GEMM (the KV cache and activations stay 16-bit either way, so the
saving is on the linear weights).

**Validated (measured through assay's own gate): Ada (sm_89 - e.g. L4,
RTX 4090).** Ampere/Hopper (>= sm_80) are expected to work by the same
weight-only Marlin path but are **not yet independently validated here**.
**Turing (sm_75) is excluded**: a known vLLM issue makes the NVFP4 Marlin GEMM
emit incorrect output on Turing, so that architecture is not claimed until it
is fixed and independently measured. Every published checkpoint's model card
states this same measured floor - see [`docs/design.md`](docs/design.md) for
the full reasoning.

## Quickstart

```bash
uv sync                       # install deps + the package (needed before launch.sh)
uv run pytest                 # fast, GPU-free unit suite - proves the logic

# export secrets + config, then launch
export RUNPOD_API_KEY=...      # dedicated, pod-scoped, rotatable
export HF_TOKEN=...            # fine-grained, write-scoped to the target repo
export ASSAY_VOLUME_ID=...     # your pre-staged weights volume id
export ASSAY_IMAGE=ghcr.io/<you>/assay:0.6.0   # your PUBLIC image
export ASSAY_CHECKPOINT_REPO=yourorg/Qwen2.5-7B-Instruct-NVFP4A16
export ASSAY_PIPELINE_URL=https://github.com/uist-labs/assay/tree/v0.6.0  # tag-pinned
uv run scripts/launch.sh --dry-run   # inspect the pod payload (secrets redacted)
uv run scripts/launch.sh             # real run
```

Every non-secret knob is an `ASSAY_*` env var (recipe selection + scalar
overrides in `src/assay/recipes.py` / `config.py`; infra knobs in `config.py`);
a cheaper smoke run needs no image rebuild, e.g. `ASSAY_NUM_CALIB=8` or
`ASSAY_RECIPE=r1_distill_qwen_7b`.

`ASSAY_PIPELINE_URL` is optional but recommended for a real publish: it is
surfaced in the model card's provenance line and BibTeX note, and a cert-tier
run warns if it is unset. Pin it to a TAG, not `main` - a model card is a frozen
certification record, so the link must resolve to the exact code that produced
it, not to whatever the repo looks like when someone reads the card later.

## The gate

Publish requires all of (defaults live in `config.py`; a recipe may override
them via its own `gate` field - see [Recipes](#recipes)):

- mean accuracy retention >= 99%,
- no single accuracy task down more than 2.0 points,
- wikitext perplexity increase <= 3%.

The 3% perplexity bar is deliberate and metal-earned: it clears excellent
weight-only quantization with margin while decisively rejecting the perplexity
blow-up that fully-4-bit (W4A4) activation quantization causes. See
[`docs/design.md`](docs/design.md).

## Limitations

- **RunPod-only.** The pod-control layer targets the RunPod SDK. No other backend.
- **Build pipeline defaults to Blackwell.** The RunPod launcher's default pod is
  an RTX 5090 (sm_120) - the only combination validated end-to-end for the full
  quantize -> eval -> gate run; override with `ASSAY_GPU_TYPE` / `ASSAY_REGION`.
  This does not mean the *output* needs Blackwell - see
  [Hardware support](#hardware-support).
- **Single-GPU.** Sized for a 7B model on one 32 GB card.
- **Weights staging is on you.** No tool to upload weights to the volume is included.
- **Reference implementation.** See the framing note at the top.

## Secrets model

Secrets live in the launch environment only - never on disk, never in the image,
never committed. The launcher disables shell trace before any secret-bearing line;
the heartbeat log and durable failure traceback redact known secrets before writing.
Keys are least-privilege and meant to be rotated freely. See
[`docs/design.md`](docs/design.md) for the full threat model.

## Docs

- [`docs/design.md`](docs/design.md) - gate thesis, secrets/threat model, escalation ladder.
- [`docs/gpu-selection.md`](docs/gpu-selection.md) - the GPU-selection heuristic.
- [`deploy/README.md`](deploy/README.md) - building the batch image.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
