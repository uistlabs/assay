# assay

**Quantize a language model, certify it against a benchmark gate, and publish it
only if it passes.**

`assay` is an NVFP4 quantization + *benchmark-gated publishing* pipeline. It
quantizes a Hugging Face model with [llm-compressor](https://github.com/vllm-project/llm-compressor),
evaluates the bf16 baseline and the quantized checkpoint side by side with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and
uploads the result to the Hugging Face Hub **only if** it clears a hard, stated
accuracy bar. The name is the point: an assay is the test that certifies a refined
metal's purity -- here, the benchmark gate is what earns a checkpoint the right to
ship.

> **What this is.** A working reference implementation, proven end-to-end on real
> Blackwell hardware. It is not a turnkey product: it targets one cloud (RunPod),
> one GPU class (Blackwell / NVFP4), and it assumes you can stage weights and build
> a container image. A motivated engineer can reproduce a certified checkpoint from
> this repo in an afternoon; it will not one-click for you. The prerequisites below
> are honest about the walls.

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
in isolated subprocesses, applies the gate, and -- only on a pass -- writes a model
card and uploads. It tears the pod down on every exit path and enforces a wall-clock
backstop, so an unattended run can never burn a GPU indefinitely.

## Prerequisites (the honest five)

1. **A RunPod account + API key.** The launcher creates a GPU pod via the RunPod
   SDK. Key is injected at launch, never stored.
2. **A network volume, in your chosen region, with the base model's bf16 weights
   pre-staged on it.** assay mounts the volume and reads the weights from it. **No
   staging tool is provided** -- you upload the weights yourself, once. This is the
   prerequisite most likely to strand you.
3. **Your own PUBLIC container image built from `deploy/Dockerfile`** (~15-18 GB),
   pushed to a registry the pod can pull anonymously (e.g. `ghcr.io/<you>/assay`).
   The default create path passes no registry credentials, so the image must be
   public. The image carries no secrets by construction.
4. **A Hugging Face org (or user) + a write-scoped token** for the target repo.
5. **`ASSAY_CHECKPOINT_REPO` set** to your target repo id (e.g.
   `yourorg/Qwen2.5-7B-Instruct-NVFP4A16`). Required -- there is no default org.

Blackwell-only: NVFP4 runs only on Blackwell GPUs. The defaults target an RTX 5090
in `EUR-IS-1`; override with `ASSAY_GPU_TYPE` / `ASSAY_REGION`.

## Quickstart

```bash
uv sync                       # install deps + the package (needed before launch.sh)
uv run pytest                 # fast, GPU-free unit suite -- proves the logic

# export secrets + config, then launch
export RUNPOD_API_KEY=...      # dedicated, pod-scoped, rotatable
export HF_TOKEN=...            # fine-grained, write-scoped to the target repo
export ASSAY_VOLUME_ID=...     # your pre-staged weights volume id
export ASSAY_IMAGE=ghcr.io/<you>/assay:0.2.0   # your PUBLIC image
export ASSAY_CHECKPOINT_REPO=yourorg/Qwen2.5-7B-Instruct-NVFP4A16
uv run scripts/launch.sh --dry-run   # inspect the pod payload (secrets redacted)
uv run scripts/launch.sh             # real run
```

Every non-secret knob is an `ASSAY_*` env var (see `src/assay/config.py`); a
reduced-battery smoke run needs no image rebuild, e.g.
`ASSAY_NUM_CALIB=8 ASSAY_ACC_TASKS=arc_challenge`.

## The gate

Publish requires all of (defaults, tunable in `config.py`):

- mean accuracy retention >= 99%,
- no single accuracy task down more than 2.0 points,
- wikitext perplexity increase <= 3%.

The 3% perplexity bar is deliberate and metal-earned: it clears excellent
weight-only quantization with margin while decisively rejecting the perplexity
blow-up that fully-4-bit (W4A4) activation quantization causes. See
[`docs/design.md`](docs/design.md).

## Limitations

- **RunPod-only.** The pod-control layer targets the RunPod SDK. No other backend.
- **Blackwell-only NVFP4.** Validated on RTX 5090 (sm_120). NVFP4 requires Blackwell.
- **Single-GPU.** Sized for a 7B model on one 32 GB card.
- **Weights staging is on you.** No tool to upload weights to the volume is included.
- **Reference implementation.** See the framing note at the top.

## Secrets model

Secrets live in the launch environment only -- never on disk, never in the image,
never committed. The launcher disables shell trace before any secret-bearing line;
the heartbeat log and durable failure traceback redact known secrets before writing.
Keys are least-privilege and meant to be rotated freely. See
[`docs/design.md`](docs/design.md) for the full threat model.

## Docs

- [`docs/design.md`](docs/design.md) -- gate thesis, secrets/threat model, escalation ladder.
- [`docs/gpu-selection.md`](docs/gpu-selection.md) -- the GPU-selection heuristic.
- [`deploy/README.md`](deploy/README.md) -- building the batch image.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
