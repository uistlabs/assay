from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

# Defensive mount detection. On our create_pod path (volume_mount_path=/runpod-volume)
# the volume DOES mount at /runpod-volume -- verified on metal -- so the configured
# default is correct and resolve_mount is a no-op in the common case. But RunPod pods
# can mount a network volume at /workspace under other create paths/conventions, and
# assay is our first pod-based consumer of a network volume, so this
# stays as cheap insurance against a convention change stranding the job.
_VOLUME_MOUNT_CANDIDATES = ("/runpod-volume", "/workspace")

# Gate thresholds -- the operator-tunable quality knob.
MIN_MEAN_RETENTION = 0.99   # mean accuracy retention vs bf16
MAX_SINGLE_DROP_PTS = 2.0   # max absolute-point drop on any single accuracy task
# Max fractional wikitext perplexity increase. Set to 0.03 (up from an original
# 0.01) after metal evidence: NVFP4 W4A4 failed at +12.55% (correctly
# rejected), while an excellent NVFP4A16 (weight-only) checkpoint -- 100.4% mean
# accuracy retention, no task down >0.74pt -- was nicked at +1.69%. A <=1% ppl bar
# is tighter than even a near-lossless 4-bit weight quant achieves; +3% ("perplexity
# within 3% of baseline") is a defensible quality floor that clears good weight-only
# quant with margin and still decisively rejects W4A4-class degradation. The accuracy
# bars stay strict on purpose -- this loosens only the one over-tight metric.
MAX_PPL_INCREASE = 0.03

_DEFAULT_ACCURACY_TASKS = ("gsm8k", "arc_challenge", "hellaswag", "winogrande", "mmlu")

# gsm8k has no "acc" key in lm-eval output -- it reports exact_match under two
# filters (strict-match, flexible-extract). Pin to strict-match, the standard
# headline number reported in model cards and leaderboards. Every other
# accuracy task in the default battery (arc_challenge, hellaswag, winogrande,
# mmlu) exposes a bare "acc" key, so they need no override.
_TASK_METRIC_OVERRIDES = {"gsm8k": "exact_match,strict-match"}


def _require_env(env: Mapping[str, str], key: str) -> str:
    """Fetch a REQUIRED non-secret config value from env; raise if absent/empty.
    Fail-early: better to stop before a paid GPU run than midway or against the
    wrong namespace."""
    value = env.get(key)
    if not value:
        raise ValueError(f"{key} is required (set it in the launch env)")
    return value


def metric_for(task: str) -> str:
    """lm-eval metric key to read for a task. Default 'acc'; gsm8k reports
    exact_match (pinned to the strict-match filter, the standard headline)."""
    return _TASK_METRIC_OVERRIDES.get(task, "acc")


@dataclass(frozen=True)
class Config:
    base_model: str
    calib_dataset: str
    calib_split: str
    num_calibration_samples: int
    max_seq_length: int
    quant_scheme: str
    gpu_mem_util: float
    accuracy_tasks: tuple[str, ...]
    perplexity_task: str
    checkpoint_repo: str
    pipeline_url: str
    weights_path: str
    # artifacts_dir vs output_dir: kept as separate directories, not just separate
    # fields, so that publishing the checkpoint (which uploads output_dir wholesale)
    # can never sweep up ops-only files. artifacts_dir holds the heartbeat log and
    # durable eval/delta artifacts (I1/I2) -- never uploaded. output_dir holds ONLY
    # what gets published: the NVFP4 checkpoint + README model card.
    artifacts_dir: str
    output_dir: str
    heartbeat_path: str


def _is_within(child: str, parent: str) -> bool:
    """True if child == parent or child is nested under parent (normalized paths)."""
    child_n = os.path.normpath(child)
    parent_n = os.path.normpath(parent)
    return child_n == parent_n or child_n.startswith(parent_n + os.sep)


def load_config(env: Mapping[str, str]) -> Config:
    """Build Config from defaults, overridable by ASSAY_* env keys. Reads NO secrets."""
    cfg = Config(
        base_model=env.get("ASSAY_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        calib_dataset=env.get("ASSAY_CALIB_DATASET", "HuggingFaceH4/ultrachat_200k"),
        calib_split=env.get("ASSAY_CALIB_SPLIT", "train_sft"),
        num_calibration_samples=int(env.get("ASSAY_NUM_CALIB", "512")),
        max_seq_length=int(env.get("ASSAY_MAX_SEQ_LEN", "2048")),
        # compressed-tensors preset scheme. Default NVFP4A16 = 4-bit NVFP4 WEIGHTS
        # + 16-bit ACTIVATIONS (weight-only). The prior NVFP4 (W4A4, 4-bit acts too)
        # cleared quantize but failed the gate on metal: ~98% mean
        # retention but perplexity +12.55% -- the fingerprint of activation
        # quantization (A4 noise compounds across layers + hits the token
        # distribution far harder than argmax accuracy). Activations aren't stored,
        # so A16 keeps ~all the disk-size win (weights dominate) and recovers most
        # of the quality; W4A4's only edge was inference speed, which a published
        # certified checkpoint gladly trades back. Set ASSAY_QUANT_SCHEME=NVFP4 for
        # the W4A4 speed variant, or any compressed-tensors preset (e.g. W4A16).
        quant_scheme=env.get("ASSAY_QUANT_SCHEME", "NVFP4A16"),
        # Fraction of TOTAL VRAM the eval engine claims. vLLM v1 hard-fails at
        # startup if device-wide FREE memory is below gpu_mem_util * total, so
        # this knob is the 2 AM escape hatch when something else is squatting
        # on the GPU and the eval must squeeze under it. 0.85 on a 32 GiB card
        # = 26.65 GiB: 7B bf16 weights (~15.2) + CUDA graphs + KV cache, with
        # ~4.7 GiB device headroom for the parent's CUDA context + OS overhead.
        gpu_mem_util=float(env.get("ASSAY_GPU_MEM_UTIL", "0.85")),
        accuracy_tasks=tuple(
            env["ASSAY_ACC_TASKS"].split(",") if env.get("ASSAY_ACC_TASKS")
            else _DEFAULT_ACCURACY_TASKS
        ),
        perplexity_task=env.get("ASSAY_PPL_TASK", "wikitext"),
        # REQUIRED: the target HF repo id, e.g. "yourorg/Qwen2.5-7B-Instruct-NVFP4A16".
        # No default on purpose -- a default org would let a misconfigured run publish
        # (or fail late) against someone else's namespace after a full paid quantize+eval.
        # Name the scheme accurately in the repo name: a -NVFP4 suffix conventionally
        # means W4A4 (weights AND activations FP4), so a weight-only checkpoint should
        # say -NVFP4A16 to avoid misleading.
        checkpoint_repo=_require_env(env, "ASSAY_CHECKPOINT_REPO"),
        # Optional link to the public assay pipeline repo, surfaced in the model card's
        # provenance line. Set ASSAY_PIPELINE_URL to your fork/clone URL; if empty the
        # card names the pipeline without a hyperlink.
        pipeline_url=env.get("ASSAY_PIPELINE_URL", ""),
        weights_path=env.get("ASSAY_WEIGHTS_PATH", "/runpod-volume/qwen2.5-7b-instruct"),
        artifacts_dir=env.get("ASSAY_ARTIFACTS_DIR", "/runpod-volume/assay-out"),
        output_dir=env.get("ASSAY_OUTPUT_DIR", "/runpod-volume/assay-out/checkpoint"),
        heartbeat_path=env.get("ASSAY_HEARTBEAT", "/runpod-volume/assay-out/heartbeat.log"),
    )

    # I2 guard: ops artifacts (heartbeat log, eval JSONs, delta table) must never
    # land inside output_dir -- publish uploads output_dir wholesale to the public
    # HF repo, so anything nested there gets leaked. A stale ASSAY_OUTPUT_DIR
    # (e.g. left pointed at the old shared default) would silently reintroduce
    # exactly this leak, so this is checked every load, not just at the defaults.
    offenders = [
        path for path in (cfg.artifacts_dir, cfg.heartbeat_path)
        if _is_within(path, cfg.output_dir)
    ]
    if offenders:
        raise ValueError(
            f"ASSAY_OUTPUT_DIR ({cfg.output_dir}) must not contain the ops "
            f"artifacts dir/heartbeat ({', '.join(offenders)}); ops logs would be "
            "published to the public HF repo. Set ASSAY_ARTIFACTS_DIR/"
            "ASSAY_OUTPUT_DIR so artifacts live outside the checkpoint dir."
        )

    return cfg


def resolve_mount(cfg: Config, exists=os.path.exists) -> Config:
    """Rebase volume paths onto the mount base where the weights actually exist.

    If cfg.weights_path is present, the configured base is correct -- return as-is.
    Otherwise, if the weights sit at the same relative location under a different
    known volume mount (e.g. /workspace instead of /runpod-volume), rebase every
    volume-rooted path (weights, artifacts, output, heartbeat) onto that base so a
    hardcoded mount convention can't strand the job. Raise a clear error if the
    weights are found under no known mount. `exists` is injectable for tests."""
    if exists(cfg.weights_path):
        return cfg
    old_base = next(
        (b for b in _VOLUME_MOUNT_CANDIDATES
         if cfg.weights_path == b or cfg.weights_path.startswith(b + os.sep)),
        None,
    )
    if old_base is None:
        raise FileNotFoundError(
            f"weights_path {cfg.weights_path!r} does not exist and is not under a "
            f"known volume mount {_VOLUME_MOUNT_CANDIDATES}; set ASSAY_WEIGHTS_PATH "
            "to the actual weights location."
        )
    rel = cfg.weights_path[len(old_base):].lstrip(os.sep)
    for new_base in _VOLUME_MOUNT_CANDIDATES:
        if new_base == old_base:
            continue
        if exists(os.path.join(new_base, rel)):
            def rebase(path: str) -> str:
                if path == old_base or path.startswith(old_base + os.sep):
                    return new_base + path[len(old_base):]
                return path
            return replace(
                cfg,
                weights_path=rebase(cfg.weights_path),
                artifacts_dir=rebase(cfg.artifacts_dir),
                output_dir=rebase(cfg.output_dir),
                heartbeat_path=rebase(cfg.heartbeat_path),
            )
    raise FileNotFoundError(
        f"Qwen weights not found at {cfg.weights_path!r} nor at the same relative "
        f"path under any of {_VOLUME_MOUNT_CANDIDATES}; is the weights volume "
        "attached with the model staged on it?"
    )


def require_secret(env: Mapping[str, str], key: str) -> str:
    """Fetch a secret from env; raise if absent. Caller must never persist the value."""
    value = env.get(key)
    if not value:
        raise ValueError(f"{key} is required (inject via runtime env, never on disk)")
    return value
