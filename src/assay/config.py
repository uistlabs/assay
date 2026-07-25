from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # recipes imports GateThresholds from here; runtime import is lazy
    from assay.recipes import Recipe

# Defensive mount detection. On our create_pod path (volume_mount_path=/runpod-volume)
# the volume DOES mount at /runpod-volume - verified on metal - so the configured
# default is correct and resolve_mount is a no-op in the common case. But RunPod pods
# can mount a network volume at /workspace under other create paths/conventions, and
# assay is our first pod-based consumer of a network volume, so this
# stays as cheap insurance against a convention change stranding the job.
_VOLUME_MOUNT_CANDIDATES = ("/runpod-volume", "/workspace")


@dataclass(frozen=True)
class GateThresholds:
    """The operator-tunable quality bar. Shaped as a struct (not bare floats) so a
    later CI-aware upgrade (k_stderr / n_seeds) slots in without a signature change.
    The 3% perplexity bar was earned on metal: W4A4 failed at +12.55% (correctly
    rejected), an excellent weight-only checkpoint was nicked at +1.69% under 1%.
    Every field is independently optional (None => skip that check) so a recipe
    composes exactly the gate it wants: Qwen keeps the point-gate below; a
    reasoning recipe sets k_stderr and drops the point checks."""
    min_mean_retention: float | None
    max_single_drop_pts: float | None
    max_ppl_increase: float | None
    k_stderr: float | None = None      # k in "fail iff drop > k*combined_stderr"; None => no significance check


DEFAULT_GATE = GateThresholds(
    min_mean_retention=0.99,
    max_single_drop_pts=2.0,
    max_ppl_increase=0.03,
)


@dataclass(frozen=True)
class TierProfile:
    """Eval-weakening for a run tier. cert = the certified methodology (no weakening);
    smoke = plumbing (seconds); dev = cheap ~30-min validation. Numbers are conservative
    starting points calibrated on the first metal dev run (spec s11) - tuning, not
    architecture. None on a field means 'use the recipe's value'."""
    limit: int | None          # eval docs per task; None = full battery
    max_gen_toks: int | None   # cap generation length; None = recipe's value
    repeats_cap: int | None    # cap avg@K per task; None = recipe's K


TIER_PROFILES: dict[str, "TierProfile"] = {
    "cert":  TierProfile(limit=None, max_gen_toks=None, repeats_cap=None),
    "dev":   TierProfile(limit=50,   max_gen_toks=4096, repeats_cap=8),
    "smoke": TierProfile(limit=2,    max_gen_toks=8,    repeats_cap=None),
}


# Env vars whose PRESENCE means the run diverged from the pristine git recipe, so a
# real (non-dry-run) publish is refused. The recipe-override set is EXACTLY what
# _apply_recipe_overrides consumes (calib x4, base_model, quant_scheme); plus the eval
# budget knob and the test-only stall-injection knob. Keep in lockstep with
# _apply_recipe_overrides - a new override there MUST be added here.
_NONPRISTINE_VARS = (
    "ASSAY_CALIB_DATASET", "ASSAY_CALIB_SPLIT", "ASSAY_NUM_CALIB",
    "ASSAY_MAX_SEQ_LEN", "ASSAY_BASE_MODEL", "ASSAY_QUANT_SCHEME",
    "ASSAY_LIMIT", "ASSAY_INJECT_STALL_AFTER",
)


def _is_pristine(env: Mapping[str, str], tier: str) -> bool:
    """True only for an unmodified cert-tier run - the SOLE config allowed to publish a
    real cert. Any non-cert tier OR any recognized runtime override forces dry-run.
    Presence-based (`in env`), matching how _apply_recipe_overrides applies overrides,
    so 'if it was applied, pristine breaks' holds; a lingering empty var forcing
    dry-run is the safe failure direction (spec s4)."""
    if tier != "cert":
        return False
    return not any(k in env for k in _NONPRISTINE_VARS)


def _require_env(env: Mapping[str, str], key: str) -> str:
    """Fetch a REQUIRED non-secret config value from env; raise if absent/empty.
    Fail-early: better to stop before a paid GPU run than midway or against the
    wrong namespace."""
    value = env.get(key)
    if not value:
        raise ValueError(f"{key} is required (set it in the launch env)")
    return value


@dataclass(frozen=True)
class RunConfig:
    """Per-run infrastructure config wrapping a resolved Recipe.

    The recipe carries the science (model, scheme, calibration, eval battery,
    gate); RunConfig carries the run's plumbing. artifacts_dir vs output_dir:
    kept as SIBLING directories (neither nests inside the other), not just
    separate fields, so the two independent wholesale uploads can never cross
    - publishing the checkpoint (which uploads output_dir wholesale to the
    public HF repo) can never sweep up ops-only files, and publish_artifacts
    (which uploads artifacts_dir wholesale to the private run-artifacts
    dataset, SP1) can never sweep up the multi-GB checkpoint. artifacts_dir
    holds the heartbeat log and durable eval/delta artifacts (I1/I2).
    output_dir holds ONLY what gets published: the NVFP4 checkpoint + README
    model card."""
    recipe: Recipe
    checkpoint_repo: str
    pipeline_url: str
    weights_path: str
    artifacts_dir: str
    output_dir: str
    heartbeat_path: str
    # Fraction of TOTAL VRAM the eval engine claims. vLLM v1 hard-fails at
    # startup if device-wide FREE memory is below gpu_mem_util * total, so
    # this knob is the 2 AM escape hatch when something else is squatting
    # on the GPU and the eval must squeeze under it. 0.85 on a 32 GiB card
    # = 26.65 GiB: 7B bf16 weights (~15.2) + CUDA graphs + KV cache, with
    # ~4.7 GiB device headroom for the parent's CUDA context + OS overhead.
    gpu_mem_util: float
    # Run tier: "cert" (default, the certified methodology), "dev" (cheap ~30-min
    # validation), or "smoke" (seconds-long plumbing). Selected by ASSAY_TIER; drives
    # eval-weakening (TIER_PROFILES) and, via the guard, publish dry-run.
    tier: str = "cert"
    # Effective eval limit (docs/task) after tier + ASSAY_LIMIT. None = full battery.
    # Never 1 (refused at load).
    eval_limit: int | None = None
    # Provenance guard: True only for an unmodified cert-tier run. Drives the single
    # publish chokepoint (dry_run = not pristine): a weakened/overridden run cannot
    # mint a real DOI cert.
    pristine: bool = True


def _is_within(child: str, parent: str) -> bool:
    """True if child == parent or child is nested under parent (normalized paths)."""
    child_n = os.path.normpath(child)
    parent_n = os.path.normpath(parent)
    return child_n == parent_n or child_n.startswith(parent_n + os.sep)


def _apply_recipe_overrides(recipe: Recipe, env: Mapping[str, str]) -> Recipe:
    """2 AM escape hatch: scalar env vars override recipe fields for a one-off,
    without editing code. Structured science (task lists, mode, sampling) is changed
    by editing the recipe in git."""
    calib = recipe.calib
    if "ASSAY_CALIB_DATASET" in env:
        calib = replace(calib, dataset=env["ASSAY_CALIB_DATASET"])
    if "ASSAY_CALIB_SPLIT" in env:
        calib = replace(calib, split=env["ASSAY_CALIB_SPLIT"])
    if "ASSAY_NUM_CALIB" in env:
        calib = replace(calib, num_samples=int(env["ASSAY_NUM_CALIB"]))
    if "ASSAY_MAX_SEQ_LEN" in env:
        calib = replace(calib, max_seq_len=int(env["ASSAY_MAX_SEQ_LEN"]))
    recipe = replace(recipe, calib=calib)
    if "ASSAY_BASE_MODEL" in env:
        recipe = replace(recipe, base_model=env["ASSAY_BASE_MODEL"])
    if "ASSAY_QUANT_SCHEME" in env:
        recipe = replace(recipe, quant_scheme=env["ASSAY_QUANT_SCHEME"])
    return recipe


def _resolve_tier(env: Mapping[str, str]) -> str:
    """ASSAY_TIER in {cert, dev, smoke}, default cert (the full certified run). Empty
    string -> cert (a lingering blank knob must not strand a burn). An unrecognized
    value RAISES rather than silently running a full cert (fail-loud, spec s8)."""
    raw = env.get("ASSAY_TIER", "").strip().lower()
    tier = raw or "cert"
    if tier not in TIER_PROFILES:
        raise ValueError(
            f"ASSAY_TIER={raw!r} is not one of {tuple(TIER_PROFILES)}; unset it for a "
            "full cert run, or pick 'dev' (cheap validation) or 'smoke' (plumbing)")
    return tier


def _resolve_eval_limit(env: Mapping[str, str], tier: str) -> int | None:
    """Effective eval limit (docs/task): the tier default, overridden by ASSAY_LIMIT
    (the ~30-min budget lever). Refuse an EFFECTIVE limit < 2 (covers 1, 0, and any
    negative): n=1 makes lm-eval emit an 'N/A' stderr, and sqrt(nan) turns the
    significance gate into an unconditional PASS (see evaluate._pick_stderr) - so a
    sub-2 limit can never validate anything; n<=0 doesn't evaluate at all."""
    raw = env.get("ASSAY_LIMIT", "").strip()
    if raw:
        try:
            limit = int(raw)
        except ValueError:
            raise ValueError(f"ASSAY_LIMIT={raw!r} must be an integer >= 2") from None
    else:
        limit = TIER_PROFILES[tier].limit
    if limit is not None and limit < 2:
        raise ValueError(
            f"effective eval limit == {limit} crashes the significance gate (n<2 => "
            "'N/A' stderr => unconditional PASS, or n<=0 evaluates nothing); use "
            ">= 2 (dev default 50)")
    return limit


def load_config(env: Mapping[str, str]) -> RunConfig:
    """Build RunConfig from a selected recipe + ASSAY_* env. Reads NO secrets.

    The recipe is selected by ASSAY_RECIPE (a slug in recipes.RECIPES; default
    qwen2_5_7b_instruct). ASSAY_CHECKPOINT_REPO is REQUIRED with no default on
    purpose - a default org would let a misconfigured run publish (or fail
    late) against someone else's namespace after a full paid quantize+eval.
    Name the scheme accurately in the repo name: a -NVFP4 suffix conventionally
    means W4A4, so a weight-only checkpoint should say -NVFP4A16."""
    # Lazy import: recipes.py imports GateThresholds/DEFAULT_GATE from this
    # module, so a module-level import here would be circular whenever
    # assay.recipes loads first.
    from assay.recipes import get_recipe, validate_recipe

    recipe = _apply_recipe_overrides(
        get_recipe(env.get("ASSAY_RECIPE", "qwen2_5_7b_instruct")), env)
    validate_recipe(recipe)

    output_dir = env.get("ASSAY_OUTPUT_DIR", "/runpod-volume/assay-out/checkpoint")
    artifacts_dir = env.get("ASSAY_ARTIFACTS_DIR", "/runpod-volume/assay-out/artifacts")
    # Derive the heartbeat default FROM artifacts_dir (one source of truth) so a
    # per-run ASSAY_ARTIFACTS_DIR (pod_entry appends the pod id) carries the heartbeat
    # into the same per-run dir - no cross-run bleed on a reused volume.
    heartbeat_path = env.get("ASSAY_HEARTBEAT",
                             os.path.join(artifacts_dir, "heartbeat.log"))

    # I2 guard: ops artifacts (heartbeat log, eval JSONs, delta table) must never
    # land inside output_dir - publish uploads output_dir wholesale to the public
    # HF repo, so anything nested there gets leaked. A stale ASSAY_OUTPUT_DIR
    # (e.g. left pointed at the old shared default) would silently reintroduce
    # exactly this leak, so this is checked every load, not just at the defaults.
    offenders = [p for p in (artifacts_dir, heartbeat_path) if _is_within(p, output_dir)]
    if offenders:
        raise ValueError(
            f"ASSAY_OUTPUT_DIR ({output_dir}) must not contain the ops artifacts "
            f"dir/heartbeat ({', '.join(offenders)}); ops logs would be published to "
            "the public HF repo. Set ASSAY_ARTIFACTS_DIR/ASSAY_OUTPUT_DIR so artifacts "
            "live outside the checkpoint dir.")

    # Bidirectional I2 guard: the reverse nesting is just as dangerous. SP1's
    # publish_artifacts uploads artifacts_dir wholesale to the PRIVATE run-artifacts
    # dataset on every run; if output_dir (the multi-GB NVFP4 checkpoint) is nested
    # inside artifacts_dir, that upload would sweep the checkpoint in too, blowing the
    # 300s timeout (so on a FAIL the small forensics may never upload) and wasting
    # paid pod time. artifacts_dir and output_dir must be sibling directories.
    if _is_within(output_dir, artifacts_dir):
        raise ValueError(
            f"ASSAY_OUTPUT_DIR ({output_dir}) must not be nested inside "
            f"ASSAY_ARTIFACTS_DIR ({artifacts_dir}): publish_artifacts uploads the "
            "artifacts dir wholesale to the private run-artifacts dataset, which would "
            "sweep the multi-GB checkpoint into it and blow the upload timeout. Keep "
            "them as sibling directories.")

    tier = _resolve_tier(env)
    eval_limit = _resolve_eval_limit(env, tier)
    pristine = _is_pristine(env, tier)

    # Optional link to the public assay pipeline repo, surfaced in the model card's
    # provenance line and BibTeX note. There is deliberately NO default: defaulting to
    # our org URL would be wrong for anyone running a fork, the same reason
    # ASSAY_CHECKPOINT_REPO has no default.
    pipeline_url = env.get("ASSAY_PIPELINE_URL", "")
    if pristine and not pipeline_url:
        # Gated on `pristine`, not on tier: pristine is exactly "this run will publish
        # a REAL card" (job.py derives dry_run from it), so the warning marks a
        # published artifact instead of nagging every dev/smoke run. A WARNING and not
        # a failure - a missing back-link degrades a card, it does not invalidate a
        # cert, and failing the run here would trade a cosmetic gap for a lost burn.
        print("assay.config: ASSAY_PIPELINE_URL unset on a live cert run - the "
              "published model card will name the pipeline with no pipeline link. "
              "Set it to the TAG-PINNED repo URL, e.g. "
              "https://github.com/uistlabs/assay/tree/v0.5.1 - a card is a frozen "
              "certification record, so the link must pin the exact certifying code.",
              file=sys.stderr)

    return RunConfig(
        recipe=recipe,
        checkpoint_repo=_require_env(env, "ASSAY_CHECKPOINT_REPO"),
        pipeline_url=pipeline_url,
        weights_path=env.get("ASSAY_WEIGHTS_PATH", "/runpod-volume/qwen2.5-7b-instruct"),
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        heartbeat_path=heartbeat_path,
        gpu_mem_util=float(env.get("ASSAY_GPU_MEM_UTIL", "0.85")),
        tier=tier,
        eval_limit=eval_limit,
        pristine=pristine,
    )


def resolve_mount(cfg: RunConfig, exists=os.path.exists) -> RunConfig:
    """Rebase volume paths onto the mount base where the weights actually exist.

    If cfg.weights_path is present, the configured base is correct - return as-is.
    Otherwise, if the weights sit at the same relative location under a different
    known volume mount (e.g. /workspace instead of /runpod-volume), rebase every
    volume-rooted path (weights, artifacts, output, heartbeat) onto that base so a
    hardcoded mount convention can't strand the job. recipe/checkpoint_repo/
    gpu_mem_util pass through untouched. Raise a clear error if the weights are
    found under no known mount. `exists` is injectable for tests."""
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
