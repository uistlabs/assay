from __future__ import annotations

import time
from collections.abc import Mapping

from assay.config import require_secret

# Defaults for the target GPU + datacenter. Overridable per-run via ASSAY_GPU_TYPE /
# ASSAY_REGION (RunPod gpu-type id and datacenter id). Defaults target the only
# Blackwell card + region combination validated on metal for the NVFP4 gate; change
# both together if you move hardware (e.g. a different Blackwell SKU / region where
# your weights volume lives).
RTX_5090 = "NVIDIA GeForce RTX 5090"
REGION = "EUR-IS-1"

# Capacity knobs, set explicitly rather than riding SDK defaults - runpod's
# create_pod silently substitutes container_disk_in_gb=10 and min_memory_in_gb=1
# for None, and a 10GB container disk under a ~18GB image left near-zero headroom,
# so a stray write during model load hit ENOSPC -> OSError -> teardown (the
# original silent-fail root cause). Never ride a default for a money-bearing,
# failure-bearing knob; surfaced in the dry-run payload too.
CONTAINER_DISK_GB = 40   # generous headroom over the image for caches/tmp/compile
MIN_MEMORY_GB = 32       # 15GB model in CPU RAM + eval engine, comfortably
MIN_VCPU = 8
CLOUD_TYPE = "SECURE"    # network volumes exist only in secure-cloud DCs anyway

# Constrain WHICH hosts RunPod schedules us onto by their driver's supported CUDA
# version, so we are never placed on a too-old-driver host. Scheduler-level complement
# to the bare-ubi9 base (which removed the cuda-compat error-804 mechanism, verified on metal).
#
# FLOOR IS 12.9, NOT 12.8 (metal-proven). We first tried [12.8,12.9,13.0]
# reasoning that cu129 torch runs on 12.8/r570 hosts via CUDA minor-version compatibility.
# That is true for torch's precompiled cubin kernels - but vLLM's FlashAttention-2 kernel
# for Blackwell (sm_120) ships as PTX that the HOST DRIVER must JIT at load, and an r570
# (CUDA 12.8) driver's PTX compiler is too old for cu129's PTX ISA -> at eval-engine start
# it dies with `torch.AcceleratorError: CUDA error: the provided PTX was compiled with an
# unsupported toolchain` (cudaErrorUnsupportedPtxVersion). Deterministic on r570 hosts;
# the runs that worked landed on r580/CUDA-13.0 hosts. So the cu129 stack genuinely needs
# an r575+/CUDA-12.9 driver, and 12.8 must be excluded. Narrower pool, but a 12.8 host is
# a guaranteed failure; the launch retry loop covers transient can't-schedule.
ALLOWED_CUDA_VERSIONS = ["12.9", "13.0"]

# Minimum host download throughput (Mbps). The batch image is multi-GB; cold-pulling
# it from GHCR to EUR-IS-1 landed us on pathologically slow hosts (~32 Mbps observed
# = ~4 MB/s) where the big layer took 60-90 min and then dropped with `unexpected EOF`
# (verified on metal). RunPod can gate scheduling on measured host network speed, so pin
# a floor well above the slow tier: 300 Mbps (~37 MB/s) pulls the image in a couple of
# minutes and excludes the hosts that could never finish. Tunable DOWN if it ever
# starves the (already narrow: EUR-IS-1 + 5090 + secure) pool - launch surfaces a
# can't-schedule as a retry, not a silent hang. This is the host-side lever; the image
# slim/split is the complementary image-side lever (bottleneck could be either end).
MIN_DOWNLOAD_MBPS = 300

# Reader-pod (F-040) defaults. The image is digest-pinned for the same reason the
# launcher rejects a bare-tag ASSAY_IMAGE (a mutable tag served stale burned a run);
# python:3.12-slim is the CPU-pod image the 07-27 post-mortem reader proved.
READER_IMAGE = ("docker.io/library/python:3.12-slim"
                "@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de")
READER_VCPU = 2
READER_DISK_GB = 10

# Reader pods are named "{READER_NAME_PREFIX}{main_pod_id}" (build_reader_payload
# below) - hoisted to a module constant (whole-branch review FIX 2) so
# fleet_stray_warning_lines exempts readers from the same source that names them,
# never a hand-copied second literal.
READER_NAME_PREFIX = "assay-reader-"


def build_pod_payload(*, image: str, volume_id: str, env_keys: list[str],
                      env: Mapping[str, str],
                      extra_env: Mapping[str, str] | None = None) -> dict:
    """Construct the create-pod request. Env values are pulled at call time for
    exactly env_keys - a missing value is an error, never a silent default.

    extra_env (I3) carries non-secret ASSAY_* config overrides from the
    operator's shell straight through to the pod - appended as-is, no
    require_secret check and no redaction, so a reduced-battery smoke run
    (e.g. ASSAY_NUM_CALIB=8) doesn't need an image rebuild.

    NOTE env shape: this is the GraphQL surface, so `env` below is a LIST of
    {key, value} dicts. build_reader_payload's `env` is the REST surface's flat
    dict instead - the two are intentionally different wire shapes and must not
    be unified into one helper."""
    gpu_type = env.get("ASSAY_GPU_TYPE", RTX_5090)
    region = env.get("ASSAY_REGION", REGION)
    resolved = [{"key": k, "value": require_secret(env, k)} for k in env_keys]
    resolved += [{"key": k, "value": v} for k, v in (extra_env or {}).items()]
    return {
        "imageName": image,
        "gpuTypeId": gpu_type,
        "dataCenterId": region,
        "networkVolumeId": volume_id,
        "gpuCount": 1,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "minMemoryInGb": MIN_MEMORY_GB,
        "minVcpuCount": MIN_VCPU,
        "cloudType": CLOUD_TYPE,
        "allowedCudaVersions": ALLOWED_CUDA_VERSIONS,
        "minDownload": MIN_DOWNLOAD_MBPS,
        "env": resolved,
    }


def _validated_positive_number(name: str, value: str) -> str:
    """Validate a numeric env override before it reaches the pod (whole-branch review
    finding): an unparseable or non-positive ASSAY_READER_TTL silently voids the TTL
    backstop, and the same for ASSAY_READER_INTERVAL storms HF's commit rate limit
    instead. Raise naming the var and the bad value - launch.sh's warn-guard turns
    that into a loud, $0 skip of the reader rather than a reader that launches broken."""
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a positive number, got: {value!r}") from None
    if not parsed > 0:
        raise ValueError(f"{name} must be a positive number, got: {value!r}")
    return value


def build_reader_payload(*, main_pod_id: str, volume_id: str, dataset: str,
                         loop_b64: str, snapshot_b64: str,
                         env: Mapping[str, str]) -> dict:
    """REST POST /v1/pods body for the F-040 reader pod (CPU: computeType + vcpuCount
    REPLACE the gpu fields - the shape that worked first try 2026-07-27). The loop and
    cycle scripts ride base64-embedded in dockerStartCmd so the pod needs no image
    build and no network fetch of our code. Secrets are pulled at call time like
    build_pod_payload; the reader carries the SAME per-session key the main pod does
    (poll + self-terminate), never the management key.

    NOTE env shape: `env` below is a flat {key: value} dict, the REST surface's
    shape - build_pod_payload's GraphQL surface uses a LIST of {key, value} dicts
    instead. Intentionally different wire shapes; do not unify."""
    image = env.get("ASSAY_READER_IMAGE", READER_IMAGE)
    if "@sha256:" not in image:
        raise ValueError(f"reader image must be digest-pinned, got: {image}")
    interval = _validated_positive_number(
        "ASSAY_READER_INTERVAL", env.get("ASSAY_READER_INTERVAL", "600"))
    ttl = _validated_positive_number(
        "ASSAY_READER_TTL", env.get("ASSAY_READER_TTL", "86400"))
    boot_escalate_min = _validated_positive_number(
        "ASSAY_READER_BOOT_ESCALATE_MIN",
        env.get("ASSAY_READER_BOOT_ESCALATE_MIN", "30"))
    bootstrap = (
        "mkdir -p /tmp/reader"
        f" && echo {loop_b64} | base64 -d > /tmp/reader/reader_loop.sh"
        f" && echo {snapshot_b64} | base64 -d > /tmp/reader/reader_snapshot.py"
        " && bash /tmp/reader/reader_loop.sh"
    )
    return {
        "name": f"{READER_NAME_PREFIX}{main_pod_id}",
        "computeType": "CPU",
        "vcpuCount": READER_VCPU,
        "imageName": image,
        "containerDiskInGb": READER_DISK_GB,
        "cloudType": CLOUD_TYPE,
        "dataCenterIds": [env.get("ASSAY_REGION", REGION)],
        "networkVolumeId": volume_id,
        "volumeMountPath": "/runpod-volume",
        "dockerStartCmd": ["bash", "-lc", bootstrap],
        "env": {
            "RUNPOD_API_KEY": require_secret(env, "RUNPOD_API_KEY"),
            "HF_TOKEN": require_secret(env, "HF_TOKEN"),
            "ASSAY_READER_MAIN_POD_ID": main_pod_id,
            "ASSAY_ARTIFACTS_DATASET": dataset,
            "ASSAY_ARTIFACTS_DIR": env.get(
                "ASSAY_ARTIFACTS_DIR", "/runpod-volume/assay-out/artifacts"),
            "ASSAY_READER_INTERVAL": interval,
            "ASSAY_READER_TTL": ttl,
            "ASSAY_READER_BOOT_ESCALATE_MIN": boot_escalate_min,
        },
    }


def self_terminate(env: Mapping[str, str], api=None, *,
                   attempts: int = 5, backoff_s: float = 10.0, sleep=time.sleep) -> None:
    """Terminate this pod by its own injected id. `api` is injectable for tests;
    in production it is the `runpod` module itself, authenticated by setting the
    module-level `runpod.api_key` from RUNPOD_API_KEY (SDK 1.10.1 has no
    runpod.Api() class - termination goes through the module-level
    runpod.terminate_pod()).

    terminate_pod is the ONE call that stops billing, and it is a network round-trip
    to the RunPod GraphQL API. A single transient 5xx/timeout at that moment would
    leave the pod un-terminated - RunPod then restarts the exited container and the
    whole multi-hour quantize+eval re-runs, billing the GPU until an operator notices.
    So retry with a bounded linear backoff before giving up. pod_entry.sh's `|| echo`
    still swallows a final, persistent failure (the exit code must survive), but a
    transient blip no longer costs a full re-run. `attempts`/`backoff_s`/`sleep` are
    injectable so tests exercise the retry without real waits. Worst-case added pod
    time on total API failure is ~(attempts-1)*backoff_s, negligible vs a re-run."""
    pod_id = env.get("RUNPOD_POD_ID")
    if not pod_id:
        raise ValueError("RUNPOD_POD_ID is required to self-terminate")
    if attempts < 1:
        # Guard the empty-loop case: without this the tail `raise last_exc` would
        # `raise None` (TypeError), masking the real "termination failed" signal.
        raise ValueError(f"self_terminate: attempts must be >= 1, got {attempts}")
    key = require_secret(env, "RUNPOD_API_KEY")  # also strips (F-043) - keep the
    if api is None:                              # stripped value on the live branch
        import runpod
        runpod.api_key = key
        api = runpod
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            api.terminate_pod(pod_id)
            return
        except Exception as exc:  # noqa: BLE001 - any API/network error is retryable here
            last_exc = exc
            if attempt < attempts - 1:
                sleep(backoff_s)
    # Exhausted every attempt: re-raise so pod_entry.sh logs it (the `|| echo` path).
    raise last_exc


# --- Fleet-mode pre-launch pod check (D8/F-009 T9) ---------------------------------
#
# Ken's ruling (2026-07-28): pod listing is pod-control domain, so this lives here,
# not in preflight.py (which is deliberately NO-NETWORK - see its module docstring;
# the plan that dispatched this task assumed preflight.py already held a "0 assay
# pods" check to extend, but that check never existed as code - it was a MANUAL
# runbook step, `runpod.get_pods()` run by hand before every launch. This is the
# first time it is automated, built fleet-aware from the start per D8).


def list_assay_pods(env: Mapping[str, str], api=None) -> list[dict] | None:
    """Read-only `api.get_pods()` listing, filtered to pods assay itself created:
    main launch pods (named "assay-nvfp4") and F-040 reader pods (named
    "assay-reader-<id>"). The name-prefix filter matters because this RunPod
    account is shared with other UIST products' pods - without it, an
    unrelated product's pod would register as a false-positive stray.

    `api` is injectable, matching `self_terminate`'s seam; `env` supplies
    RUNPOD_API_KEY the same way. Bounded, read-only, warn-only (D8, same F-035
    posture as `balance_warning_line`): returns None on ANY failure - missing
    key, API error, malformed response - rather than raising. A pod list that
    cannot be fetched must never block or crash a launch. Callers must treat
    None as "nothing to check", never as "zero pods confirmed"."""
    if api is None:  # pragma: no cover - real SDK only on the operator's box
        key = env.get("RUNPOD_API_KEY", "")
        if not key:
            return None
        import runpod
        runpod.api_key = key
        api = runpod
    try:
        pods = api.get_pods()
    except Exception:  # noqa: BLE001 - warn-only by design
        return None
    if pods is None:
        return None
    return [p for p in pods if str(p.get("name", "")).startswith("assay")]


def parse_fleet_expected(raw: "str | None") -> "set[str] | None":
    """Parse ASSAY_FLEET_EXPECTED (D8/F-009 T9). Three states, matching Ken's
    ruling exactly - documented here at the consumption site:

    - `raw is None` (var UNSET): returns None, meaning the caller must run NO
      check at all. Preserves byte-identical current behavior - the "0 assay
      pods" rule was never automated (manual runbook only), so an operator who
      has not opted into fleet mode sees no change.
    - `raw == ""` (var SET but blank): returns an EMPTY set - "expect ZERO
      assay pods", automating the runbook's original rule for the first time.
      Any running assay pod warns.
    - `raw` a non-empty comma-separated string: returns the parsed name set -
      running assay pods must be a SUBSET of it; anything else is a stray.

    PRISTINE NOTE: ASSAY_FLEET_EXPECTED is deliberately absent from
    config.py's `_RECIPE_OVERRIDES` / `_NONRECIPE_OVERRIDES` (the F-026
    one-table rule driving `_NONPRISTINE_VARS`). It gates an operator
    pod-inventory safety WARNING at launch time, never the recipe or the
    measured environment, so it must never affect pristine - see the matching
    note on `_resolve_fleet_est_hours` in cost/collect.py for the balance half.
    """
    if raw is None:
        return None
    return {name.strip() for name in raw.split(",") if name.strip()}


def fleet_stray_warning_lines(expected_names, pods) -> list[str]:
    """Pure warn-only fleet-membership check (D8/F-009 T9): one line per
    running assay pod whose name is NOT in `expected_names`. Never raises,
    never blocks - the F-035 philosophy applied to pod inventory: strays are
    surfaced, the operator decides. `pods` is the already-fetched,
    already-filtered list from `list_assay_pods` (a plain argument rather than
    a live call, so this half stays pure and trivially testable without the
    SDK).

    Reader pods (READER_NAME_PREFIX, whole-branch review FIX 2) are exempt from
    this check by construction: they are default-on, launch-derived, TTL-
    backstopped (ASSAY_READER_TTL self-terminates them regardless), and their
    names embed the main pod's id, which is unknowable in advance - they can
    never be listed in ASSAY_FLEET_EXPECTED, so treating them as strays would
    warn on every single launch."""
    expected = set(expected_names)
    return [
        f"[fleet] WARN: stray pod running: {p.get('name') or p.get('id', '?')} "
        "(not in ASSAY_FLEET_EXPECTED) - operator decides, launch proceeds."
        for p in (pods or [])
        if p.get("name") not in expected
        and not str(p.get("name", "")).startswith(READER_NAME_PREFIX)
    ]
