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


def build_pod_payload(*, image: str, volume_id: str, env_keys: list[str],
                      env: Mapping[str, str],
                      extra_env: Mapping[str, str] | None = None) -> dict:
    """Construct the create-pod request. Env values are pulled at call time for
    exactly env_keys - a missing value is an error, never a silent default.

    extra_env (I3) carries non-secret ASSAY_* config overrides from the
    operator's shell straight through to the pod - appended as-is, no
    require_secret check and no redaction, so a reduced-battery smoke run
    (e.g. ASSAY_NUM_CALIB=8) doesn't need an image rebuild."""
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
    require_secret(env, "RUNPOD_API_KEY")  # presence check; value used to build api
    if api is None:  # pragma: no cover - exercised only on the live pod
        import runpod
        runpod.api_key = env["RUNPOD_API_KEY"]
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
