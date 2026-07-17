import pytest

from assay.runpod_ctl import (
    build_pod_payload, self_terminate, RTX_5090, REGION,
    CONTAINER_DISK_GB, MIN_MEMORY_GB, MIN_VCPU, CLOUD_TYPE,
    ALLOWED_CUDA_VERSIONS, MIN_DOWNLOAD_MBPS,
)


def test_payload_sets_capacity_knobs_explicitly():
    # Regression: the runpod SDK silently substitutes container_disk_in_gb=10 for
    # None, which under our ~18GB image left no headroom -> ENOSPC during model load
    # -> the original silent teardown. These money/failure-bearing knobs must be set
    # explicitly in the payload (and thus visible in --dry-run), never SDK defaults.
    p = build_pod_payload(image="img", volume_id="v", env_keys=["HF_TOKEN"], env={"HF_TOKEN": "t"})
    assert p["containerDiskInGb"] == CONTAINER_DISK_GB >= 40
    assert p["minMemoryInGb"] == MIN_MEMORY_GB >= 32
    assert p["minVcpuCount"] == MIN_VCPU
    assert p["cloudType"] == CLOUD_TYPE


def test_payload_pins_allowed_cuda_versions():
    # Scheduler-level guarantee: only place us on hosts whose driver supports our cu129
    # stack. FLOOR IS 12.9: 12.8/r570 hosts fail at eval-engine start
    # with cudaErrorUnsupportedPtxVersion because vLLM's FA2 sm_120 kernel ships PTX the
    # r570 driver's JIT is too old to load. 12.8 MUST be excluded -- it's a guaranteed
    # failure, not a wider pool. See runpod_ctl.ALLOWED_CUDA_VERSIONS for the full RCA.
    p = build_pod_payload(image="img", volume_id="v", env_keys=["HF_TOKEN"], env={"HF_TOKEN": "t"})
    assert p["allowedCudaVersions"] == ALLOWED_CUDA_VERSIONS
    assert "12.8" not in p["allowedCudaVersions"]
    assert "12.9" in p["allowedCudaVersions"]


def test_payload_pins_min_download():
    # Host-side pull-reliability lever: gate scheduling on measured
    # download throughput so we never land on a ~32Mbps host where the multi-GB image
    # cold-pull stalls 60-90 min and EOFs. Must be well above the slow tier.
    p = build_pod_payload(image="img", volume_id="v", env_keys=["HF_TOKEN"], env={"HF_TOKEN": "t"})
    assert p["minDownload"] == MIN_DOWNLOAD_MBPS
    assert MIN_DOWNLOAD_MBPS >= 100


def test_payload_pins_blackwell_and_region():
    p = build_pod_payload(
        image="ghcr.io/uistlabs/assay:0.2.0",
        volume_id="vol123",
        env_keys=["HF_TOKEN"],
        env={"HF_TOKEN": "tok"},
    )
    assert p["gpuTypeId"] == RTX_5090
    assert p["dataCenterId"] == REGION
    assert p["networkVolumeId"] == "vol123"


def test_gpu_type_and_region_default_to_blackwell_eur():
    p = build_pod_payload(image="img", volume_id="v", env_keys=["HF_TOKEN"],
                          env={"HF_TOKEN": "t"})
    assert p["gpuTypeId"] == RTX_5090
    assert p["dataCenterId"] == REGION


def test_gpu_type_and_region_env_override():
    p = build_pod_payload(
        image="img", volume_id="v", env_keys=["HF_TOKEN"],
        env={"HF_TOKEN": "t", "ASSAY_GPU_TYPE": "NVIDIA B200", "ASSAY_REGION": "US-CA-2"},
    )
    assert p["gpuTypeId"] == "NVIDIA B200"
    assert p["dataCenterId"] == "US-CA-2"


def test_payload_injects_only_requested_env():
    p = build_pod_payload(
        image="img", volume_id="v", env_keys=["HF_TOKEN"],
        env={"HF_TOKEN": "tok", "OTHER": "nope"},
    )
    keys = {e["key"] for e in p["env"]}
    assert keys == {"HF_TOKEN"}
    assert p["env"][0]["value"] == "tok"


def test_payload_missing_env_value_raises():
    with pytest.raises(ValueError, match="HF_TOKEN"):
        build_pod_payload(image="i", volume_id="v", env_keys=["HF_TOKEN"], env={})


def test_payload_includes_extra_env_alongside_secrets():
    # I3: ASSAY_* config knobs from the operator's shell must reach the pod
    # payload as non-secret env, without going through require_secret.
    p = build_pod_payload(
        image="img", volume_id="v", env_keys=["HF_TOKEN"],
        env={"HF_TOKEN": "tok"},
        extra_env={"ASSAY_NUM_CALIB": "8", "ASSAY_BASE_MODEL": "foo/bar"},
    )
    by_key = {e["key"]: e["value"] for e in p["env"]}
    assert by_key["HF_TOKEN"] == "tok"
    assert by_key["ASSAY_NUM_CALIB"] == "8"
    assert by_key["ASSAY_BASE_MODEL"] == "foo/bar"


def test_payload_without_extra_env_is_unchanged():
    # omitting extra_env must behave exactly as before I3
    p = build_pod_payload(
        image="img", volume_id="v", env_keys=["HF_TOKEN"],
        env={"HF_TOKEN": "tok", "OTHER": "nope"},
    )
    keys = {e["key"] for e in p["env"]}
    assert keys == {"HF_TOKEN"}


def test_self_terminate_targets_own_pod_id():
    calls = {}

    class FakeApi:
        def terminate_pod(self, pod_id):
            calls["pod_id"] = pod_id

    self_terminate({"RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": "k"}, api=FakeApi())
    assert calls["pod_id"] == "pod-abc"


def test_self_terminate_missing_id_raises():
    with pytest.raises(ValueError, match="RUNPOD_POD_ID"):
        self_terminate({"RUNPOD_API_KEY": "k"}, api=object())
