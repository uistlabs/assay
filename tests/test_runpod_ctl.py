import pytest

from assay.runpod_ctl import (
    build_pod_payload, self_terminate, RTX_5090, REGION,
    CONTAINER_DISK_GB, MIN_MEMORY_GB, MIN_VCPU, CLOUD_TYPE,
    ALLOWED_CUDA_VERSIONS, MIN_DOWNLOAD_MBPS,
    list_assay_pods, parse_fleet_expected, fleet_stray_warning_lines,
    READER_NAME_PREFIX, build_reader_payload,
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
    # r570 driver's JIT is too old to load. 12.8 MUST be excluded - it's a guaranteed
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
        image="ghcr.io/uist-labs/assay:0.2.0",
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


def test_self_terminate_live_branch_uses_stripped_key(monkeypatch):
    # F-043 residual (delta review, 07-28): the api-is-None live branch set
    # runpod.api_key from the RAW env value, discarding require_secret's strip -
    # on the one call that stops billing. Pin the stripped value end to end.
    import sys
    import types
    stub = types.ModuleType("runpod")
    calls = {}
    stub.terminate_pod = lambda pod_id: calls.setdefault("pod_id", pod_id)
    monkeypatch.setitem(sys.modules, "runpod", stub)
    self_terminate({"RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": "k-live\n"},
                   api=None)
    assert calls["pod_id"] == "pod-abc"
    assert stub.api_key == "k-live"


def test_self_terminate_retries_transient_failure_then_succeeds():
    # The sole billing-stopping call must survive a transient API blip: fail twice,
    # succeed on the third attempt. Without the retry a single 5xx leaves the pod
    # billing until it restarts and re-runs the whole job.
    attempts = {"n": 0}
    sleeps = []

    class FlakyApi:
        def terminate_pod(self, pod_id):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient 502 from RunPod")

    self_terminate({"RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": "k"},
                   api=FlakyApi(), backoff_s=0.0, sleep=sleeps.append)
    assert attempts["n"] == 3
    assert len(sleeps) == 2  # slept between the two failures, not after success


def test_self_terminate_raises_after_exhausting_attempts():
    # Persistent failure across all attempts must re-raise (pod_entry.sh's `|| echo`
    # logs it) rather than silently returning as if the pod were terminated.
    calls = {"n": 0}

    class DeadApi:
        def terminate_pod(self, pod_id):
            calls["n"] += 1
            raise RuntimeError("RunPod API down")

    with pytest.raises(RuntimeError, match="RunPod API down"):
        self_terminate({"RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": "k"},
                       api=DeadApi(), attempts=3, backoff_s=0.0, sleep=lambda *_: None)
    assert calls["n"] == 3


def test_self_terminate_rejects_nonpositive_attempts():
    # attempts<=0 skips the retry loop entirely, leaving last_exc=None; the tail
    # `raise last_exc` would then `raise None` -> TypeError, masking the real signal.
    # Fail fast with a clear error instead.
    with pytest.raises(ValueError, match="attempts"):
        self_terminate({"RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": "k"},
                       api=object(), attempts=0)


# --- F-009 T9 (D8): fleet-mode pod-subset check ------------------------------------

def test_parse_fleet_expected_unset_returns_none():
    # UNSET means "run no check at all" - byte-identical to pre-D8 reality, where
    # the "0 assay pods" rule lived only in the operator runbook, never in code.
    assert parse_fleet_expected(None) is None


def test_parse_fleet_expected_empty_string_means_expect_zero():
    # SET but blank automates the runbook's original "0 assay pods" rule for the
    # first time: an empty expected-set, so ANY running assay pod is a stray.
    assert parse_fleet_expected("") == set()


def test_parse_fleet_expected_parses_comma_separated_names():
    assert parse_fleet_expected("assay-nvfp4-a, assay-nvfp4-b") == {
        "assay-nvfp4-a", "assay-nvfp4-b"}


def test_fleet_stray_warning_lines_empty_when_subset():
    pods = [{"name": "assay-nvfp4-a"}]
    assert fleet_stray_warning_lines({"assay-nvfp4-a", "assay-nvfp4-b"}, pods) == []


def test_fleet_stray_warning_lines_names_the_stray():
    pods = [{"name": "assay-nvfp4-a"}, {"name": "assay-leftover"}]
    lines = fleet_stray_warning_lines({"assay-nvfp4-a"}, pods)
    assert len(lines) == 1
    assert "assay-leftover" in lines[0]
    assert "WARN" in lines[0]


def test_fleet_stray_warning_lines_expect_zero_warns_on_any_pod():
    pods = [{"name": "assay-nvfp4"}]
    lines = fleet_stray_warning_lines(set(), pods)
    assert len(lines) == 1 and "assay-nvfp4" in lines[0]


def test_fleet_stray_warning_lines_no_pods_is_silent():
    assert fleet_stray_warning_lines(set(), []) == []
    assert fleet_stray_warning_lines({"assay-nvfp4"}, None) == []


# --- whole-branch review FIX 2: reader pods are not strays -----------------------

def test_fleet_stray_warning_lines_exempts_reader_pods():
    # Readers are default-on, launch-derived, TTL-backstopped, and their names embed
    # an unknowable-in-advance pod id - they can NEVER be listed in
    # ASSAY_FLEET_EXPECTED, so they must never be reported as strays.
    pods = [{"name": "assay-nvfp4"}, {"name": f"{READER_NAME_PREFIX}xyz123"}]
    lines = fleet_stray_warning_lines({"assay-nvfp4"}, pods)
    assert lines == []


def test_fleet_stray_warning_lines_still_warns_on_a_genuine_stray_alongside_a_reader():
    pods = [{"name": f"{READER_NAME_PREFIX}xyz123"}, {"name": "assay-leftover"}]
    lines = fleet_stray_warning_lines(set(), pods)
    assert len(lines) == 1
    assert "assay-leftover" in lines[0]


def test_reader_name_prefix_matches_build_reader_payload_naming():
    # Same source, no hand-copied second literal: the prefix the exemption checks
    # against must be the exact one build_reader_payload uses to name a reader.
    import base64
    p = build_reader_payload(
        main_pod_id="mainpod1", volume_id="vol1", dataset="org/run-artifacts",
        loop_b64=base64.b64encode(b"loop").decode(),
        snapshot_b64=base64.b64encode(b"snap").decode(),
        env={"RUNPOD_API_KEY": "rk-test", "HF_TOKEN": "hf-test"})
    assert p["name"].startswith(READER_NAME_PREFIX)


def test_list_assay_pods_filters_to_assay_named_pods():
    class FakeApi:
        def get_pods(self):
            return [{"name": "assay-nvfp4", "id": "p1"},
                    {"name": "assay-reader-p1", "id": "p2"},
                    {"name": "other-product-worker", "id": "p3"}]

    pods = list_assay_pods({"RUNPOD_API_KEY": "k"}, api=FakeApi())
    names = {p["name"] for p in pods}
    assert names == {"assay-nvfp4", "assay-reader-p1"}


def test_list_assay_pods_swallows_api_errors():
    class Boom:
        def get_pods(self):
            raise RuntimeError("RunPod API down")

    assert list_assay_pods({"RUNPOD_API_KEY": "k"}, api=Boom()) is None


def test_list_assay_pods_without_key_returns_none():
    # No api injected AND no RUNPOD_API_KEY: must degrade to None (warn-only), never
    # raise or attempt a live call with no credential.
    assert list_assay_pods({}, api=None) is None
