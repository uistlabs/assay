import pytest
import subprocess
from importlib.metadata import version as _dist_version
from pathlib import Path
from assay.manifest import (ManifestV1, StackPin, Hardware, EccWindow, Capture,
                            SCHEMA_VERSION, capture_stack, PinMismatchError,
                            image_only_names, parse_nvsmi_line, capture_hardware,
                            read_ecc_counters, HardwareCaptureError, build_ecc_window,
                            begin_capture, finalize, BeginCapture, ManifestEnvError)

def _sample_manifest():
    return ManifestV1(
        schema_version=SCHEMA_VERSION,
        image="ghcr.io/uist-labs/assay@sha256:" + "a" * 64,
        build_sha="deadbeef",
        stack=(StackPin(name="torch", pinned="2.7.1+cu129", observed="2.7.1+cu129"),),
        python="3.12.8",
        cuda_runtime="12.9",
        hardware=Hardware(gpu_name="NVIDIA GeForce RTX 5090", vram_total_mib=32607,
                          driver_version="575.57.08", cuda_driver="12.9",
                          ecc_supported=False, ecc_enabled=None, gpu_mem_util=0.85),
        ecc_window=EccWindow(counters_begin=None, counters_end=None,
                             uncorrected_delta=None, corrected_delta=None,
                             verdict="not-applicable"),
        capture=Capture(begin_utc="2026-07-28T00:00:00Z", end_utc="2026-07-28T04:00:00Z",
                        tool_queries=("nvidia-smi --query-gpu=...",)),
    )

def test_json_round_trip():
    m = _sample_manifest()
    assert ManifestV1.from_json(m.to_json()) == m

def test_schema_version_serialized():
    import json
    assert json.loads(_sample_manifest().to_json())["schema_version"] == 1

def test_capture_stack_against_real_constraints():
    # In the dev env, skip image-only pins (cu129 GPU stack) because sm_61 cannot
    # install them. In-pod callers will pass exclude=frozenset() to assert EVERY pin.
    # The dev env's pins hold by the existing test_build_pins contract.
    pins = capture_stack("deploy/constraints.txt", exclude=image_only_names("deploy/constraints.txt"))
    byname = {p.name: p for p in pins}
    assert "llmcompressor" in byname
    assert byname["llmcompressor"].pinned == byname["llmcompressor"].observed

def test_capture_stack_mismatch_raises(tmp_path):
    bad = tmp_path / "constraints.txt"
    bad.write_text("llmcompressor==0.0.1\n")
    with pytest.raises(PinMismatchError, match="llmcompressor"):
        capture_stack(str(bad))

def test_capture_stack_missing_package_raises(tmp_path):
    bad = tmp_path / "constraints.txt"
    bad.write_text("not-a-real-package-xyz==1.0\n")
    with pytest.raises(PinMismatchError, match="not-a-real-package-xyz"):
        capture_stack(str(bad))

def test_capture_stack_asserts_image_only_by_default(tmp_path):
    # Image-only pins (cu129 GPU stack) are asserted when no exclude is passed.
    # This proves the in-pod default (exclude=frozenset()) will catch F-030 ABI drift.
    # Use pytest pinned wrong with # image-only marker (it's definitely installed).
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("pytest==0.0.1  # image-only\n")
    with pytest.raises(PinMismatchError, match="pytest"):
        capture_stack(str(constraints))

def test_capture_stack_multi_offender_lists_all(tmp_path):
    # Multiple mismatches must ALL appear in the error message.
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("llmcompressor==0.0.1\ncompressed-tensors==0.0.1\n")
    with pytest.raises(PinMismatchError) as exc_info:
        capture_stack(str(constraints))
    error_msg = str(exc_info.value)
    assert "llmcompressor" in error_msg
    assert "compressed-tensors" in error_msg


FIX = Path("tests/fixtures/nvidia_smi")

def test_parse_captured_1070_line():
    line = (FIX / "gtx1070_query.csv").read_text().strip().splitlines()[0]
    d = parse_nvsmi_line(line)
    assert d["ecc_supported"] is False
    assert d["ecc_counters"] is None
    assert d["vram_total_mib"] > 0


def test_h100_fixture_pending_capture():
    pytest.skip("captured H100 fixture does not exist yet - capture on first HBM run")


def test_capture_hardware_failure_raises(tmp_path):
    def broken_run(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    with pytest.raises(HardwareCaptureError):
        capture_hardware(0.85, run=broken_run)


def test_ecc_clean():
    w = build_ecc_window(True, (0, 3), (0, 5))
    assert (w.verdict, w.uncorrected_delta, w.corrected_delta) == ("clean", 0, 2)


def test_ecc_void_on_uncorrected():
    assert build_ecc_window(True, (0, 0), (1, 0)).verdict == "void"


def test_ecc_counter_reset_is_not_captured_never_negative_clean():
    w = build_ecc_window(True, (5, 9), (2, 1))
    assert w.verdict == "not-captured"
    assert w.uncorrected_delta is None


def test_ecc_missing_read_on_ecc_hw_is_not_captured():
    assert build_ecc_window(True, (0, 0), None).verdict == "not-captured"


def test_ecc_not_applicable_without_ecc():
    assert build_ecc_window(False, None, None).verdict == "not-applicable"


def _real_pin_constraints(tmp_path, pkg="pytest") -> str:
    """A tmp constraints.txt whose single pin is a REAL installed dist at its
    REAL observed version (captured fixture, never invented) - so a
    begin_capture() call exercises its TRUE default (assert-everything, no
    exclude param) without hitting deploy/constraints.txt's image-only cu129
    pins, which the dev box legitimately does not have installed."""
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(f"{pkg}=={_dist_version(pkg)}\n")
    return str(constraints)


def _fake_nvsmi_run(line):
    def run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout=line + "\n", stderr="")
    return run


def test_begin_finalize_orchestrator_on_1070_fixture(tmp_path):
    line = (FIX / "gtx1070_query.csv").read_text().strip().splitlines()[0]
    fake_run = _fake_nvsmi_run(line)
    env = {"ASSAY_IMAGE": "ghcr.io/uist-labs/assay@sha256:" + "a" * 64,
           "ASSAY_BUILD_SHA": "deadbeef"}
    begin = begin_capture(env, 0.85, constraints_path=_real_pin_constraints(tmp_path),
                          run=fake_run)
    assert isinstance(begin, BeginCapture)
    manifest = finalize(begin, run=fake_run)
    assert manifest.image == env["ASSAY_IMAGE"]
    assert manifest.build_sha == "deadbeef"
    assert manifest.capture.end_utc is not None
    assert manifest.ecc_window.verdict == "not-applicable"  # 1070 has no ECC


def test_begin_capture_missing_image_raises_manifest_env_error(tmp_path):
    line = (FIX / "gtx1070_query.csv").read_text().strip().splitlines()[0]
    with pytest.raises(ManifestEnvError, match="ASSAY_IMAGE"):
        begin_capture({"ASSAY_BUILD_SHA": "deadbeef"}, 0.85,
                      constraints_path=_real_pin_constraints(tmp_path),
                      run=_fake_nvsmi_run(line))


def test_begin_capture_missing_build_sha_raises_manifest_env_error(tmp_path):
    line = (FIX / "gtx1070_query.csv").read_text().strip().splitlines()[0]
    with pytest.raises(ManifestEnvError, match="ASSAY_BUILD_SHA"):
        begin_capture({"ASSAY_IMAGE": "ghcr.io/uist-labs/assay@sha256:" + "a" * 64}, 0.85,
                      constraints_path=_real_pin_constraints(tmp_path),
                      run=_fake_nvsmi_run(line))


# --- whole-branch review FIX 1: begin-time ECC-counter hard fail ------------------

def test_begin_capture_hard_fails_when_ecc_supported_but_counters_unreadable(tmp_path):
    # ECC-capable GPU (mode reported as Enabled) but the volatile error-counter
    # fields are unreadable on this query - exactly the flake class the begin-time
    # hard fail exists to catch: without it, a flaked read silently disarms the
    # mid-run ECC check AND guarantees an end-of-run void after hours of paid GPU.
    # Spec capture-point 1 promises death BEFORE GPU spend, not after.
    line = "NVIDIA H100, 81920, 575.57.08, Enabled, [N/A], [N/A]"
    env = {"ASSAY_IMAGE": "ghcr.io/uist-labs/assay@sha256:" + "a" * 64,
           "ASSAY_BUILD_SHA": "deadbeef"}
    with pytest.raises(HardwareCaptureError, match="ECC"):
        begin_capture(env, 0.85, constraints_path=_real_pin_constraints(tmp_path),
                      run=_fake_nvsmi_run(line))


def test_begin_capture_non_ecc_hardware_unaffected_by_the_hard_fail(tmp_path):
    # The 1070 (ecc_supported=False): counters_begin is legitimately None and must
    # NOT raise - the hard fail is scoped to ecc_supported hardware only.
    line = (FIX / "gtx1070_query.csv").read_text().strip().splitlines()[0]
    env = {"ASSAY_IMAGE": "ghcr.io/uist-labs/assay@sha256:" + "a" * 64,
           "ASSAY_BUILD_SHA": "deadbeef"}
    begin = begin_capture(env, 0.85, constraints_path=_real_pin_constraints(tmp_path),
                          run=_fake_nvsmi_run(line))
    assert begin.counters_begin is None
