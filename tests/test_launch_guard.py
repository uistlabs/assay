"""launch.sh must reject a bare-tag ASSAY_IMAGE: a mutable tag can be served stale from
RunPod's image cache (the miss that burned a run). Digest-pinning is content-addressed."""
import os
import pathlib
import subprocess

LAUNCH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "launch.sh"


def _run(image: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "RUNPOD_API_KEY": "dummy", "HF_TOKEN": "dummy",
        "ASSAY_VOLUME_ID": "vol", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16",
        "ASSAY_WEIGHTS_PATH": "/runpod-volume/model",  # required since F-015
        "ASSAY_IMAGE": image,
    }
    return subprocess.run(["bash", str(LAUNCH), "--dry-run"],
                          capture_output=True, text=True, env=env, timeout=60)


def test_rejects_bare_tag():
    p = _run("ghcr.io/uistlabs/assay:0.5.0")
    assert p.returncode != 0
    assert "digest-pinned" in p.stderr


def test_accepts_digest():
    # Digest passes the guard; --dry-run then builds the payload and exits 0.
    p = _run("ghcr.io/uistlabs/assay@sha256:" + "a" * 64)
    assert p.returncode == 0, p.stderr


import pytest  # noqa: E402


@pytest.mark.parametrize("var,val", [
    ("ASSAY_JOB_CMD", "echo GATE PASSED"),
])
def test_rejects_tests_only_vars_at_launch(var, val):
    # A leftover tests-only override (fake job-command) in the launch
    # shell must fail loud, never ride along to the paid pod. Even with a valid digest.
    env = {
        **os.environ,
        "RUNPOD_API_KEY": "dummy", "HF_TOKEN": "dummy",
        "ASSAY_VOLUME_ID": "vol", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16",
        "ASSAY_WEIGHTS_PATH": "/runpod-volume/model",
        "ASSAY_IMAGE": "ghcr.io/uistlabs/assay@sha256:" + "a" * 64,
        var: val,
    }
    p = subprocess.run(["bash", str(LAUNCH), "--dry-run"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode != 0
    assert var in p.stderr and "TESTS-ONLY" in p.stderr


def test_requires_weights_path_before_spend():
    """F-015 amendment 3, launch-side half: load_config now REQUIRES
    ASSAY_WEIGHTS_PATH, so a launch without it would create a pod that dies at
    boot - paid create + image pull for nothing. The required-vars block must
    catch it on the operator's box at $0."""
    env = {
        **os.environ,
        "RUNPOD_API_KEY": "dummy", "HF_TOKEN": "dummy",
        "ASSAY_VOLUME_ID": "vol", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16",
        "ASSAY_IMAGE": "ghcr.io/uistlabs/assay@sha256:" + "a" * 64,
    }
    env.pop("ASSAY_WEIGHTS_PATH", None)
    p = subprocess.run(["bash", str(LAUNCH), "--dry-run"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode != 0
    assert "ASSAY_WEIGHTS_PATH" in p.stderr


def test_pin_drift_check_is_wired_warn_only():
    """F-015 amendment 4: the launch-side hub cross-check of the recipe's identity
    pins. Contract-grep (same style as the pod_entry marker test): it must exist,
    be non-fatal (drift is information; the hard gate is the in-pod verifier), be
    time-bounded, and stand down when ASSAY_BASE_MODEL deliberately overrides the
    base model (the pins do not describe the overridden model)."""
    text = LAUNCH.read_text()
    assert "pin_base_files.py" in text
    check_line = next(l for l in text.splitlines() if "pin_base_files.py" in l and "python" in l)
    assert "--check" in check_line
    assert "|| true" in check_line or "|| true" in text.split("pin_base_files.py")[1][:200]
    assert "timeout" in check_line
    assert "ASSAY_BASE_MODEL" in text
