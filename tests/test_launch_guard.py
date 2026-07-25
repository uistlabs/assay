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
        "ASSAY_IMAGE": "ghcr.io/uistlabs/assay@sha256:" + "a" * 64,
        var: val,
    }
    p = subprocess.run(["bash", str(LAUNCH), "--dry-run"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode != 0
    assert var in p.stderr and "TESTS-ONLY" in p.stderr
