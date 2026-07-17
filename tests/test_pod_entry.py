"""pod_entry.sh must never treat a no-op as a successful run.

Regression: job.py once lacked a `__main__` guard, so `python -m assay.job`
imported the module and exited 0 without running anything. pod_entry.sh trusted
that clean exit, the EXIT-trap teardown self-terminated the pod, and a rented GPU
was silently burned with no heartbeat and no error. These tests drive the real
pod_entry.sh with a fake job (ASSAY_JOB_CMD) and the backstop disabled, asserting
it fails loudly unless the job printed a GATE completion marker.
"""
import os
import pathlib
import subprocess

POD_ENTRY = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pod_entry.sh"


def _run(job_cmd: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "ASSAY_JOB_CMD": job_cmd,
        "ASSAY_BACKSTOP_SECONDS": "0",  # no lingering backstop subshell in tests
        # teardown self_terminate is best-effort (|| echo): with a dummy key it
        # fails fast and is swallowed, so it never affects the asserted exit code.
        "RUNPOD_POD_ID": "test-pod",
        "RUNPOD_API_KEY": "dummy",
    }
    return subprocess.run(
        ["bash", str(POD_ENTRY)], capture_output=True, text=True, env=env, timeout=60
    )


def test_passes_when_job_prints_gate_marker():
    p = _run("echo GATE PASSED")
    assert p.returncode == 0, p.stderr


def test_fails_loud_on_silent_noop():
    # exit 0 with no marker -- the exact signature of the __main__-guard bug.
    p = _run("true")
    assert p.returncode == 1, f"expected loud failure, got rc={p.returncode}\n{p.stdout}"
    assert "ran no work" in p.stderr, p.stderr


def test_propagates_job_failure_exit_code():
    p = _run("sh -c 'echo boom >&2; exit 3'")
    assert p.returncode == 3, p.stderr
