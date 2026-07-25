"""pod_entry.sh must never treat a no-op as a successful run.

Regression: job.py once lacked a `__main__` guard, so `python -m assay.job`
imported the module and exited 0 without running anything. pod_entry.sh trusted
that clean exit, the EXIT-trap teardown self-terminated the pod, and a rented GPU
was silently burned with no heartbeat and no error. These tests drive the real
pod_entry.sh with a fake job (ASSAY_JOB_CMD) and fast terminate-retry env overrides,
asserting it fails loudly unless the job printed a GATE completion marker.
"""
import os
import pathlib
import re
import subprocess
import time as _time

from assay.job import GATE_FAILED_MARKER, GATE_PASSED_MARKER

POD_ENTRY = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pod_entry.sh"


def test_gate_marker_contract():
    """The GATE markers job.main() prints and the grep pod_entry.sh matches are a
    two-sided string contract with no shared runtime. Pin them together: extract the
    grep pattern from pod_entry.sh and assert it matches BOTH constants job.py emits.
    A reword of either side (e.g. 'Gate: PASSED', lowercase via the heartbeat) breaks
    this test instead of silently misreporting a multi-hour paid run as a no-op."""
    text = POD_ENTRY.read_text()
    m = re.search(r"grep -q(?:\s+-\w+)*\s+'([^']*)'", text)
    assert m, "could not find the GATE-marker grep line in pod_entry.sh"
    # Translate BRE alternation (\|) to Python's alternation (|) for re.search.
    pattern = m.group(1).replace(r"\|", "|")
    assert re.search(pattern, GATE_PASSED_MARKER), (
        f"pod_entry.sh grep {pattern!r} does not match job.py's {GATE_PASSED_MARKER!r}")
    assert re.search(pattern, GATE_FAILED_MARKER), (
        f"pod_entry.sh grep {pattern!r} does not match job.py's {GATE_FAILED_MARKER!r}")


def _run(job_cmd: str, artifacts_dir: pathlib.Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "ASSAY_JOB_CMD": job_cmd,
        # pod_entry.sh now mkdir -p's and writes stdout.log under ASSAY_ARTIFACTS_DIR
        # unconditionally; point it at a tmp dir so tests never touch the production
        # default (/runpod-volume/assay-out), which doesn't exist off a real pod.
        "ASSAY_ARTIFACTS_DIR": str(artifacts_dir),
        # teardown self_terminate is best-effort (|| echo): with a dummy key it
        # fails fast and is swallowed, so it never affects the asserted exit code.
        # ATTEMPTS=1/BACKOFF=0 keep the trap's self_terminate from sleeping on
        # retries against the dummy key (no ~40s per test).
        "ASSAY_TERMINATE_ATTEMPTS": "1",
        "ASSAY_TERMINATE_BACKOFF": "0",
        "RUNPOD_POD_ID": "test-pod",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)  # unset: keeps publish_artifacts on the fast skip path
    return subprocess.run(
        ["bash", str(POD_ENTRY)], capture_output=True, text=True, env=env, timeout=60
    )


def test_passes_when_job_prints_gate_marker(tmp_path):
    p = _run("echo GATE PASSED", tmp_path)
    assert p.returncode == 0, p.stderr


def test_fails_loud_on_silent_noop(tmp_path):
    # exit 0 with no marker - the exact signature of the __main__-guard bug.
    p = _run("true", tmp_path)
    assert p.returncode == 1, f"expected loud failure, got rc={p.returncode}\n{p.stdout}"
    assert "ran no work" in p.stderr, p.stderr


def test_propagates_job_failure_exit_code(tmp_path):
    p = _run("sh -c 'echo boom >&2; exit 3'", tmp_path)
    assert p.returncode == 3, p.stderr


def test_tees_redacted_log_to_artifacts_dir(tmp_path):
    # The redacted secret here is RUNPOD_API_KEY, not HF_TOKEN: log_tee redacts
    # both (Task 7), so RUNPOD_API_KEY exercises the same _redact() path while
    # leaving HF_TOKEN unset. That takes publish_artifacts's real no-token-skip
    # branch (Task 8) instead of attempting a live HF upload with a fake token -
    # keeping this test fast and deterministic with no network dependency, while
    # still genuinely proving redaction on the volume copy.
    art = tmp_path / "art"
    secret = "RPKEY_SECRET_abc123"
    env = {
        **os.environ,
        "ASSAY_JOB_CMD": f"sh -c 'echo {secret}; echo GATE PASSED'",
        "ASSAY_ARTIFACTS_DIR": str(art),
        "ASSAY_TERMINATE_ATTEMPTS": "1",
        "ASSAY_TERMINATE_BACKOFF": "0",
        "RUNPOD_POD_ID": "pod-123",
        "RUNPOD_API_KEY": secret,
    }
    env.pop("HF_TOKEN", None)  # unset: publish_artifacts skips the upload fast
    p = subprocess.run(
        ["bash", str(POD_ENTRY)], capture_output=True, text=True, env=env, timeout=60
    )
    assert p.returncode == 0, p.stderr
    # per-run subdir: <ASSAY_ARTIFACTS_DIR>/<pod_id>/stdout.log - a reused volume
    # must never bleed one run's log into another's.
    vol_log = (art / "pod-123" / "stdout.log").read_text()
    assert "GATE PASSED" in vol_log
    assert secret not in vol_log  # redacted on the volume copy


def test_uploads_from_trap_even_on_job_failure(tmp_path):
    # The forensics upload must run from the EXIT trap (before self-terminate), so a
    # job that dies still ships its logs. With no ASSAY_ARTIFACTS_DATASET the upload
    # is a clean skip, but the stdout.log must still be written under the per-run dir.
    art = tmp_path / "art"
    env = {
        **os.environ,
        "ASSAY_JOB_CMD": "sh -c 'echo boom >&2; exit 5'",
        "ASSAY_ARTIFACTS_DIR": str(art),
        "ASSAY_TERMINATE_ATTEMPTS": "1",
        "ASSAY_TERMINATE_BACKOFF": "0",
        "RUNPOD_POD_ID": "pod-123",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    p = subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                       env=env, timeout=60)
    assert p.returncode == 5, p.stderr
    # per-run subdir: <ASSAY_ARTIFACTS_DIR>/<pod_id>/stdout.log
    assert (art / "pod-123" / "stdout.log").exists()
    assert "boom" in (art / "pod-123" / "stdout.log").read_text()


def test_terminate_still_fires_when_artifacts_dir_unwritable(tmp_path):
    # self_terminate is the ONLY thing that stops the pod billing now that the
    # wall-clock backstop is gone. If the trap isn't armed until AFTER mkdir/mktemp,
    # a bad volume (read-only/ENOSPC) exits under set -e with NO terminate call ->
    # RunPod restarts the container into the same failure -> crash-loop billing.
    # Force that startup failure: make ASSAY_ARTIFACTS_DIR's parent a FILE, so
    # `mkdir -p` dies with ENOTDIR before any teardown-relevant var is set up.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    env = {
        **os.environ,
        "ASSAY_ARTIFACTS_DIR": str(blocker / "art"),
        "ASSAY_TERMINATE_ATTEMPTS": "1",
        "ASSAY_TERMINATE_BACKOFF": "0",
        "RUNPOD_POD_ID": "pod-unwritable",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    p = subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                       env=env, timeout=60)
    assert p.returncode != 0, f"expected mkdir to fail, got rc=0\n{p.stdout}"
    # Proves the terminate trap fired despite the startup failure - the breadcrumb
    # is the only evidence an operator has that billing may not have stopped.
    assert "self_terminate failed" in p.stderr, (
        f"terminate trap did not fire on startup failure; stderr={p.stderr!r}")


def test_teardown_terminate_is_fast_with_env_override(tmp_path):
    # With ASSAY_TERMINATE_ATTEMPTS=1 / BACKOFF=0 the trap's self_terminate must not
    # sleep on retries even though the dummy key makes the real API call fail.
    env = {
        **os.environ,
        "ASSAY_JOB_CMD": "echo GATE PASSED",
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "ASSAY_TERMINATE_ATTEMPTS": "1",
        "ASSAY_TERMINATE_BACKOFF": "0",
        "RUNPOD_POD_ID": "pod-fast",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    t0 = _time.monotonic()
    p = subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                       env=env, timeout=60)
    assert p.returncode == 0, p.stderr
    assert _time.monotonic() - t0 < 20, "terminate retries were not shortened by env"


def _run_with_stub_python(job_cmd, tmp_path, pod_id="pod-teardown"):
    """Shadow python3.12 with a stub that appends its module name to order.log, so we
    can assert publish_artifacts runs before self_terminate in the trap. The stub
    handles `-m assay.log_tee` (passthrough tee) and `-m assay.publish_artifacts` /
    `-c ...self_terminate` (record + succeed)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    order = tmp_path / "order.log"
    stub = bindir / "python3.12"
    stub.write_text(f"""#!/usr/bin/env bash
# args for the tee call are: -m assay.log_tee <raw_log> <vol_log>  ($3 = raw_log).
# The GATE grep reads the RAW log ($3), so the stub must cat stdin into $3.
if [ "$1" = "-m" ] && [ "$2" = "assay.log_tee" ]; then exec cat > "$3"; fi
if [ "$1" = "-m" ] && [ "$2" = "assay.publish_artifacts" ]; then echo publish >> {order}; exit 0; fi
if [ "$1" = "-c" ]; then echo terminate >> {order}; exit 0; fi
exit 0
""")
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ASSAY_JOB_CMD": job_cmd,
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "RUNPOD_POD_ID": pod_id,
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    p = subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                       env=env, timeout=60)
    return p, (order.read_text().split() if order.exists() else [])


def test_teardown_uploads_then_terminates_on_pass(tmp_path):
    _p, order = _run_with_stub_python("echo GATE PASSED", tmp_path)
    assert order == ["publish", "terminate"], order


def test_teardown_uploads_then_terminates_on_fail(tmp_path):
    _p, order = _run_with_stub_python("sh -c 'echo GATE FAILED; exit 0'", tmp_path)
    # GATE FAILED is a valid completed run (rc 0, marker present); teardown still fires.
    assert order == ["publish", "terminate"], order
