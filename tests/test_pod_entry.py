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


def _run_with_cost_stub(job_cmd, tmp_path, cost_exit=0, pod_id="pod-cost"):
    """Shadow python3.12 with a stub that records call ORDER and lets us force the
    cost calls to fail. Proves cost runs before the upload, and that a hard cost
    failure changes nothing about the run's outcome."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    order = tmp_path / "order.log"
    stub = bindir / "python3.12"
    stub.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "assay.log_tee" ]; then exec cat > "$3"; fi
if [ "$1" = "-m" ] && [ "$2" = "assay.cost" ]; then echo "cost-$3" >> {order}; exit {cost_exit}; fi
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


def test_cost_begins_before_the_job_and_finalizes_before_upload(tmp_path):
    # Ordering is load-bearing: finalize must land BEFORE publish_artifacts so
    # cost.json rides the existing upload, and before _terminate because a
    # terminated pod is no longer queryable.
    p, order = _run_with_cost_stub("echo GATE PASSED", tmp_path)
    assert p.returncode == 0, p.stderr
    assert order == ["cost-begin", "cost-finalize", "publish", "terminate"], order


def test_cost_failure_does_not_change_a_passing_exit_code(tmp_path):
    # The critical safety property. Cost is an observer; a hard failure in it must
    # not change the run's exit code or skip teardown.
    p, order = _run_with_cost_stub("echo GATE PASSED", tmp_path, cost_exit=9)
    assert p.returncode == 0, p.stderr
    assert order == ["cost-begin", "cost-finalize", "publish", "terminate"], order


def test_cost_failure_does_not_mask_a_job_failure(tmp_path):
    # A failing cost call must not swallow or alter the job's real exit code.
    p, order = _run_with_cost_stub("sh -c 'echo boom >&2; exit 5'", tmp_path,
                                   cost_exit=9)
    assert p.returncode == 5, p.stderr
    assert "publish" in order and "terminate" in order, order


def test_cost_finalize_receives_the_job_exit_code(tmp_path):
    # finalize is passed ${rc:-}; on a normal completion rc is set, so the record
    # can distinguish gate_fail from infra_fail.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args_log = tmp_path / "args.log"
    stub = bindir / "python3.12"
    stub.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "assay.log_tee" ]; then exec cat > "$3"; fi
if [ "$1" = "-m" ] && [ "$2" = "assay.cost" ] && [ "$3" = "finalize" ]; then
  echo "$@" >> {args_log}; exit 0; fi
exit 0
""")
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ASSAY_JOB_CMD": "sh -c 'echo GATE FAILED; exit 0'",
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "RUNPOD_POD_ID": "pod-rc",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                   env=env, timeout=60)
    line = args_log.read_text()
    assert "--rc 0" in line, line
    assert "--log" in line, line


def test_cost_finalize_receives_a_nonzero_job_exit_code(tmp_path):
    # Sibling to the test above: a job that fails must round-trip its EXACT rc into
    # finalize, not get clamped to 0 or silently dropped. Without this, a genuine
    # gate_fail/job crash could get cost-recorded as if it were infra_fail (empty)
    # or a clean pass (0) - the same distinction the comment on the test above
    # exists to protect, just on the nonzero side of it.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args_log = tmp_path / "args.log"
    stub = bindir / "python3.12"
    stub.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "assay.log_tee" ]; then exec cat > "$3"; fi
if [ "$1" = "-m" ] && [ "$2" = "assay.cost" ] && [ "$3" = "finalize" ]; then
  echo "$@" >> {args_log}; exit 0; fi
exit 0
""")
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ASSAY_JOB_CMD": "sh -c 'echo boom >&2; exit 5'",
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "RUNPOD_POD_ID": "pod-rc5",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    p = subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                       env=env, timeout=60)
    assert p.returncode == 5, p.stderr
    line = args_log.read_text()
    assert "--rc 5" in line, line
    assert "--log" in line, line


def test_teardown_finalizes_and_terminates_when_rc_is_never_assigned(tmp_path):
    # WHY THIS TEST EXISTS: teardown() is an EXIT trap under `set -euo pipefail`.
    # `rc` is only assigned at `rc=${PIPESTATUS[0]}`, AFTER the job pipeline runs.
    # If the trap fires before that line - a startup failure or a signal, anywhere
    # between `trap teardown EXIT` and that assignment - a bare `$rc` would abort
    # the trap under `set -u` partway through, before ever reaching `_terminate()`,
    # which is the ONE call that stops billing (no wall-clock backstop exists behind
    # it - see the header comment in pod_entry.sh). `${rc:-}` is the guard that
    # keeps `_terminate()` reachable in that exact window. Code review confirmed the
    # guard is statically correct, but nothing exercised the window it protects -
    # this test does, by actually landing there and checking self_terminate still
    # ran, not by pre-seeding `rc` (which would prove nothing about the guard).
    #
    # MECHANISM: stub `-m assay.cost begin` (the first fallible call made AFTER
    # `trap teardown EXIT` is armed but BEFORE `eval "$JOB_CMD" | ...` even starts)
    # sends SIGTERM to pod_entry.sh's own bash process - its grandparent, since this
    # stub's parent is the `timeout` wrapping the call (timeout execve's the target
    # directly, no intermediate shell), and timeout's parent is pod_entry.sh itself.
    # Bash runs EXIT traps on SIGTERM (unlike SIGKILL), so `teardown` fires with
    # `rc` never having been assigned.
    #
    # EMPIRICALLY VERIFIED (not just asserted here, but hand-checked before writing
    # this test): running pod_entry.sh directly with this exact stub, bash exits
    # with rc=143 (SIGTERM) at the shell level - subprocess.run instead reports this
    # as returncode == -15 (Python's convention for signal-terminated children,
    # negative signal number rather than 128+n). The job command's stdout never
    # appears anywhere - it truly never runs, confirming the interrupt lands before
    # `eval`, not after. `--rc` arrives as a genuinely empty argument (verified via
    # one-arg-per-line capture, not a substring match that an empty value would
    # vanish inside of), and both `publish` and `terminate` still fire afterward.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    order = tmp_path / "order.log"
    finalize_args = tmp_path / "finalize_args.log"
    stub = bindir / "python3.12"
    stub.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "assay.log_tee" ]; then exec cat > "$3"; fi
if [ "$1" = "-m" ] && [ "$2" = "assay.cost" ] && [ "$3" = "begin" ]; then
  # $PPID is `timeout`'s pid (it execve's us directly); ITS parent is pod_entry.sh's
  # own bash process - kill that one so the top-level script dies mid-startup,
  # before it can ever reach `rc=${{PIPESTATUS[0]}}`.
  entry_pid=$(awk '{{print $4}}' "/proc/$PPID/stat" 2>/dev/null)
  [ -n "$entry_pid" ] && kill -TERM "$entry_pid"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "assay.cost" ] && [ "$3" = "finalize" ]; then
  printf '%s\\n' "$@" >> {finalize_args}
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "assay.publish_artifacts" ]; then echo publish >> {order}; exit 0; fi
if [ "$1" = "-c" ]; then echo terminate >> {order}; exit 0; fi
exit 0
""")
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        # A distinctive tripwire, not the default job: if the SIGTERM mechanism
        # ever failed to interrupt the script before `eval`, this marker would show
        # up in stdout instead of the test failing silently.
        "ASSAY_JOB_CMD": "echo MECHANISM_FAILED_JOB_RAN",
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "ASSAY_TERMINATE_ATTEMPTS": "1",
        "ASSAY_TERMINATE_BACKOFF": "0",
        "RUNPOD_POD_ID": "pod-rc-unset",
        "RUNPOD_API_KEY": "dummy",
    }
    env.pop("HF_TOKEN", None)
    p = subprocess.run(["bash", str(POD_ENTRY)], capture_output=True, text=True,
                       env=env, timeout=60)

    # Died to the signal (not a normal `exit "$rc"` return) - proof the trap path
    # ran, not the script's ordinary tail.
    assert p.returncode == -15, (
        f"expected the process to die to SIGTERM (-15); got {p.returncode}\n"
        f"stdout={p.stdout!r}\nstderr={p.stderr!r}")
    assert "MECHANISM_FAILED_JOB_RAN" not in p.stdout + p.stderr, (
        "the job command ran - the SIGTERM never interrupted the script before "
        "eval, so this test did not exercise the rc-unset path at all")

    # (1) finalize was still called, and rc was genuinely EMPTY - not "0", not
    # dropped: one argument, and that argument is the empty string, because
    # ${rc:-} substituted for a variable that was never assigned.
    assert finalize_args.exists(), "cost finalize never ran - teardown did not fire"
    argv = finalize_args.read_text().splitlines()
    assert "--rc" in argv, argv
    assert argv[argv.index("--rc") + 1] == "", (
        f"expected an EMPTY --rc value (rc never assigned); argv={argv}")

    # (2) self_terminate still ran - the property that actually matters here.
    order_lines = order.read_text().split() if order.exists() else []
    assert "terminate" in order_lines, (
        f"self_terminate did not run when rc was unset; order={order_lines}")
    # (3) and it ran AFTER finalize + publish, i.e. teardown executed its full,
    # correctly-ordered body rather than something shortcutting straight to it.
    assert order_lines == ["publish", "terminate"], order_lines
