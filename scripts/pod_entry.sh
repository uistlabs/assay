#!/usr/bin/env bash
# Pod entrypoint. Self-terminates on EXIT (success OR failure) after uploading the
# run's forensics. There is NO wall-clock backstop: RunPod credits are the external
# money ceiling (no auto-pay), the in-pod progress-stall watchdog kills a genuinely
# wedged eval, and the external wks1 observer covers a whole-pod freeze. A blind
# clock kill only ever guillotined working runs (the R1 avg@16 burn), so it is gone.
set -euo pipefail

# Injectable ONLY so the loud-failure assert below can be exercised by tests with a
# fake job. Production uses the default versioned interpreter (ubi9 has no bare python).
JOB_CMD="${ASSAY_JOB_CMD:-python3.12 -m assay.job}"

# Per-run artifacts dir: append the pod id so a reused network volume can never bleed
# one run's heartbeat/traceback/stdout into the next run's audit trail. Exported so
# the job (heartbeat, eval JSONs, traceback via config.load_config) writes here too.
#
# - SPLIT-BRAIN WARNING -
# pod_entry does NOT run assay.config.resolve_mount (the job-side helper that rebases
# the job's artifacts/heartbeat paths onto wherever the volume ACTUALLY mounted, e.g.
# /workspace instead of /runpod-volume). If a given pod's volume mounts at /workspace,
# this script still writes to and uploads from a phantom /runpod-volume/... dir while
# the job's real heartbeat/eval-json/traceback go to /workspace - the job's forensics
# silently never upload, with no error anywhere in this trail. Our cert procedure
# launches with /runpod-volume today, so this is safe NOW; it MUST be reconciled
# (either resolve the mount here too, or standardize the mount point) before any
# /workspace-mounted run. See the loud warning right below.
export ASSAY_ARTIFACTS_DIR="${ASSAY_ARTIFACTS_DIR:-/runpod-volume/assay-out/artifacts}/${RUNPOD_POD_ID:-run}"
ARTIFACTS_DIR="$ASSAY_ARTIFACTS_DIR"
if [ ! -d /runpod-volume ] && [ -d /workspace ]; then
    echo "pod_entry: WARNING - volume appears at /workspace, not /runpod-volume; forensics dir may diverge from the job's (see comment); heartbeat/eval JSONs may not upload" >&2
fi

# The ONE call that stops billing. Env-overridable attempts/backoff (tests set 1/0).
# Reads ONLY os.environ, so it is safe to arm before any other shell var below exists.
_terminate() {
    set +x  # never trace the terminate call (env carries the RunPod API key)
    python3.12 -c '
import os
from assay.runpod_ctl import self_terminate
kw = {}
try:
    if os.environ.get("ASSAY_TERMINATE_ATTEMPTS"):
        kw["attempts"] = int(os.environ["ASSAY_TERMINATE_ATTEMPTS"])
    if os.environ.get("ASSAY_TERMINATE_BACKOFF"):
        kw["backoff_s"] = float(os.environ["ASSAY_TERMINATE_BACKOFF"])
except ValueError:
    kw = {}  # a garbage override must never abort the terminate - fall back to safe defaults
self_terminate(os.environ, **kw)
' || echo "teardown: self_terminate failed (pod may already be gone)" >&2
}

# Arm a terminate-only trap IMMEDIATELY, before any fallible command, so a startup
# failure (mkdir/mktemp on a bad volume) still stops billing instead of exiting under
# set -e with nothing armed -> RunPod restarts the container into the same failure ->
# crash-loop billing. Upgraded to the full teardown (upload + terminate) below once the
# artifact dirs exist.
trap _terminate EXIT

# NO org default: an org-scoped default (e.g. uist-labs/...) would make every external
# run of this public image try to write the operator's OWN token into someone else's
# namespace -> a silent 403 that loses their audit trail. Same no-default-namespace rule
# config.py enforces for ASSAY_CHECKPOINT_REPO. Empty => publish_artifacts skips the
# upload cleanly. UIST supplies its own dataset via ASSAY_ARTIFACTS_DATASET at launch.
ARTIFACTS_DATASET="${ASSAY_ARTIFACTS_DATASET:-}"
mkdir -p "$ARTIFACTS_DIR"
run_log="$(mktemp)"                       # ephemeral, local, UNREDACTED (for the grep)
vol_log="$ARTIFACTS_DIR/stdout.log"       # durable, on the volume, REDACTED by log_tee
# Export the raw local log path so the job's in-pod stall watchdog can watch its mtime
# as the primary liveness signal (tqdm writes every iteration through log_tee).
export ASSAY_RAW_LOG="$run_log"

teardown() {
    set +x  # never trace the upload (env carries tokens)
    # Best-effort: fold any lines that overflowed to the LOCAL spill (queue full, or
    # a volume stall that outlived the run) into the artifacts dir BEFORE the upload,
    # so they still ride the off-site copy instead of being stranded on the
    # container's ephemeral fs. `2>/dev/null || true` - a missing spill (the common
    # case, no overflow ever happened) must never fail or slow teardown. Bounded with
    # timeout to prevent a stalled volume from hanging the trap before _terminate.
    timeout -k 5 30 cp "${run_log}.spill" "$ARTIFACTS_DIR/stdout.spill" 2>/dev/null || true
    # Upload forensics BEFORE terminating: after self_terminate the pod is dying and a
    # 300s upload would never finish. The volume copy is the source of truth; a failed
    # or hung HF push must never block or over-spend - hence the timeout + `|| echo`.
    # `-k 10` sends KILL 10s after TERM so a TERM-ignoring child can never block the
    # billing stop below.
    timeout -k 10 300 python3.12 -m assay.publish_artifacts "$ARTIFACTS_DIR" "$ARTIFACTS_DATASET" \
        || echo "teardown: artifact upload failed/timed out (log is on the volume)" >&2
    _terminate
}
trap teardown EXIT

set +e
# Tee the WHOLE process tree (job + eval subprocess + vLLM EngineCore grandchild)
# through the redactor: raw -> ephemeral local, redacted -> volume (async) + spill.
# Explicit local spill path (3rd arg): the failover for a queue-overflow or a
# stalled-volume write must live on the LOCAL fs alongside run_log, never derived
# from vol_log - see log_tee's own default-derivation comment for why a
# volume-path spill would wedge the tee and freeze the watchdog's raw-mtime signal.
eval "$JOB_CMD" 2>&1 | python3.12 -m assay.log_tee "$run_log" "$vol_log" "${run_log}.spill"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -ne 0 ]; then
    echo "pod_entry: assay.job failed (rc=$rc)" >&2
elif ! grep -q 'GATE PASSED\|GATE FAILED' "$run_log"; then
    # Only reachable when rc==0: a nonzero rc already carries a real, specific
    # failure code (test_propagates_job_failure_exit_code) and must not be
    # clobbered by the generic no-marker sentinel below.
    echo "pod_entry: FATAL - assay.job exited with no GATE marker; it ran no work" >&2
    rc=1
fi

exit "$rc"
