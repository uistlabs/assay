#!/usr/bin/env bash
# Pod entrypoint. Self-terminates on EXIT (success OR failure) and enforces a
# wall-clock backstop so an unattended, money-burning pod can never run away.
set -euo pipefail

BACKSTOP_SECONDS="${ASSAY_BACKSTOP_SECONDS:-21600}"  # 6h; set 0 to disable (tests)
# The job command is injectable ONLY so the loud-failure assert below can be
# exercised by tests with a fake job. Production uses the default: the batch
# image installs assay into the system python3.12 (ubi9 has no unversioned
# `python` on PATH), so the versioned interpreter is used explicitly.
JOB_CMD="${ASSAY_JOB_CMD:-python3.12 -m assay.job}"

teardown() {
    set +x  # never trace the terminate (would print the api key)
    python3.12 -c 'import os; from assay.runpod_ctl import self_terminate; self_terminate(os.environ)' \
        || echo "teardown: self_terminate failed (pod may already be gone)"
}
trap teardown EXIT

# Wall-clock backstop: kill our own process group after the ceiling.
if [ "$BACKSTOP_SECONDS" -gt 0 ]; then
    ( sleep "$BACKSTOP_SECONDS"; echo "backstop reached -- forcing teardown"; kill -TERM 0 ) &
fi

# Run the job and ASSERT it actually did work. run_job's main() prints a GATE
# marker as its final act; a silent exit-0 with NO marker means the entrypoint
# ran nothing -- exactly the failure a missing `__main__` guard once produced:
# the pod pulled the image, no-op'd, and self-terminated with no error, silently
# burning a rented GPU. Capture stdout to an EPHEMERAL local file (raw stdout is
# unredacted and must never be persisted to the network volume) and fail loudly
# if the marker is absent, so a no-op can never masquerade as a successful run.
# The teardown trap still self-terminates the pod on every exit path below.
run_log="$(mktemp)"
set +e
# JOB_CMD is trusted: it is the built-in default (python3.12 -m assay.job) unless a
# test explicitly overrides ASSAY_JOB_CMD. It never carries external/operator input.
eval "$JOB_CMD" 2>&1 | tee "$run_log"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -ne 0 ]; then
    echo "pod_entry: assay.job failed (rc=$rc)" >&2
    exit "$rc"
fi
if ! grep -q 'GATE PASSED\|GATE FAILED' "$run_log"; then
    echo "pod_entry: FATAL -- assay.job exited 0 with no GATE marker; it ran no work" >&2
    echo "pod_entry: (entrypoint no-op or silent early exit) -- treating as failure, not success" >&2
    exit 1
fi
