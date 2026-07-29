#!/usr/bin/env bash
# scripts/reader_loop.sh - F-040 in-pod reader loop. Runs on python:3.12-slim via
# the dockerStartCmd bootstrap (see build_reader_payload). Deliberately NOT `set -e`:
# a transient cycle failure must fall through to the sleep and retry - an exiting
# container gets RESTARTED by RunPod into a billing crash loop (pod_entry.sh:52-56
# lore), which is exactly what the TTL exists to prevent. Cycles are stateless, so
# a restart mid-loop is harmless (R-11); the TTL anchors on the pod's createdAt via
# REST inside reader_snapshot.py, so restarts cannot reset it (R-4).
#
# The reader NEVER writes to /runpod-volume (spec discipline; contract-tested).
set -u
set +x  # never trace (env carries RUNPOD_API_KEY / HF_TOKEN)

: "${RUNPOD_POD_ID:?injected by RunPod}"
: "${RUNPOD_API_KEY:?}"
: "${HF_TOKEN:?}"
: "${ASSAY_READER_MAIN_POD_ID:?}"
: "${ASSAY_ARTIFACTS_DATASET:?}"

# Pinned install (R-13), retried: a registry blip at boot must not crash-loop us.
until pip install --quiet "huggingface_hub==1.23.0"; do
  echo "reader: pip install failed; retrying in 30s" >&2
  sleep 30
done

echo "reader: up for main pod ${ASSAY_READER_MAIN_POD_ID}," \
     "interval ${ASSAY_READER_INTERVAL:-600}s"

while true; do
  python3 /tmp/reader/reader_snapshot.py
  rc=$?
  # Finalize ONLY on 10 (main pod gone/terminal), 11 (reader TTL expired), or 127
  # (python3 launch itself broken - a boot-broken reader stopping its own billing
  # is correct). Everything else, including a transient SIGKILL/OOM cycle (rc 137),
  # falls through to the sleep and retries next cycle instead of self-deleting on
  # a signal death (whole-branch review finding).
  if [ "$rc" -eq 10 ] || [ "$rc" -eq 11 ] || [ "$rc" -eq 127 ]; then
    echo "reader: finalized (rc=$rc); self-terminating"
    break
  fi
  sleep "${ASSAY_READER_INTERVAL:-600}"
done

# THE ONE DELETE CALL SITE. Own pod id ONLY - the same key could kill the main pod
# (R-12), so the target is hardcoded to the RunPod-injected $RUNPOD_POD_ID, asserted
# non-empty by the guard block above. python3 + urllib because the pinned
# python:3.12-slim image has no curl/wget (python3 + pip only - whole-branch review
# finding). Retry forever: this is the call that stops billing, and exiting instead
# would restart the container and bill on (R-4). REST DELETE is primary; on ANY
# failure the same attempt falls back to GraphQL podTerminate (F-042: the 07-28
# drill hit an 18-min window where in-pod REST 403'd a valid key while GraphQL
# worked - and F-044 had the TTL rescue inert, i.e. unbounded billing). A GraphQL
# "not found" error means already terminated: success. The heredoc must never
# contain the literal word e-x-i-t (contract test): failures raise, python
# returns nonzero, the until-loop retries.
until python3 - <<'PY'
import json, os, urllib.request
pid = os.environ["RUNPOD_POD_ID"]
key = os.environ["RUNPOD_API_KEY"]
try:
    req = urllib.request.Request(
        "https://rest.runpod.io/v1/pods/" + pid,
        headers={"Authorization": "Bearer " + key, "User-Agent": "curl/8.0"},
        method="DELETE")
    urllib.request.urlopen(req, timeout=30)
except Exception:
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query":
            'mutation { podTerminate(input: {podId: "%s"}) }' % pid}).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json", "User-Agent": "curl/8.0"},
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    errs = payload.get("errors") or []
    if errs and not any("not found" in (e.get("message") or "").lower()
                        for e in errs):
        raise RuntimeError("podTerminate failed: %s" % errs)
PY
do
  echo "reader: self-delete failed; retrying in 30s" >&2
  sleep 30
done
