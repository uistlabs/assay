#!/usr/bin/env bash
# F-040 reader-pod launcher. Same secrets posture as launch.sh: xtrace off FIRST,
# secrets from the operator environment only. Usage:
#   reader_pod.sh <main_pod_id>                  create the reader via REST
#   reader_pod.sh --print-payload <main_pod_id>  print redacted payload, no network
set +x
set -euo pipefail

PRINT=""
if [ "${1:-}" = "--print-payload" ]; then PRINT="1"; shift; fi
MAIN_POD_ID="${1:?usage: reader_pod.sh [--print-payload] <main_pod_id>}"

: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY (per-session, pod-scoped)}"
: "${HF_TOKEN:?export HF_TOKEN (write-scoped to the artifacts dataset)}"
: "${ASSAY_VOLUME_ID:?export ASSAY_VOLUME_ID}"
: "${ASSAY_ARTIFACTS_DATASET:?export ASSAY_ARTIFACTS_DATASET (reader has no org default)}"

HERE="$(cd "$(dirname "$0")" && pwd)"
export ASSAY_READER_HERE="$HERE" ASSAY_READER_MAIN="$MAIN_POD_ID" \
       ASSAY_READER_PRINT="$PRINT"

python - <<'PY'
import base64, json, os, sys, urllib.request
from assay.runpod_ctl import build_reader_payload

here = os.environ["ASSAY_READER_HERE"]


def b64(name):
    with open(os.path.join(here, name), "rb") as fh:
        return base64.b64encode(fh.read()).decode()


payload = build_reader_payload(
    main_pod_id=os.environ["ASSAY_READER_MAIN"],
    volume_id=os.environ["ASSAY_VOLUME_ID"],
    dataset=os.environ["ASSAY_ARTIFACTS_DATASET"],
    loop_b64=b64("reader_loop.sh"), snapshot_b64=b64("reader_snapshot.py"),
    env=os.environ)

if os.environ["ASSAY_READER_PRINT"]:
    shown = dict(payload)
    shown["env"] = {k: ("***" if k in ("RUNPOD_API_KEY", "HF_TOKEN") else v)
                    for k, v in payload["env"].items()}
    print("reader payload (secrets redacted):")
    print(json.dumps(shown, indent=2))
    sys.exit(0)

req = urllib.request.Request(
    "https://rest.runpod.io/v1/pods",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
             "Content-Type": "application/json", "User-Agent": "curl/8.0"},
    method="POST")
with urllib.request.urlopen(req, timeout=60) as resp:
    pod = json.load(resp)
print("created reader pod:", pod["id"])
PY
