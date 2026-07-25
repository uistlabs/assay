#!/usr/bin/env bash
# Local launcher. Secrets come from the operator's environment ONLY (never a file).
# xtrace must be OFF before anything else runs: if an operator debugs with
# `bash -x`, the presence-check guards below expand RUNPOD_API_KEY/HF_TOKEN
# to their real values on the trace. Neutralize xtrace first, before set -e,
# so secrets never reach the trace/log even under `bash -x scripts/launch.sh`.
set +x
set -euo pipefail

# ASSAY_IMAGE must be a PUBLIC registry image. This launcher creates the pod via
# the runpod SDK create_pod (GraphQL) path, which takes no registry-auth argument,
# so RunPod pulls the image anonymously. Keep ghcr.io/uistlabs/assay public. To
# pull a private image, switch this create call to the REST POST /v1/pods endpoint
# and pass containerRegistryAuthId (a deliberate follow-on, not wired here).
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY (dedicated, pod-scoped, rotatable)}"
: "${HF_TOKEN:?export HF_TOKEN (fine-grained, write-scoped to the target repo)}"
: "${ASSAY_VOLUME_ID:?export ASSAY_VOLUME_ID (your pre-staged weights network volume id)}"
: "${ASSAY_IMAGE:?export ASSAY_IMAGE (ghcr.io/uistlabs/assay:TAG, PUBLIC image)}"
: "${ASSAY_CHECKPOINT_REPO:?export ASSAY_CHECKPOINT_REPO (your target HF repo id, e.g. yourorg/Model-NVFP4A16)}"

# Digest-pin guard: a mutable tag can be served stale from RunPod's image cache - the
# exact miss that burned a run. Require a content-addressed digest.
case "$ASSAY_IMAGE" in
  *@sha256:*) : ;;  # digest-pinned, good
  *)
    echo "FATAL: ASSAY_IMAGE must be digest-pinned (ghcr.io/uistlabs/assay@sha256:...)." >&2
    echo "A mutable tag can be served stale from RunPod's image cache. Get the digest" >&2
    echo "from the push output, or:" >&2
    echo "  podman inspect --format '{{.Digest}}' ghcr.io/uistlabs/assay:<tag>" >&2
    exit 1
    ;;
esac

# Tests-only var guard: ASSAY_JOB_CMD exists ONLY so the loud-failure assert in
# pod_entry.sh can be exercised by tests with a fake job. A leftover in a real launch
# env would run something other than the real job - reject it up front.
for _v in ASSAY_JOB_CMD; do
  if [ -n "${!_v:-}" ]; then
    echo "FATAL: $_v is set in the launch environment. It is a TESTS-ONLY override" >&2
    echo "(fake job-command override) - must never reach a paid pod." >&2
    echo "Unset it before launching:  unset $_v" >&2
    exit 1
  fi
done

# No-fetch base-card reminder (built from the selected recipe; no network, no secrets).
python -c '
import os
from assay.recipes import get_recipe
from assay.config import _apply_recipe_overrides
from assay.preflight import launch_reminder
r = _apply_recipe_overrides(get_recipe(os.environ.get("ASSAY_RECIPE", "qwen2_5_7b_instruct")), os.environ)
print(launch_reminder(r))
' >&2

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="1"

# Build the payload via the tested pure function; redact secret values for display.
# extra_env (I3): any ASSAY_* var in the operator's shell rides along as
# non-secret pod config - e.g. ASSAY_NUM_CALIB=8 for a cheap smoke run -
# without needing an image rebuild. These are config, not secrets, so they are
# shown unredacted in the dry-run output; only the two real secrets get "***".
PAYLOAD="$(python -c '
import json, os
from assay.runpod_ctl import build_pod_payload
extra = {k: v for k, v in os.environ.items() if k.startswith("ASSAY_")}
p = build_pod_payload(image=os.environ["ASSAY_IMAGE"],
                      volume_id=os.environ["ASSAY_VOLUME_ID"],
                      env_keys=["HF_TOKEN", "RUNPOD_API_KEY"], env=os.environ,
                      extra_env=extra)
for e in p["env"]:
    if e["key"] in ("HF_TOKEN", "RUNPOD_API_KEY"):
        e["value"] = "***"
print(json.dumps(p, indent=2))
')"

if [ -n "$DRY" ]; then
    echo "[dry-run] pod payload (secrets redacted):"
    echo "$PAYLOAD"
    exit 0
fi

set +x  # never trace the create call (would print the api key)
python -c '
import os, runpod
from assay.runpod_ctl import build_pod_payload
runpod.api_key = os.environ["RUNPOD_API_KEY"]
extra = {k: v for k, v in os.environ.items() if k.startswith("ASSAY_")}
p = build_pod_payload(image=os.environ["ASSAY_IMAGE"],
                      volume_id=os.environ["ASSAY_VOLUME_ID"],
                      env_keys=["HF_TOKEN", "RUNPOD_API_KEY"], env=os.environ,
                      extra_env=extra)
pod = runpod.create_pod(**{
    "name": "assay-nvfp4",
    "image_name": p["imageName"],
    "gpu_type_id": p["gpuTypeId"],
    "data_center_id": p["dataCenterId"],
    "network_volume_id": p["networkVolumeId"],
    "gpu_count": 1,
    "container_disk_in_gb": p["containerDiskInGb"],
    "min_memory_in_gb": p["minMemoryInGb"],
    "min_vcpu_count": p["minVcpuCount"],
    "cloud_type": p["cloudType"],
    "allowed_cuda_versions": p["allowedCudaVersions"],
    "min_download": p["minDownload"],
    "env": {e["key"]: e["value"] for e in p["env"]},
})
print("created pod:", pod["id"])
'
