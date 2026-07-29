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
# so RunPod pulls the image anonymously. Keep ghcr.io/uist-labs/assay public. To
# pull a private image, switch this create call to the REST POST /v1/pods endpoint
# and pass containerRegistryAuthId (a deliberate follow-on, not wired here).
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY (dedicated, pod-scoped, rotatable)}"
: "${HF_TOKEN:?export HF_TOKEN (fine-grained, write-scoped to the target repo)}"
: "${ASSAY_VOLUME_ID:?export ASSAY_VOLUME_ID (your pre-staged weights network volume id)}"
: "${ASSAY_IMAGE:?export ASSAY_IMAGE (ghcr.io/uist-labs/assay:TAG, PUBLIC image)}"
: "${ASSAY_CHECKPOINT_REPO:?export ASSAY_CHECKPOINT_REPO (your target HF repo id, e.g. yourorg/Model-NVFP4A16)}"
# Required since F-015: load_config has no weights-path default any more (the old
# default was a Qwen path for EVERY recipe). Catch the omission HERE, at $0, not as
# a paid pod-create that dies at boot.
: "${ASSAY_WEIGHTS_PATH:?export ASSAY_WEIGHTS_PATH (staged base-model dir on the volume, e.g. /runpod-volume/qwen2.5-7b-instruct)}"

# Digest-pin guard: a mutable tag can be served stale from RunPod's image cache - the
# exact miss that burned a run. Require a content-addressed digest.
case "$ASSAY_IMAGE" in
  *@sha256:*) : ;;  # digest-pinned, good
  *)
    echo "FATAL: ASSAY_IMAGE must be digest-pinned (ghcr.io/uist-labs/assay@sha256:...)." >&2
    echo "A mutable tag can be served stale from RunPod's image cache. Get the digest" >&2
    echo "from the push output, or:" >&2
    echo "  podman inspect --format '{{.Digest}}' ghcr.io/uist-labs/assay:<tag>" >&2
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
# Also runs derive_max_model_len here so a junk ASSAY_MAX_MODEL_LEN (non-integer,
# non-positive, or <= the recipe's max_gen_toks) raises on the operator's box at $0,
# instead of surfacing only after a paid pod create + image pull.
python -c '
import os
from assay.recipes import get_recipe
from assay.config import _apply_recipe_overrides, derive_max_model_len
from assay.preflight import launch_reminder
r = _apply_recipe_overrides(get_recipe(os.environ.get("ASSAY_RECIPE", "qwen2_5_7b_instruct")), os.environ)
derive_max_model_len(r, os.environ)
print(launch_reminder(r))
' >&2

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="1"

# F-040 reader hook: default-on passive observability. ONE warn-guarded unit
# (R-14) - it runs AFTER the paid pod exists under `set -e`, so no failure in
# here may kill the launcher. Preconditions checked at $0 (R-5): a reader with
# no destination must never boot and bill.
reader_hook() {  # $1 = main pod id, $2 = "1" for dry-run
  if [ "${ASSAY_READER:-1}" != "1" ]; then
    echo "[reader] reader disabled (ASSAY_READER=${ASSAY_READER:-})" >&2
    return 0
  fi
  if [ -z "${ASSAY_ARTIFACTS_DATASET:-}" ]; then
    echo "[reader] reader disabled: no ASSAY_ARTIFACTS_DATASET - mid-run" >&2
    echo "[reader] observability is OFF for this burn (export it to enable)." >&2
    return 0
  fi
  if [ -n "$2" ]; then
    bash "$(dirname "$0")/reader_pod.sh" --print-payload "$1"
  else
    bash "$(dirname "$0")/reader_pod.sh" "$1"
  fi
}

# Cost pre-flight: what this run costs per hour, BEFORE the spend. Read-only catalog
# query, bounded + non-fatal - a pricing lookup must never delay or block a launch.
# Skipped on --dry-run: a dry run's whole point is "show me the payload without
# touching the network", and this is a live RunPod API call.
#
# ASSAY_FLEET_EST_HOURS rides here too (D8/F-009 T9): balance_warning_line reads it
# straight off os.environ (no new plumbing needed - it is already ambient in the
# operator's exported shell env, same as every other ASSAY_* knob these functions
# read directly). ASSAY_FLEET_EXPECTED drives the separate pod-subset check below -
# same bounded/non-fatal block, per Ken's ruling that pod listing is pod-control
# domain (runpod_ctl.py), not preflight.py (deliberately kept NO-NETWORK).
if [ -z "$DRY" ]; then
  timeout -k 5 30 python -c '
import os
from assay.cost.collect import preflight_line, balance_warning_line
from assay.runpod_ctl import (list_assay_pods, parse_fleet_expected,
                              fleet_stray_warning_lines)
print(preflight_line(os.environ))
_w = balance_warning_line(os.environ)
if _w:
    print(_w)
_expected = parse_fleet_expected(os.environ.get("ASSAY_FLEET_EXPECTED"))
if _expected is not None:
    _pods = list_assay_pods(os.environ)
    if _pods is not None:
        for _line in fleet_stray_warning_lines(_expected, _pods):
            print(_line)
' >&2 || true
fi

# Warn-only pin drift check (F-015 amendment 4): does the hub still match the
# recipe's pinned base-model identity? Drift is INFORMATION (upstream moved since
# pinning - re-review before re-pinning), never a block: the hard gate is the
# in-pod verifier against the git pins, which needs no network. Bounded and
# non-fatal like the cost pre-flight. Skipped on --dry-run (network call) and
# under ASSAY_BASE_MODEL (the pins deliberately do not describe the overridden
# model; the in-pod verifier stands down for the same reason).
if [ -z "$DRY" ] && [ -z "${ASSAY_BASE_MODEL:-}" ]; then
  timeout -k 5 30 python "$(dirname "$0")/pin_base_files.py" "${ASSAY_RECIPE:-qwen2_5_7b_instruct}" --check >&2 || true
fi

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
    reader_hook "DRY-RUN" "1" || echo "WARN: reader payload build failed" >&2
    exit 0
fi

set +x  # never trace the create call (would print the api key)
MAIN_POD_ID="$(python -c '
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
print(pod["id"])
')"
echo "created pod: $MAIN_POD_ID"

# Warn-only (R-14): a reader problem must never abort a paid burn.
reader_hook "$MAIN_POD_ID" "" || {
  echo "WARN: reader launch failed - burn continues WITHOUT mid-run" >&2
  echo "WARN: observability. RunPod console log tab is the fallback." >&2
}
