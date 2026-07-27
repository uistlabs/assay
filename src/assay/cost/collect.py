"""Impure edge of cost telemetry: the RunPod probe, atomic record I/O, and the
begin/finalize entry points scripts/pod_entry.sh calls.

Everything that touches the network or the filesystem lives here so model.py can
stay pure. NOTHING in this module may raise into pod_entry.sh's control flow - a
cost failure must never change a run's exit code, abort a run, or skip teardown.
Probe and I/O functions therefore return None/False on failure rather than raising.
ASCII only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from assay import __version__ as _assay_version
from assay.cost import model as _model
from assay.cost import rates as _rates

# Permitted direction of the one-way dependency: cost may import core. The GATE
# markers are already single-sourced in job.py with a two-sided contract test
# (tests/test_pod_entry.py::test_gate_marker_contract); re-deriving the literals
# here would create exactly the string drift that test exists to prevent.
from assay.job import GATE_FAILED_MARKER, GATE_PASSED_MARKER

# Same permitted direction: cost may import core, and this module has no SDK
# import (nothing in runpod_ctl.py touches the network at import time), so this is
# a plain constant import, not the lazy-SDK pattern used in _authenticated_runpod.
# Single-sourcing here means a SKU change in runpod_ctl.py cannot silently leave a
# stale duplicate pricing a different GPU in the pre-flight line or the in-pod
# catalog fallback.
from assay.runpod_ctl import CLOUD_TYPE as _DEFAULT_CLOUD_TYPE
from assay.runpod_ctl import REGION as _DEFAULT_REGION
from assay.runpod_ctl import RTX_5090 as _DEFAULT_GPU_TYPE

RECORD_NAME = "cost.json"


def record_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, RECORD_NAME)


def _authenticated_runpod():  # pragma: no cover - real SDK only on a live pod
    """The runpod module with api_key set, or None if no key is present.

    Imported lazily so `python -m assay.cost` stays light and unit tests never
    touch the SDK.
    """
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        return None
    import runpod
    runpod.api_key = key
    return runpod


def probe_pod(pod_id: str, api=None) -> dict | None:
    """One read-only get_pod. `api` is injectable, matching runpod_ctl.py's seam.

    Returns None when the pod is not queryable - which is the NORMAL state for a
    TERMINATED pod (verified 2026-07-25 against five real terminated assay pod ids:
    RunPod returns None for every one). That is precisely why cost must be captured
    in-run: there is no post-hoc recovery path.

    Never raises. An API error degrades to None so a cost probe cannot take down a
    paid run.
    """
    if not pod_id:
        return None
    if api is None:  # pragma: no cover - real SDK only on a live pod
        api = _authenticated_runpod()
        if api is None:
            return None
    try:
        return api.get_pod(pod_id)
    except Exception:
        return None


def probe_gpu_price(gpu_type_id: str, cloud_type: str = "SECURE",
                    api=None) -> float | None:
    """Catalog $/hr for a GPU type - the fallback when the pod probe fails.

    PRICE TRAP: pick securePrice for SECURE cloud. RTX 5090 reports securePrice
    $0.99 but communityPrice and lowestPrice.uninterruptablePrice are both $0.69,
    and assay pins CLOUD_TYPE=SECURE (runpod_ctl.py) because network volumes exist
    only in secure-cloud datacenters. Reading the lowest price would undercount
    every quote by 30%.

    Never raises; returns None on any failure.
    """
    if api is None:  # pragma: no cover - real SDK only on a live pod
        api = _authenticated_runpod()
        if api is None:
            return None
    try:
        gpu = api.get_gpu(gpu_type_id, gpu_quantity=1)
    except Exception:
        return None
    if not gpu:
        return None
    field = "securePrice" if str(cloud_type).upper() == "SECURE" else "communityPrice"
    price = gpu.get(field)
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def _tmp_record_path(artifacts_dir: str) -> str:
    """The scratch path write_record stages into before the atomic rename.

    Leading dot on the basename (.cost.json.tmp, not cost.json.tmp): if a SIGKILL
    lands mid-write, the torn file survives on the volume, and publish_artifacts
    uploads the whole artifacts_dir with no ignore patterns. A leading dot makes it
    visibly a scratch file rather than a second candidate "real" record sitting next
    to cost.json - the same "which file is real?" confusion the atomic-write design
    already exists to prevent, one layer up in the published dataset.
    """
    final_dir, final_name = os.path.split(record_path(artifacts_dir))
    return os.path.join(final_dir, "." + final_name + ".tmp")


def write_record(artifacts_dir: str, record: dict) -> bool:
    """Atomically write the cost record. Returns True on success, never raises.

    tmp + os.replace so a mid-write pod death cannot leave a TORN record on the
    volume - the volume outlives the pod, and a half-written record there would be
    worse than none. ASCII-only, matching the rest of the repo's artifacts.
    """
    try:
        os.makedirs(artifacts_dir, exist_ok=True)
        final = record_path(artifacts_dir)
        tmp = _tmp_record_path(artifacts_dir)
        with open(tmp, "w", encoding="ascii", errors="replace") as fh:
            json.dump(record, fh, indent=2, sort_keys=True, ensure_ascii=True)
            fh.write("\n")
        os.replace(tmp, final)
        return True
    except Exception as exc:
        print(f"assay.cost: record write failed: {exc}", file=sys.stderr)
        # Best-effort cleanup of the temp file. The volume outlives the pod and is
        # read by humans at 2 AM - a stray temp file next to a stale/absent
        # cost.json creates a "which file is real?" confusion that the atomic-write
        # design exists to prevent. If cleanup itself fails, swallow the error and
        # return False without masking the original failure message already printed.
        try:
            os.remove(_tmp_record_path(artifacts_dir))
        except OSError:
            pass
        return False


def read_record(artifacts_dir: str) -> dict | None:
    """Read back a previously written record, or None if absent/corrupt."""
    try:
        with open(record_path(artifacts_dir), encoding="ascii",
                  errors="replace") as fh:
            return json.load(fh)
    except Exception:
        return None


def scan_gate_result(log_path: str | None) -> str | None:
    """Normalize the run log's GATE marker to "pass" | "fail" | None.

    Uses whichever marker occurs LAST in the log, not first-match-wins: job.py
    prints its marker as the FINAL act of the run, so the last occurrence is the
    real outcome even if a stray earlier "GATE PASSED"/"GATE FAILED" string shows
    up in output before that (e.g. echoed from a sub-step, a retry, or test
    fixture noise). gate_fail is billable audit work and pass means a certified
    deliverable, so misreading one as the other is commercially load-bearing, not
    cosmetic.

    SECURITY: this reads the RAW, UNREDACTED run log (the same file pod_entry.sh
    greps). It must return ONLY the normalized token and never any log content -
    returning log text would put a secret into cost.json.
    """
    if not log_path:
        return None
    try:
        with open(log_path, encoding="ascii", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    last_passed = text.rfind(GATE_PASSED_MARKER)
    last_failed = text.rfind(GATE_FAILED_MARKER)
    if last_passed == -1 and last_failed == -1:
        return None
    return "pass" if last_passed > last_failed else "fail"


def effective_uptime_seconds(pod: dict | None, basis: dict | None,
                             now: float | None = None) -> float:
    """Billed seconds for this pod: provider truth when available, else local accrual.

    Provider truth wins because RunPod's uptimeSeconds is what it actually bills.
    When the pod can no longer be probed, accrue from the basis captured at begin:
    `uptime_seconds_at_begin + (now - began_at_unix)`. The basis offset is billed
    IMAGE-PULL time that the job never sees (uptime starts at pod creation, and a
    pull has taken 60-90 min on slow hosts), so dropping it undercounts the run.

    Clamped so backwards clock skew can never produce a negative accrual.
    """
    if pod and pod.get("uptimeSeconds") is not None:
        try:
            return max(0.0, float(pod["uptimeSeconds"]))
        except (TypeError, ValueError):
            pass
    if not basis:
        return 0.0
    try:
        at_begin = max(0.0, float(basis.get("uptime_seconds_at_begin") or 0))
    except (TypeError, ValueError):
        at_begin = 0.0
    began = basis.get("began_at_unix")
    if began is None:
        return at_begin
    stamp = time.time() if now is None else now
    try:
        elapsed = max(0.0, float(stamp) - float(began))
    except (TypeError, ValueError):
        elapsed = 0.0
    return at_begin + elapsed


SCHEMA_VERSION = 1


def _base_model(env) -> str:
    """The model that was quantized - a predictor join key, and the honest value
    when ASSAY_BASE_MODEL overrode the recipe. Degrades to "" rather than raising."""
    if env.get("ASSAY_BASE_MODEL"):
        return env["ASSAY_BASE_MODEL"]
    try:
        from assay.recipes import get_recipe
        return get_recipe(env.get("ASSAY_RECIPE", "qwen2_5_7b_instruct")).base_model
    except Exception:
        return ""


def build_record(*, env, pod, basis, gate_result, finalized, rc=None,
                 catalog_price=None, now=None) -> dict:
    """Assemble the cost record.

    Self-contained by design: it carries the rates it was computed with, so a
    recomputation years from now is deterministic after RunPod changes prices. That
    is the difference between a number and an AUDITABLE number.

    Pure with respect to its arguments (no module state), because the host-side
    reconciler re-runs this same function over stored records. `rc` is an explicit
    parameter for exactly that reason.

    Only non-secret values are read from `env` - never RUNPOD_API_KEY or HF_TOKEN.
    """
    pod = pod or {}
    stamp = time.time() if now is None else float(now)

    cost_per_hr = pod.get("costPerHr")
    if cost_per_hr is not None:
        rate_source = "provider"
    elif basis and basis.get("rate_source") in ("provider", "catalog"):
        # Carry forward the rate captured at begin: costPerHr is FIXED for the life
        # of a pod, so the begin value stays valid even once the pod is unqueryable.
        cost_per_hr = (basis or {}).get("cost_per_hr")
        rate_source = basis.get("rate_source")
    elif catalog_price is not None:
        cost_per_hr, rate_source = catalog_price, "catalog"
    else:
        cost_per_hr, rate_source = None, "unknown"

    uptime = effective_uptime_seconds(pod or None, basis, now=stamp)
    container_disk = pod.get("containerDiskInGb")
    if container_disk is None and basis:
        container_disk = basis.get("container_disk_gb")
    volume_disk = pod.get("volumeInGb")
    if volume_disk is None and basis:
        volume_disk = basis.get("volume_disk_gb")

    # GPU identity: a live pod probe wins (it is provider truth). Once the pod is
    # unqueryable, prefer the basis captured at begin over the env/default fallback -
    # otherwise a finalize on a terminated pod silently rewrites the run's actual
    # hardware (e.g. an H200, gpuCount 2) to the env/default guess, which every
    # cost-record consumer (predictor, reconciler) treats as a join key.
    gpu_display_name = (pod.get("machine") or {}).get("gpuDisplayName")
    if gpu_display_name is None and basis:
        gpu_display_name = basis.get("gpu_display_name")
    if gpu_display_name is None:
        gpu_display_name = env.get("ASSAY_GPU_TYPE", _DEFAULT_GPU_TYPE)

    gpu_count = pod.get("gpuCount")
    if gpu_count is None and basis:
        gpu_count = basis.get("gpu_count")
    if gpu_count is None:
        gpu_count = 1

    breakdown = _model.marginal_usd(
        cost_per_hr=cost_per_hr, uptime_seconds=uptime,
        container_disk_gb=container_disk, volume_disk_gb=volume_disk)

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "pod_id": env.get("RUNPOD_POD_ID", ""),
            "assay_version": _assay_version,
            "build_sha": env.get("ASSAY_BUILD_SHA", ""),
            "recipe": env.get("ASSAY_RECIPE", "qwen2_5_7b_instruct"),
            "base_model": _base_model(env),
            "tier": env.get("ASSAY_TIER", "cert"),
            "checkpoint_repo": env.get("ASSAY_CHECKPOINT_REPO", ""),
        },
        "basis": {
            "began_at_unix": (basis or {}).get("began_at_unix", stamp),
            "uptime_seconds_at_begin": (basis or {}).get(
                "uptime_seconds_at_begin", pod.get("uptimeSeconds") or 0),
            "cost_per_hr": cost_per_hr,
            "container_disk_gb": container_disk,
            "volume_disk_gb": volume_disk,
            "gpu_display_name": gpu_display_name,
            "gpu_count": gpu_count,
            "rate_source": rate_source,
        },
        "provider": {
            "gpu_display_name": gpu_display_name,
            "gpu_count": gpu_count,
            "cost_per_hr": cost_per_hr,
            "uptime_seconds": uptime,
            "container_disk_gb": container_disk,
            "volume_disk_gb": volume_disk,
            "desired_status": pod.get("desiredStatus", ""),
            "data_center_id": env.get("ASSAY_REGION", _DEFAULT_REGION),
            "cloud_type": _DEFAULT_CLOUD_TYPE,
        },
        "rates": {
            "rate_table_version": _rates.RATE_TABLE_VERSION,
            "container_disk_gb_month_running":
                _rates.CONTAINER_DISK_GB_MONTH_RUNNING,
            "volume_disk_gb_month_running": _rates.VOLUME_DISK_GB_MONTH_RUNNING,
        },
        "marginal_usd": breakdown.as_dict(),
        "outcome": _model.classify_outcome(
            rc=rc, gate_result=gate_result, finalized=finalized),
        "finalized": bool(finalized),
    }


def cmd_begin(artifacts_dir: str, env, api=None, now=None) -> int:
    """Capture the cost basis. ALWAYS returns 0 - cost can never cost a run."""
    pod = probe_pod(env.get("RUNPOD_POD_ID", ""), api=api)
    catalog = None
    if not pod or pod.get("costPerHr") is None:
        catalog = probe_gpu_price(
            env.get("ASSAY_GPU_TYPE", _DEFAULT_GPU_TYPE), _DEFAULT_CLOUD_TYPE,
            api=api)
    record = build_record(env=env, pod=pod, basis=None, gate_result=None,
                          finalized=False, catalog_price=catalog, now=now)
    if write_record(artifacts_dir, record):
        print(f"assay.cost: basis captured, rate_source="
              f"{record['basis']['rate_source']}", file=sys.stderr)
    return 0


def cmd_finalize(artifacts_dir: str, env, log_path, rc, api=None, now=None) -> int:
    """True the record against provider truth and set the terminal outcome.

    ALWAYS returns 0 - cost can never cost a run.
    """
    basis = read_record(artifacts_dir)
    basis = (basis or {}).get("basis")
    pod = probe_pod(env.get("RUNPOD_POD_ID", ""), api=api)
    record = build_record(env=env, pod=pod, basis=basis,
                          gate_result=scan_gate_result(log_path), finalized=True,
                          rc=rc, catalog_price=None, now=now)
    if write_record(artifacts_dir, record):
        print(f"assay.cost: {record['outcome']} "
              f"total=${record['marginal_usd']['total']:.4f}", file=sys.stderr)
    return 0


# Reference duration for the pre-flight projection. A real wall-clock predictor is a
# DOWNSTREAM consumer of the ledger this feature produces (it cannot be fitted until
# run history exists), so this is an explicitly-labelled reference figure, not an
# estimate of THIS run.
_PREFLIGHT_REFERENCE_HOURS = 3.0


def preflight_line(env, api=None) -> str:
    """One-line cost expectation for launch.sh, printed BEFORE any spend.

    Quotes the rate matching the pinned cloud type. assay runs SECURE (network
    volumes exist only in secure DCs), and the community rate is ~30% lower, so
    quoting the wrong one here would set a false expectation for every run.
    """
    gpu_type = env.get("ASSAY_GPU_TYPE", _DEFAULT_GPU_TYPE)
    price = probe_gpu_price(gpu_type, _DEFAULT_CLOUD_TYPE, api=api)
    if price is None:
        return (f"[cost] {gpu_type} ({_DEFAULT_CLOUD_TYPE}): catalog price "
                "unavailable - the run still proceeds; cost is recorded in-pod")
    projected = price * _PREFLIGHT_REFERENCE_HOURS
    # The RATE is real; the projection is NOT a prediction of this run, because no
    # duration predictor exists yet - it is a downstream consumer of the very ledger
    # this feature produces, and cannot be fitted until run history accumulates.
    # Say so in the OPERATOR-VISIBLE string, not just in this comment: at 2 AM the
    # concrete dollar figure is what sticks, and a hedge that is easy to skim past
    # ("e.g.") is not a hedge. Give the reason too - "reference only" invites "why?",
    # and the answer is what stops someone repeating the number back as a quote.
    return (f"[cost] {gpu_type} ({_DEFAULT_CLOUD_TYPE}): ${price:.2f}/hr GPU. "
            "REFERENCE ONLY, not a prediction of this run (no duration predictor "
            f"yet): a {_PREFLIGHT_REFERENCE_HOURS:.0f}h run would be about "
            f"${projected:.2f} + storage. Actual cost is recorded to cost.json in "
            "the artifacts dir.")


def main(argv, env=None, api=None, now=None) -> int:
    """begin | finalize. Returns 0 on every runtime path; 2 only for bad argv,
    which is a wiring bug worth surfacing in the pod log."""
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(prog="assay.cost", add_help=True)
    sub = parser.add_subparsers(dest="command")
    p_begin = sub.add_parser("begin")
    p_begin.add_argument("artifacts_dir")
    p_final = sub.add_parser("finalize")
    p_final.add_argument("artifacts_dir")
    p_final.add_argument("--rc", default="")
    p_final.add_argument("--log", default="")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse exits with code 0 for -h/--help (a successful operation for a
        # junior admin at 2 AM asking syntax questions). Bad argv exits with code 2
        # (a wiring bug worth surfacing). Return success for help, failure for argv
        # errors.
        exit_code = e.code if e.code is not None else 0
        return 0 if exit_code == 0 else 2
    if args.command == "begin":
        return cmd_begin(args.artifacts_dir, env, api=api, now=now)
    if args.command == "finalize":
        return cmd_finalize(args.artifacts_dir, env, args.log, args.rc,
                            api=api, now=now)
    return 2
