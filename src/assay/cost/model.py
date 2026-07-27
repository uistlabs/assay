"""Pure cost math - zero I/O, zero network, no mocks needed to test any of it.

The host-side reconciler imports the SAME functions the pod uses. Two
implementations of this math is how a reconciler starts disagreeing with the thing
it reconciles, which is why this module exists separately from collect.py rather
than being inlined there. ASCII only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from assay.cost import rates as _default_rates

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class CostBreakdown:
    """Marginal USD for one pod-run, split so a quote can show its work."""
    gpu: float
    container_disk: float
    volume_disk: float
    total: float

    def as_dict(self) -> dict:
        return asdict(self)


def _non_negative(value) -> float:
    """Coerce to a non-negative float. None/garbage -> 0.0.

    Deliberately total: a cost record must never abort a run, and a NEGATIVE cost
    would silently credit a run and corrupt a roll-up, so clamp rather than
    propagate.
    """
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def marginal_usd(*, cost_per_hr, uptime_seconds, container_disk_gb=0,
                 volume_disk_gb=0, rates=_default_rates) -> CostBreakdown:
    """Marginal USD for one pod-run.

    `cost_per_hr` is RunPod's GPU-COMPUTE-ONLY rate (verified 2026-07-25: storage
    bills separately, and savings plans cover compute only), so disk is added here
    from the rate table rather than assumed to be included.

    Storage is prorated by the run's actual hours - charging a full month's disk
    against a 3-hour run would overstate it by roughly 250x.
    """
    hours = _non_negative(uptime_seconds) / SECONDS_PER_HOUR
    gpu = _non_negative(cost_per_hr) * hours
    container = (_non_negative(container_disk_gb)
                 * rates.CONTAINER_DISK_GB_MONTH_RUNNING / rates.HOURS_PER_MONTH
                 * hours)
    volume = (_non_negative(volume_disk_gb)
              * rates.VOLUME_DISK_GB_MONTH_RUNNING / rates.HOURS_PER_MONTH
              * hours)
    # This record is serialized verbatim into cost.json - an auditable artifact
    # that will eventually back customer invoices - so it must never carry raw
    # float noise (e.g. gpu=3.0915500000000002 where the spec's own example shows
    # a clean value). Round every component to 6dp (micro-dollars: far beyond
    # meaningful precision, but enough resolution that summing many records does
    # not drift), uniformly with total, and derive total from the ALREADY-ROUNDED
    # components so total == gpu + container_disk + volume_disk holds exactly in
    # the stored record rather than by floating-point coincidence. Do not
    # re-round an already-rounded value to fewer places elsewhere (e.g.
    # round(gpu, 4)) - double-rounding can land on the wrong side of a tie point
    # and silently disagree with the value actually on disk.
    gpu = round(gpu, 6)
    container = round(container, 6)
    volume = round(volume, 6)
    total = round(gpu + container + volume, 6)
    return CostBreakdown(
        gpu=gpu,
        container_disk=container,
        volume_disk=volume,
        total=total,
    )


# The billing-relevant outcome enum. `begin` writes IN_PROGRESS; only `finalize`
# sets a terminal value, so a record still reading IN_PROGRESS at roll-up time IS
# the infra-fail signal (the pod died before teardown) - abrupt-death detection
# with no extra machinery.
OUTCOME_IN_PROGRESS = "in_progress"
OUTCOME_PASS = "pass"
OUTCOME_GATE_FAIL = "gate_fail"
OUTCOME_INFRA_FAIL = "infra_fail"
OUTCOME_OPERATOR_ABORT = "operator_abort"

# Shell exit codes for a signalled process: 128 + signal number.
_RC_SIGINT = 130
_RC_SIGTERM = 143


def classify_outcome(*, rc, gate_result, finalized: bool = True) -> str:
    """Map pod_entry.sh's (rc, gate result) into the billing-relevant outcome.

    `rc` arrives as the shell's `${rc:-}` - a string, possibly EMPTY. Empty means
    teardown's EXIT trap fired before the job's exit code was assigned, i.e. the pod
    died mid-run.

    `gate_result` is the NORMALIZED token "pass" | "fail" | None. collect.py owns
    the marker grep so this module needs no assay.job import and stays pure.

    The gate_fail vs infra_fail split is the commercially load-bearing one:
    gate_fail means the client's model honestly missed the bar (billable audit work,
    and the telemetry report is the deliverable); infra_fail means OUR pipeline
    broke, which we absorb. Encoding it here, at the moment it is known, is the only
    time it is cheap - it cannot be reconstructed from a terminated pod.
    """
    if not finalized:
        return OUTCOME_IN_PROGRESS
    if rc is None or str(rc).strip() == "":
        return OUTCOME_INFRA_FAIL
    try:
        code = int(str(rc).strip())
    except (TypeError, ValueError):
        return OUTCOME_INFRA_FAIL
    # Checked BEFORE the marker: a run killed after printing its marker was still
    # aborted, and billing it as a clean pass would overstate what was delivered.
    if code in (_RC_SIGINT, _RC_SIGTERM):
        return OUTCOME_OPERATOR_ABORT
    if code == 0 and gate_result == "pass":
        return OUTCOME_PASS
    if code == 0 and gate_result == "fail":
        return OUTCOME_GATE_FAIL
    return OUTCOME_INFRA_FAIL
