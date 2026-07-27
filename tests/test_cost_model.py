from assay.cost.model import (
    CostBreakdown,
    OUTCOME_GATE_FAIL,
    OUTCOME_IN_PROGRESS,
    OUTCOME_INFRA_FAIL,
    OUTCOME_OPERATOR_ABORT,
    OUTCOME_PASS,
    classify_outcome,
    marginal_usd,
)


def test_gpu_cost_is_rate_times_hours():
    # RTX 5090 secure = $0.99/hr, measured 2026-07-25. 3600s = exactly one hour.
    b = marginal_usd(cost_per_hr=0.99, uptime_seconds=3600)
    assert b.gpu == 0.99
    assert b.total == 0.99


def test_three_hour_cert_run_matches_the_spec_economics():
    # Spec F7: about $3.00 for a 3h run on a 40 GB container disk. This is the
    # headline number every quote is built on, so pin it.
    #
    # Assert directly against the 6dp values the record actually stores, rather
    # than re-rounding a rounded number to 4dp. gpu's raw value (3.0915500000000002)
    # sits fractionally ABOVE the tie point, but round(gpu, 6) == 3.09155 is the
    # nearest double to 3.09155, which sits fractionally BELOW it - so
    # round(round(gpu, 6), 4) == 3.0915 while round(gpu, 4) == 3.0916. Comparing
    # to the stored 6dp value sidesteps that double-rounding trap entirely.
    b = marginal_usd(cost_per_hr=0.99, uptime_seconds=11242, container_disk_gb=40)
    assert b.gpu == 3.09155
    assert b.container_disk == 0.017111
    assert b.volume_disk == 0.0
    assert b.total == 3.108661


def test_container_disk_is_prorated_by_hours_not_charged_monthly():
    # $0.10/GB/month over 730h. A whole month's charge landing on one run would
    # overstate a 3h run by ~250x.
    b = marginal_usd(cost_per_hr=0.0, uptime_seconds=3600, container_disk_gb=40)
    assert round(b.container_disk, 6) == round(40 * 0.10 / 730.0, 6)


def test_volume_disk_is_counted_when_present():
    b = marginal_usd(cost_per_hr=0.0, uptime_seconds=3600, volume_disk_gb=20)
    assert round(b.volume_disk, 6) == round(20 * 0.10 / 730.0, 6)
    assert b.container_disk == 0.0


def test_zero_uptime_is_zero_cost():
    b = marginal_usd(cost_per_hr=0.99, uptime_seconds=0, container_disk_gb=40)
    assert b.total == 0.0


def test_none_inputs_degrade_to_zero_rather_than_raising():
    # A cost record must never abort a run. A partial get_pod payload (missing
    # costPerHr or uptimeSeconds) has to produce a number, not an exception.
    b = marginal_usd(cost_per_hr=None, uptime_seconds=None,
                     container_disk_gb=None, volume_disk_gb=None)
    assert b == CostBreakdown(gpu=0.0, container_disk=0.0, volume_disk=0.0, total=0.0)


def test_negative_inputs_are_clamped_to_zero():
    # A negative uptime or rate is nonsense; a NEGATIVE cost would silently credit
    # a run and corrupt a roll-up, so clamp rather than propagate.
    b = marginal_usd(cost_per_hr=-5.0, uptime_seconds=-3600, container_disk_gb=-40)
    assert b.total == 0.0


def test_breakdown_serializes_for_the_record():
    b = marginal_usd(cost_per_hr=0.99, uptime_seconds=3600)
    d = b.as_dict()
    assert set(d) == {"gpu", "container_disk", "volume_disk", "total"}
    assert d["gpu"] == 0.99


def test_totals_are_internally_consistent():
    b = marginal_usd(cost_per_hr=0.99, uptime_seconds=7200,
                     container_disk_gb=40, volume_disk_gb=10)
    assert round(b.total, 6) == round(b.gpu + b.container_disk + b.volume_disk, 6)


def test_unfinalized_record_is_in_progress():
    # begin() writes this. A record still reading in_progress at roll-up time IS
    # the infra-fail signal - the pod died before reaching teardown.
    assert classify_outcome(rc=None, gate_result=None, finalized=False) == (
        OUTCOME_IN_PROGRESS)


def test_clean_exit_with_pass_marker_is_pass():
    assert classify_outcome(rc="0", gate_result="pass") == OUTCOME_PASS


def test_clean_exit_with_fail_marker_is_gate_fail():
    # Commercially load-bearing: gate_fail means the CLIENT's model honestly missed
    # the bar. That is billable audit work and the artifacts ARE the deliverable.
    assert classify_outcome(rc="0", gate_result="fail") == OUTCOME_GATE_FAIL


def test_empty_rc_is_infra_fail():
    # pod_entry.sh passes ${rc:-}. teardown() is an EXIT trap, so an unset rc means
    # it fired before the job's exit code was assigned - the pod died mid-run.
    assert classify_outcome(rc="", gate_result=None) == OUTCOME_INFRA_FAIL
    assert classify_outcome(rc=None, gate_result=None) == OUTCOME_INFRA_FAIL
    assert classify_outcome(rc="   ", gate_result=None) == OUTCOME_INFRA_FAIL


def test_nonzero_rc_without_marker_is_infra_fail():
    # Our pipeline broke. Distinct from gate_fail because we absorb this one.
    assert classify_outcome(rc="3", gate_result=None) == OUTCOME_INFRA_FAIL


def test_zero_rc_without_marker_is_infra_fail():
    # The silent-noop signature pod_entry.sh already guards: exit 0, no work done.
    assert classify_outcome(rc="0", gate_result=None) == OUTCOME_INFRA_FAIL


def test_nonzero_rc_with_pass_marker_is_infra_fail():
    # Commercially critical: a nonzero exit with a stray pass marker must never be
    # billed as a clean pass. The code == 0 guard ensures this - a failed run is
    # not delivered, even if it printed the pass marker. Protects against billing
    # a broken pipeline as if it shipped a passing result.
    assert classify_outcome(rc="3", gate_result="pass") == OUTCOME_INFRA_FAIL


def test_signal_exit_codes_are_operator_abort():
    # 130 = SIGINT, 143 = SIGTERM. A human or the platform stopped the run; that is
    # neither our pipeline breaking nor the client's model failing.
    assert classify_outcome(rc="130", gate_result=None) == OUTCOME_OPERATOR_ABORT
    assert classify_outcome(rc="143", gate_result=None) == OUTCOME_OPERATOR_ABORT


def test_signal_exit_wins_even_with_a_marker_present():
    # A run killed after printing its marker was still aborted; billing it as a
    # clean pass would overstate what was delivered.
    assert classify_outcome(rc="130", gate_result="pass") == OUTCOME_OPERATOR_ABORT


def test_garbage_rc_is_infra_fail_not_an_exception():
    assert classify_outcome(rc="banana", gate_result="pass") == OUTCOME_INFRA_FAIL


def test_gate_fail_and_infra_fail_are_distinct_values():
    # The whole commercial point: one is billable, the other we absorb. If these
    # ever collapse to the same string the SOW distinction silently disappears.
    assert OUTCOME_GATE_FAIL != OUTCOME_INFRA_FAIL
