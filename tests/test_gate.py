import pytest

from assay.config import DEFAULT_GATE, GateThresholds
from assay.gate import evaluate_gate, render_delta_table


ACC = ("gsm8k", "arc_challenge")
PPL = "wikitext"


def _results(gsm8k, arc, ppl, gsm8k_metric="acc"):
    return {
        "gsm8k": {"metric": gsm8k_metric, "value": gsm8k},
        "arc_challenge": {"metric": "acc", "value": arc},
        "wikitext": {"metric": "word_perplexity", "value": ppl},
    }


def test_passes_when_within_tolerance():
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.795, 0.598, 10.05)  # tiny drops, +0.5% ppl
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is True
    assert r.reasons == ()
    assert 0.99 <= r.mean_retention <= 1.0


def test_fails_on_low_mean_retention():
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.70, 0.50, 10.0)  # big accuracy loss
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is False
    assert any("mean retention" in reason for reason in r.reasons)


def test_fails_on_single_task_cliff():
    base = _results(0.80, 0.60, 10.0)
    # arc drops 5 pts - and mean retention (~0.958) is ALSO below the 0.99
    # floor here, so both the single-task and mean-retention reasons fire.
    quant = _results(0.799, 0.55, 10.0)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is False
    assert any("arc_challenge" in reason for reason in r.reasons)


def test_fails_on_perplexity_blowup():
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.80, 0.60, 10.3)  # +3% ppl
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is False
    assert any("perplexity" in reason for reason in r.reasons)


# --- Boundary tests: pin each threshold exactly so a comparison-operator ---
# --- regression (e.g. `>` flipped to `>=`, or vice versa) fails loudly. ---


def test_single_task_drop_exactly_at_boundary_passes():
    # arc: 1.0 -> 0.98 is a decimal-exact 2.0-pt drop. In binary float this
    # computes as 2.0000000000000018 (the exact repro from the gate.py FP
    # fix), so a naive `> MAX_SINGLE_DROP_PTS` compare would wrongly fail a
    # spot-on boundary. gsm8k is unchanged and mean retention lands exactly
    # at 0.99 (1.0 + 0.98) / 2, so this isolates to the single-task check.
    base = _results(1.0, 1.0, 10.0)
    quant = _results(1.0, 0.98, 10.0)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is True
    assert r.reasons == ()


def test_single_task_drop_over_boundary_fails():
    # arc: 1.0 -> 0.975 is a clean 2.5-pt drop, well past the 2.0 allowance.
    base = _results(1.0, 1.0, 10.0)
    quant = _results(1.0, 0.975, 10.0)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is False
    assert any("arc_challenge" in reason for reason in r.reasons)


def test_mean_retention_exact_boundary_passes():
    # Both tasks retain exactly 0.99 (0.99 / 1.0), so mean_retention == 0.99
    # exactly - the floor is inclusive (fails only when STRICTLY below).
    # Each task's own drop is only 1.0 pt, so the single-task check can't fire.
    base = _results(1.0, 1.0, 10.0)
    quant = _results(0.99, 0.99, 10.0)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is True
    assert r.reasons == ()


def test_mean_retention_just_under_fails():
    # Both tasks retain 0.9899, just under the 0.99 floor - mean_retention
    # fails while each task's own 1.01-pt drop stays well under the 2.0-pt
    # single-task allowance, isolating the mean-retention reason.
    base = _results(1.0, 1.0, 10.0)
    quant = _results(0.9899, 0.9899, 10.0)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is False
    assert any("mean retention" in reason for reason in r.reasons)
    assert not any("dropped" in reason for reason in r.reasons)


def test_perplexity_just_under_bar_passes():
    # +2.9% ppl is just under the 3% bar (raised from 1% on metal evidence), so PASS.
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.80, 0.60, 10.29)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is True
    assert r.reasons == ()


def test_perplexity_over_bar_fails():
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.80, 0.60, 10.4)  # clean +4% increase, over the 3% bar
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is False
    assert any("perplexity" in reason for reason in r.reasons)


def test_empty_accuracy_tasks_raises_value_error():
    # Public API guard: divide-by-len(acc_deltas) must not surface as a bare
    # ZeroDivisionError to a caller who passes no accuracy tasks.
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.80, 0.60, 10.0)
    with pytest.raises(ValueError, match="at least one accuracy task"):
        evaluate_gate(base, quant, (), PPL)


def test_gsm8k_style_metric_label_flows_through_gate():
    # C1 coverage: a task whose resolved metric is exact_match,strict-match
    # (gsm8k's real lm-eval shape, not "acc") must flow through the gate with
    # that real metric name on its TaskDelta - not a hardcoded "acc" label.
    base = _results(0.80, 0.60, 10.0, gsm8k_metric="exact_match,strict-match")
    quant = _results(0.795, 0.598, 10.05, gsm8k_metric="exact_match,strict-match")
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is True
    gsm8k_delta = next(d for d in r.accuracy_deltas if d.task == "gsm8k")
    assert gsm8k_delta.metric == "exact_match,strict-match"
    table = render_delta_table(r)
    assert "exact_match,strict-match" in table


def test_delta_table_is_ascii_markdown():
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.795, 0.598, 10.05)
    table = render_delta_table(evaluate_gate(base, quant, ACC, PPL))
    assert "| task" in table
    assert "gsm8k" in table
    assert table.isascii()


# --- GateThresholds value object + optional perplexity ---


def _mk(acc_base, acc_quant, ppl_base=None, ppl_quant=None):
    base = {"mmlu": {"metric": "acc", "value": acc_base}}
    quant = {"mmlu": {"metric": "acc", "value": acc_quant}}
    if ppl_base is not None:
        base["wikitext"] = {"metric": "word_perplexity", "value": ppl_base}
        quant["wikitext"] = {"metric": "word_perplexity", "value": ppl_quant}
    return base, quant


def test_default_gate_values():
    assert DEFAULT_GATE == GateThresholds(0.99, 2.0, 0.03)


def test_perplexity_optional_skips_check_and_nulls_delta():
    base, quant = _mk(0.70, 0.70)  # no perplexity task supplied
    r = evaluate_gate(base, quant, ("mmlu",), None)
    assert r.passed is True
    assert r.perplexity_delta is None


def test_custom_thresholds_are_honored():
    base, quant = _mk(0.70, 0.68)  # 2.0 pt drop
    strict = GateThresholds(0.99, 1.0, 0.03)  # max 1.0 pt drop
    r = evaluate_gate(base, quant, ("mmlu",), None, thresholds=strict)
    assert r.passed is False
    assert any("dropped" in reason for reason in r.reasons)


# --- CI-aware significance gate (k_stderr) ---


def _se_results(gsm8k, arc, ppl, gsm8k_se, arc_se):
    return {
        "gsm8k": {"metric": "exact_match,none", "value": gsm8k, "stderr": gsm8k_se},
        "arc_challenge": {"metric": "acc,none", "value": arc, "stderr": arc_se},
        "wikitext": {"metric": "word_perplexity", "value": ppl, "stderr": None},
    }


_CI_GATE = GateThresholds(min_mean_retention=None, max_single_drop_pts=None,
                          max_ppl_increase=0.03, k_stderr=2.0)


def test_significance_pass_when_drop_within_k_stderr():
    # combined_se = sqrt(.02^2+.02^2)=.028284; 2*that=.056569; drop .05 < threshold -> not significant
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.75, 0.60, 10.0, 0.02, 0.02)
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    assert r.passed is True
    assert all(d.significant is False for d in r.accuracy_deltas)


def test_significance_fail_when_drop_exceeds_k_stderr():
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.73, 0.60, 10.0, 0.02, 0.02)  # drop .07 > .056569
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    assert r.passed is False
    assert any(d.task == "gsm8k" and d.significant for d in r.accuracy_deltas)
    assert any("gsm8k" in reason for reason in r.reasons)


def test_significance_boundary_exactly_at_threshold_passes():
    # delta exactly -k*combined_se; strict '<' means exactly-at is NOT significant
    import math
    combined = math.sqrt(0.02**2 + 0.02**2)
    quant_val = 0.80 - 2.0 * combined
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(quant_val, 0.60, 10.0, 0.02, 0.02)
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    assert r.passed is True


def test_significance_gate_ignores_point_drop_and_mean_retention():
    # a 7-pt drop that a point-gate would fail; here it is insignificant (big stderr) -> pass
    base = _se_results(0.50, 0.60, 10.0, 0.06, 0.02)
    quant = _se_results(0.44, 0.60, 10.0, 0.06, 0.02)  # drop .06 < 2*sqrt(.06^2+.06^2)=.1697
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    assert r.passed is True


def test_significance_missing_stderr_raises():
    base = {"gsm8k": {"metric": "exact_match,none", "value": 0.80, "stderr": None},
            "wikitext": {"metric": "word_perplexity", "value": 10.0, "stderr": None}}
    quant = {"gsm8k": {"metric": "exact_match,none", "value": 0.79, "stderr": 0.02},
             "wikitext": {"metric": "word_perplexity", "value": 10.1, "stderr": None}}
    with pytest.raises(ValueError, match="gsm8k"):
        evaluate_gate(base, quant, ("gsm8k",), "wikitext", _CI_GATE)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_significance_nonfinite_stderr_refuses_to_pass(bad):
    # A degenerate stderr (NaN from lm-eval's ddof=1 mean_stderr on n=1/limit=1, or inf)
    # must never silently PASS a real regression: sqrt(nan)=nan and every "significant
    # drop" comparison is False. The gate must REFUSE, not gate. This is the sole
    # accuracy criterion in significance mode, so a silent PASS here certifies a broken
    # quant. base 0.70 -> quant 0.30 is a 40-pt drop that must not slip through.
    base = {"gsm8k": {"metric": "exact_match,none", "value": 0.70, "stderr": bad},
            "wikitext": {"metric": "word_perplexity", "value": 10.0, "stderr": None}}
    quant = {"gsm8k": {"metric": "exact_match,none", "value": 0.30, "stderr": bad},
             "wikitext": {"metric": "word_perplexity", "value": 10.1, "stderr": None}}
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_gate(base, quant, ("gsm8k",), "wikitext", _CI_GATE)


def test_pick_stderr_normalizes_nonfinite_to_none():
    # The primary guard: parse-side _pick_stderr turns a NaN/inf stderr into None (the
    # "missing stderr" the gate then refuses on), so the degenerate value never reaches
    # the significance math. Exact-key and filter-suffixed forms both normalize.
    from assay.evaluate import _pick_stderr
    assert _pick_stderr({"acc,none": 0.5, "acc_stderr,none": float("nan")}, "acc,none") is None
    assert _pick_stderr({"acc,none": 0.5, "acc_stderr,none": float("inf")}, "acc,none") is None
    assert _pick_stderr({"acc,none": 0.5, "acc_stderr,none": 0.02}, "acc,none") == 0.02


def test_ppl_still_gates_under_ci_recipe():
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.80, 0.60, 10.5, 0.02, 0.02)  # +5% ppl > 3%
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    assert r.passed is False
    assert any("perplexity" in reason for reason in r.reasons)


def test_default_gate_still_point_gates_over_stderr_dicts():
    # Qwen path: DEFAULT_GATE (k_stderr None) ignores stderr, uses point checks unchanged
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.75, 0.598, 10.0, 0.02, 0.02)  # gsm8k -5pt > 2pt bar
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext")
    assert r.passed is False
    assert any("dropped" in reason for reason in r.reasons)


def test_delta_table_shows_stderr_column_and_marker():
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.73, 0.60, 10.0, 0.02, 0.02)
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    table = render_delta_table(r)
    assert "stderr" in table
    assert "*" in table   # significant regression marked


def test_significance_never_flags_an_improvement():
    # one-sided contract: quant strictly better than base is never a significant regression
    base = _se_results(0.70, 0.60, 10.0, 0.01, 0.01)
    quant = _se_results(0.80, 0.60, 10.0, 0.01, 0.01)  # +10pt, tiny stderr
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", _CI_GATE)
    assert r.passed is True
    assert all(d.significant is False for d in r.accuracy_deltas)


def test_default_gate_render_has_no_stderr_column():
    # Qwen artifact path: DEFAULT_GATE over stderr-carrying dicts -> table identical to today
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.795, 0.598, 10.05, 0.02, 0.02)
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext")  # DEFAULT_GATE
    table = render_delta_table(r)
    assert "stderr" not in table
    assert "*" not in table


def test_empty_gate_passes_anything():
    empty = GateThresholds(min_mean_retention=None, max_single_drop_pts=None,
                           max_ppl_increase=None, k_stderr=None)
    base = _se_results(0.80, 0.60, 10.0, 0.02, 0.02)
    quant = _se_results(0.10, 0.10, 99.0, 0.02, 0.02)  # catastrophic drop, but no checks active
    r = evaluate_gate(base, quant, ("gsm8k", "arc_challenge"), "wikitext", empty)
    assert r.passed is True
    assert r.reasons == ()


SIG_GATE = GateThresholds(min_mean_retention=None, max_single_drop_pts=None,
                          max_ppl_increase=0.03, k_stderr=2.0)


def _sig_results(a_base, a_quant, se=0.05):
    # one accuracy task with stderrs (significance mode) + ppl
    return (
        {"aime24_avg": {"metric": "exact_match,avg", "value": a_base, "stderr": se},
         "wikitext": {"metric": "word_perplexity", "value": 10.0, "stderr": None}},
        {"aime24_avg": {"metric": "exact_match,avg", "value": a_quant, "stderr": se},
         "wikitext": {"metric": "word_perplexity", "value": 10.1, "stderr": None}},
    )


def test_render_significance_hides_mean_headline_shows_verdict():
    base, quant = _sig_results(0.50, 0.49)  # tiny drop, within 2*combined_se
    r = evaluate_gate(base, quant, ("aime24_avg",), "wikitext", SIG_GATE)
    out = render_delta_table(r, SIG_GATE)
    assert "Mean accuracy retention:" not in out       # #2: noise headline suppressed
    assert "Gate: PASS" in out                          # verdict line instead
    assert "No task showed a statistically significant regression." in out  # #5


def test_render_point_gate_keeps_mean_headline():
    # Qwen-style point gate (DEFAULT_GATE): mean IS a criterion -> keep it, byte-compat
    base = {"gsm8k": {"metric": "acc", "value": 0.80},
            "wikitext": {"metric": "word_perplexity", "value": 10.0}}
    quant = {"gsm8k": {"metric": "acc", "value": 0.795},
             "wikitext": {"metric": "word_perplexity", "value": 10.05}}
    r = evaluate_gate(base, quant, ("gsm8k",), "wikitext", DEFAULT_GATE)
    out = render_delta_table(r, DEFAULT_GATE)
    assert "Mean accuracy retention:" in out
    assert "No task showed a statistically significant regression." not in out


def test_render_no_thresholds_is_backcompat():
    base = {"gsm8k": {"metric": "acc", "value": 0.80}}
    quant = {"gsm8k": {"metric": "acc", "value": 0.79}}
    r = evaluate_gate(base, quant, ("gsm8k",), None, DEFAULT_GATE)
    out = render_delta_table(r)  # no thresholds arg
    assert "Mean accuracy retention:" in out


def test_render_ppl_only_gate_passes_without_crash():
    # A gate with min_mean_retention AND k_stderr both None (ppl-only) hits the
    # verdict-line else-branch with k=None; formatting k must not TypeError on PASS.
    base = {"arc_challenge": {"metric": "acc", "value": 0.60},
            "wikitext": {"metric": "word_perplexity", "value": 10.0}}
    quant = {"arc_challenge": {"metric": "acc", "value": 0.60},
             "wikitext": {"metric": "word_perplexity", "value": 10.05}}
    ppl_only = GateThresholds(min_mean_retention=None, max_single_drop_pts=None,
                              max_ppl_increase=0.03, k_stderr=None)
    r = evaluate_gate(base, quant, ("arc_challenge",), "wikitext", ppl_only)
    out = render_delta_table(r, ppl_only)
    assert "Gate: PASS" in out
    assert "Mean accuracy retention:" not in out


def test_gate_refuses_pass_on_non_finite_input():
    # A NaN that would otherwise sail through (all NaN comparisons are False).
    base = {"t": {"metric": "acc,none", "value": 0.80, "stderr": None}}
    quant = {"t": {"metric": "acc,none", "value": float("nan"), "stderr": None}}
    thr = GateThresholds(min_mean_retention=0.99, max_single_drop_pts=2.0, max_ppl_increase=None)
    result = evaluate_gate(base, quant, ("t",), None, thr)
    assert result.passed is False
    assert any("non-finite" in r for r in result.reasons)


def test_gate_refuses_pass_on_non_finite_perplexity():
    # Cover the perplexity non-finite branch: baseline/quantized dicts have
    # finite accuracy tasks but a non-finite perplexity value on the quantized side.
    # Set thresholds so the finite accuracies would pass, then verify the non-finite
    # perplexity rejection fires.
    base = _results(0.80, 0.60, 10.0)
    quant = _results(0.80, 0.60, float("inf"))  # perplexity is infinite
    thr = GateThresholds(min_mean_retention=0.99, max_single_drop_pts=2.0, max_ppl_increase=0.03)
    result = evaluate_gate(base, quant, ACC, PPL, thr)
    assert result.passed is False
    assert any("non-finite" in r for r in result.reasons)
    # Confirm the reason mentions the perplexity task
    assert any("wikitext" in r for r in result.reasons)


def test_gate_refuses_pass_on_non_finite_baseline_accuracy():
    # Cover the baseline accuracy non-finite branch: baseline has a NaN for
    # one of the accuracy tasks while quantized is fine.
    base = _results(float("nan"), 0.60, 10.0)  # gsm8k is NaN
    quant = _results(0.80, 0.60, 10.0)
    thr = GateThresholds(min_mean_retention=0.99, max_single_drop_pts=2.0, max_ppl_increase=0.03)
    result = evaluate_gate(base, quant, ACC, PPL, thr)
    assert result.passed is False
    assert any("non-finite" in r for r in result.reasons)
    # Confirm the reason mentions the baseline and the task
    assert any("gsm8k" in r for r in result.reasons)
