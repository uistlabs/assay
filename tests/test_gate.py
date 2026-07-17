import pytest

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
    # arc drops 5 pts -- and mean retention (~0.958) is ALSO below the 0.99
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
    # exactly -- the floor is inclusive (fails only when STRICTLY below).
    # Each task's own drop is only 1.0 pt, so the single-task check can't fire.
    base = _results(1.0, 1.0, 10.0)
    quant = _results(0.99, 0.99, 10.0)
    r = evaluate_gate(base, quant, ACC, PPL)
    assert r.passed is True
    assert r.reasons == ()


def test_mean_retention_just_under_fails():
    # Both tasks retain 0.9899, just under the 0.99 floor -- mean_retention
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
    # that real metric name on its TaskDelta -- not a hardcoded "acc" label.
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
