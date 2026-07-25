from __future__ import annotations

import math
from dataclasses import dataclass

from assay.config import DEFAULT_GATE, GateThresholds


@dataclass(frozen=True)
class TaskDelta:
    task: str
    metric: str
    baseline: float
    quantized: float
    delta: float       # quantized - baseline
    retention: float   # quantized / baseline (guard against zero baseline)
    base_stderr: float | None = None
    quant_stderr: float | None = None
    combined_stderr: float | None = None   # sqrt(base_se^2 + quant_se^2)
    significant: bool | None = None        # significant regression? None => not evaluated


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    accuracy_deltas: tuple[TaskDelta, ...]
    perplexity_delta: TaskDelta | None
    mean_retention: float


def _delta(task, metric, base, quant, base_se=None, quant_se=None, k_stderr=None) -> TaskDelta:
    retention = quant / base if base else 0.0
    combined_se = None
    significant = None
    if k_stderr is not None:
        # Backstop (defense in depth): parse_results' _pick_stderr already normalizes a
        # missing OR non-finite stderr to None, but the gate is the certification
        # authority and must NEVER PASS on a degenerate stderr even if a future refactor
        # feeds _delta directly. A None or non-finite (NaN/inf) stderr makes combined_se
        # non-finite -> every "significant drop" test is False -> silent PASS. Refuse it.
        if (base_se is None or quant_se is None
                or not math.isfinite(base_se) or not math.isfinite(quant_se)):
            raise ValueError(
                f"gate: task {task!r} has k_stderr set but its stderr is missing or "
                f"non-finite (baseline={base_se}, quantized={quant_se}) - cannot run a "
                "significance check. Either the metric reports no stderr (wrong metric "
                "for a significance gate, or a degenerate n=1/limit=1 eval) or "
                "parse_results did not capture it.")
        combined_se = math.sqrt(base_se ** 2 + quant_se ** 2)
        # one-sided: only a real DROP fails. delta<0 is a drop; flag it significant
        # only when the drop exceeds k combined standard errors.
        # FP-tolerance, same class the point-drop check rounds away: a drop landing
        # exactly on -k*combined_se can compute as that value +/- ~2e-17 and must not
        # read as significant. Round the excess to 12 decimals before the sign test
        # (12 decimals crushes the FP noise while staying far below real eval-score
        # resolution, so a genuine regression still fails).
        significant = round((quant - base) + k_stderr * combined_se, 12) < 0.0
    return TaskDelta(task, metric, base, quant, quant - base, retention,
                     base_se, quant_se, combined_se, significant)


def evaluate_gate(baseline, quantized, accuracy_tasks, perplexity_task,
                  thresholds: GateThresholds = DEFAULT_GATE) -> GateResult:
    """Compare baseline vs quantized results; return pass/fail with reasons.

    Each results dict maps task -> {"metric": name, "value": v} (see
    evaluate.parse_results). The metric name is carried through per-task so
    the delta table/model card label each row with the real lm-eval metric
    (e.g. gsm8k | exact_match,strict-match) rather than a hardcoded "acc".
    perplexity_task=None skips the perplexity check entirely and leaves
    GateResult.perplexity_delta as None."""
    if not accuracy_tasks:
        raise ValueError(
            "evaluate_gate requires at least one accuracy task "
            "(accuracy_tasks was empty; mean retention is undefined with no tasks)"
        )
    # Backstop (defense in depth): the gate is the certification authority and must NEVER
    # emit PASS on a non-finite input, even if a future refactor bypasses parse_results'
    # primary guard. Collect offenders and fail with a clear reason.
    non_finite: list[str] = []
    for t in accuracy_tasks:
        for side, d in (("baseline", baseline), ("quantized", quantized)):
            if not math.isfinite(d[t]["value"]):
                non_finite.append(f"{t} {side} metric is non-finite ({d[t]['value']})")
    if perplexity_task is not None:
        for side, d in (("baseline", baseline), ("quantized", quantized)):
            if not math.isfinite(d[perplexity_task]["value"]):
                non_finite.append(f"{perplexity_task} {side} metric is non-finite ({d[perplexity_task]['value']})")
    k = thresholds.k_stderr
    acc_deltas = tuple(
        _delta(t, baseline[t]["metric"], baseline[t]["value"], quantized[t]["value"],
               baseline[t].get("stderr"), quantized[t].get("stderr"), k)
        for t in accuracy_tasks
    )
    mean_retention = sum(d.retention for d in acc_deltas) / len(acc_deltas)

    reasons: list[str] = list(non_finite)
    if thresholds.min_mean_retention is not None and mean_retention < thresholds.min_mean_retention:
        reasons.append(f"mean retention {mean_retention:.4f} < {thresholds.min_mean_retention}")

    if k is not None:
        # Significance path: fail a task only when its drop is beyond k combined
        # stderrs - the right test for tiny high-variance generative sets (a single
        # aime question is 3.3 pts, so a point-drop bar fails a perfect quant).
        for d in acc_deltas:
            if d.significant:
                reasons.append(
                    f"{d.task} regressed {(-d.delta * 100.0):.2f} pts "
                    f"(delta {d.delta:+.4f} beyond -{k}*{d.combined_stderr:.4f} combined stderr)")

    if thresholds.max_single_drop_pts is not None:
        for d in acc_deltas:
            # FP-tolerance: binary floats can't represent 0.90-0.88 exactly, so a
            # decimal-exact 2.0-pt drop can compute as 2.0000000000000018 and wrongly
            # fail a spot-on boundary. Round off the representation error first.
            drop_pts = -d.delta * 100.0
            if round(drop_pts, 9) > thresholds.max_single_drop_pts:
                reasons.append(f"{d.task} dropped {drop_pts:.2f} pts > {thresholds.max_single_drop_pts}")

    ppl = None
    if perplexity_task is not None:
        ppl = _delta(perplexity_task, baseline[perplexity_task]["metric"],
                     baseline[perplexity_task]["value"], quantized[perplexity_task]["value"])
        if thresholds.max_ppl_increase is not None:
            ppl_increase = (ppl.quantized - ppl.baseline) / ppl.baseline if ppl.baseline else 1.0
            if ppl_increase > thresholds.max_ppl_increase:
                reasons.append(
                    f"perplexity increased {ppl_increase * 100:.2f}% > {thresholds.max_ppl_increase * 100:.2f}%")

    return GateResult(
        passed=not reasons, reasons=tuple(reasons),
        accuracy_deltas=acc_deltas, perplexity_delta=ppl, mean_retention=mean_retention,
    )


def render_delta_table(result: GateResult, thresholds: GateThresholds | None = None) -> str:
    """ASCII markdown table of the deltas - for the model card and heartbeat.
    Adds a combined-stderr column + a significance marker only when the gate ran
    in significance mode (stderrs present); otherwise renders exactly as before.

    thresholds=None preserves the legacy output (mean-retention headline, no
    verdict line). Passing the recipe's actual GateThresholds selects the
    correct headline: a point gate (min_mean_retention set) keeps the
    mean-retention headline; a significance gate (min_mean_retention is None)
    leads with the "Gate: ..." verdict line instead, since the mean is not a
    criterion in that mode."""
    has_stderr = any(d.combined_stderr is not None for d in result.accuracy_deltas)
    if has_stderr:
        lines = [
            "| task | metric | baseline | quantized | delta | +/-stderr | retention |",
            "|------|--------|---------:|----------:|------:|----------:|----------:|",
        ]
    else:
        lines = [
            "| task | metric | baseline | quantized | delta | retention |",
            "|------|--------|---------:|----------:|------:|----------:|",
        ]
    rows = list(result.accuracy_deltas)
    if result.perplexity_delta is not None:
        rows.append(result.perplexity_delta)
    for d in rows:
        marker = " *" if d.significant else ""
        if has_stderr:
            se = f"{d.combined_stderr:.4f}" if d.combined_stderr is not None else "-"
            lines.append(
                f"| {d.task} | {d.metric} | {d.baseline:.4f} | {d.quantized:.4f} "
                f"| {d.delta:+.4f}{marker} | {se} | {d.retention:.4f} |")
        else:
            lines.append(
                f"| {d.task} | {d.metric} | {d.baseline:.4f} | {d.quantized:.4f} "
                f"| {d.delta:+.4f} | {d.retention:.4f} |")
    verdict = "PASS" if result.passed else "FAIL"
    lines.append("")
    # #2: the mean-retention headline is only meaningful when mean retention is an
    # actual gate criterion (point gates). For a significance gate (min_mean_retention
    # is None) the mean is "noise wearing a suit" - lead with the gate verdict instead.
    show_mean = thresholds is None or thresholds.min_mean_retention is not None
    if show_mean:
        lines.append(f"Mean accuracy retention: {result.mean_retention:.4f} - gate: {verdict}")
    else:
        k = thresholds.k_stderr
        lines.append(
            f"Gate: {verdict} - no task regressed beyond k={k:g} combined stderr"
            if result.passed and k is not None else f"Gate: {verdict}")
    if has_stderr:
        lines.append("(* = statistically significant regression at the recipe's k)")
        # #5: make an all-clear explicit so the legend's absence of stars reads as a
        # deliberate result, not an omission.
        if not any(d.significant for d in result.accuracy_deltas):
            lines.append("No task showed a statistically significant regression.")
    if result.reasons:
        lines.append("Reasons: " + "; ".join(result.reasons))
    return "\n".join(lines)
