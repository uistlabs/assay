from __future__ import annotations

from dataclasses import dataclass

from assay.config import MAX_PPL_INCREASE, MAX_SINGLE_DROP_PTS, MIN_MEAN_RETENTION


@dataclass(frozen=True)
class TaskDelta:
    task: str
    metric: str
    baseline: float
    quantized: float
    delta: float       # quantized - baseline
    retention: float   # quantized / baseline (guard against zero baseline)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    accuracy_deltas: tuple[TaskDelta, ...]
    perplexity_delta: TaskDelta
    mean_retention: float


def _delta(task, metric, base, quant) -> TaskDelta:
    retention = quant / base if base else 0.0
    return TaskDelta(task, metric, base, quant, quant - base, retention)


def evaluate_gate(baseline, quantized, accuracy_tasks, perplexity_task) -> GateResult:
    """Compare baseline vs quantized results; return pass/fail with reasons.

    Each results dict maps task -> {"metric": name, "value": v} (see
    evaluate.parse_results). The metric name is carried through per-task so
    the delta table/model card label each row with the real lm-eval metric
    (e.g. gsm8k | exact_match,strict-match) rather than a hardcoded "acc"."""
    if not accuracy_tasks:
        raise ValueError(
            "evaluate_gate requires at least one accuracy task "
            "(accuracy_tasks was empty; mean retention is undefined with no tasks)"
        )
    acc_deltas = tuple(
        _delta(
            t, baseline[t]["metric"], baseline[t]["value"], quantized[t]["value"]
        )
        for t in accuracy_tasks
    )
    ppl = _delta(
        perplexity_task,
        baseline[perplexity_task]["metric"],
        baseline[perplexity_task]["value"],
        quantized[perplexity_task]["value"],
    )

    mean_retention = sum(d.retention for d in acc_deltas) / len(acc_deltas)
    ppl_increase = (ppl.quantized - ppl.baseline) / ppl.baseline if ppl.baseline else 1.0

    reasons: list[str] = []
    if mean_retention < MIN_MEAN_RETENTION:
        reasons.append(
            f"mean retention {mean_retention:.4f} < {MIN_MEAN_RETENTION}"
        )
    for d in acc_deltas:
        # delta is negative on a drop; a drop beyond the allowance fails.
        # Deliberate FP-tolerance fix (deviation from a naive `> MAX_SINGLE_DROP_PTS`
        # compare, documented per workspace convention): binary floats can't represent
        # 0.90 - 0.88 exactly, so a decimal-exact 2.0-pt drop can compute as
        # 2.0000000000000018 and wrongly fail a spot-on boundary. Round off the
        # representation error before comparing so only a genuine >2.0 drop fails.
        drop_pts = -d.delta * 100.0
        if round(drop_pts, 9) > MAX_SINGLE_DROP_PTS:
            reasons.append(
                f"{d.task} dropped {drop_pts:.2f} pts > {MAX_SINGLE_DROP_PTS}"
            )
    if ppl_increase > MAX_PPL_INCREASE:
        reasons.append(
            f"perplexity increased {ppl_increase * 100:.2f}% > {MAX_PPL_INCREASE * 100:.2f}%"
        )

    return GateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        accuracy_deltas=acc_deltas,
        perplexity_delta=ppl,
        mean_retention=mean_retention,
    )


def render_delta_table(result: GateResult) -> str:
    """ASCII markdown table of the deltas -- for the model card and heartbeat."""
    lines = [
        "| task | metric | baseline | quantized | delta | retention |",
        "|------|--------|---------:|----------:|------:|----------:|",
    ]
    for d in (*result.accuracy_deltas, result.perplexity_delta):
        lines.append(
            f"| {d.task} | {d.metric} | {d.baseline:.4f} | {d.quantized:.4f} "
            f"| {d.delta:+.4f} | {d.retention:.4f} |"
        )
    verdict = "PASS" if result.passed else "FAIL"
    lines.append("")
    lines.append(f"Mean accuracy retention: {result.mean_retention:.4f} -- gate: {verdict}")
    if result.reasons:
        lines.append("Reasons: " + "; ".join(result.reasons))
    return "\n".join(lines)
