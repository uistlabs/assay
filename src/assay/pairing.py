"""Per-item score capture, pooling, and the paired standard error (BITE 2, F-001).

The defect this exists for: `gate.py` combined per-side stderrs with the
independent-samples formula, but baseline and quantized answer the IDENTICAL item
set, so they correlate through item difficulty. That covariance should cancel out
of the delta; unpaired it is counted twice, the k*SE band is too wide, and a real
regression PASSES - the false-PASS direction, in the mechanism we sell.

Three layers, matching where each must run (design points D1-D8 referenced below are
from the 2026-07-26 paired-gating design review):

- reduce_samples: IN THE EVAL CHILD, on the raw lm-eval payload, BEFORE
  _sanitize_raw and before anything crosses the Pipe. The full samples payload is
  the exact object that wedged a paid R1 burn for ~6h when serialized (F-003/F-020);
  the reduction to scalars is the entire licence for log_samples=True.
- collect_items: in the parent, per side. Resolves group tasks to their leaves
  recursively (D3), pools with leaf-namespaced keys, and runs the load-bearing
  reconciliation control (D2): the pooled mean must reproduce the harness's own
  reported aggregate, which catches wrong-metric capture, wrong filter, dropped
  docs, and ordering bugs in one identity.
- paired_stderr: at the gate. Joins the two sides on (leaf, doc_id), refuses any
  set or doc_hash mismatch (D8 - never intersect silently), and returns
  scipy.stats.sem of the per-item deltas (D6): deterministic, degenerate-safe,
  and exact for mean-aggregated metrics - no RNG in a certification decision.
"""
from __future__ import annotations

import math

_SCHEMA_KEY = "assay_per_item"
_SCHEMA_ERROR_KEY = "assay_per_item_error"


def reduce_samples(raw: dict) -> dict:
    """{task: {"<metric>,<filter>": {str(doc_id): {"v": float, "h": doc_hash}}}}.

    Keys match the recipe's own fully-qualified metric keys (D1) - the metric level
    is mandatory because multiple-choice tasks emit BOTH acc and acc_norm under one
    filter, and capturing one while the gate bands the other is a silently wrong
    significance test. Metric names come from each sample's own `metrics`
    declaration, never guessed from key names; a payload without it means the
    samples schema moved under us - fail loud (the caller's D4 wrapper turns that
    into an error marker rather than a crashed child).

    Scalars only (D4): wikitext's per-doc tuples are excluded with zero plumbing.
    doc_ids become strings here so both sides of every later join agree (the
    _sanitize_raw JSON round-trip stringifies int keys anyway). Repeats (avg@K):
    the real harness pre-averages - avg_process_results runs ONCE per doc, so one
    already-averaged entry arrives per doc_id (verified against lm-eval 0.4.12's
    evaluator; F-034). The K-entries-per-doc averaging below (sum/n) is defensive,
    for a harness that emits one entry per repeat; either shape yields the
    per-item avg@K score."""
    out: dict[str, dict] = {}
    for task, entries in (raw.get("samples") or {}).items():
        acc: dict[str, dict[str, dict]] = {}
        for s in entries:
            if "metrics" not in s:
                raise ValueError(
                    f"sample for task {task!r} has no `metrics` declaration - the "
                    "lm-eval samples schema changed; re-verify the capture layer "
                    "against the installed harness before trusting any pairing")
            if "doc_hash" not in s:
                # F-033: same posture as the missing-`metrics` raise above. A silent
                # "" default would make the cross-side identical-inputs check (D8)
                # vacuous ("" == "" always) with no signal that the guard died.
                raise ValueError(
                    f"sample for task {task!r} doc_id {s.get('doc_id')!r} has no "
                    "`doc_hash` - the lm-eval samples schema changed; re-verify the "
                    "capture layer against the installed harness before trusting "
                    "any pairing")
            doc_id = str(s["doc_id"])
            doc_hash = s["doc_hash"]
            filt = s.get("filter", "none")
            for metric in s["metrics"]:
                value = s.get(metric)
                # bool is an int subclass; tuples/lists (wikitext) excluded.
                if isinstance(value, bool):
                    value = float(value)
                elif not isinstance(value, (int, float)):
                    continue
                rec = acc.setdefault(f"{metric},{filt}", {}).setdefault(
                    doc_id, {"sum": 0.0, "n": 0, "h": doc_hash})
                if rec["h"] != doc_hash:
                    raise ValueError(
                        f"task {task!r} doc_id {doc_id}: doc_hash differs between "
                        f"repeat entries ({rec['h']!r} vs {doc_hash!r}) - corrupt "
                        "or mixed samples payload, refusing to average over it")
                rec["sum"] += float(value)
                rec["n"] += 1
        out[task] = {
            mkey: {doc: {"v": rec["sum"] / rec["n"], "h": rec["h"]}
                   for doc, rec in docs.items()}
            for mkey, docs in acc.items()
        }
    return out


def _resolve_leaves(raw: dict, task: str) -> list[str]:
    """D3: recurse group_subtasks until a name has no children. Samples exist only
    at leaves; a one-level lookup on a nested group pools an empty set."""
    children = (raw.get("group_subtasks") or {}).get(task)
    if not children:
        return [task]
    return [leaf for child in children for leaf in _resolve_leaves(raw, child)]


def collect_items(raw: dict, task: str, metric_key: str) -> dict[str, dict]:
    """Pooled per-item map {"<leaf>:<doc_id>": {"v", "h"}} for a gated task, with
    the D2 reconciliation and D8 count asserts. Raises ValueError on any violation
    - never returns partial data."""
    per_item = raw.get(_SCHEMA_KEY)
    if per_item is None:
        child_error = raw.get(_SCHEMA_ERROR_KEY, "no error marker recorded")
        raise ValueError(
            f"paired gating requires per-item data for task {task!r} but the eval "
            f"payload carries none. Child-side capture reported: {child_error}")
    items: dict[str, dict] = {}
    for leaf in _resolve_leaves(raw, task):
        leaf_metrics = per_item.get(leaf)
        if leaf_metrics is None:
            raise ValueError(
                f"task {task!r}: no per-item capture for leaf {leaf!r} - the eval "
                "ran without capture for this task, or the group resolved wrongly")
        docs = leaf_metrics.get(metric_key)
        if docs is None:
            raise ValueError(
                f"task {task!r} leaf {leaf!r}: metric {metric_key!r} was not "
                f"captured; captured keys: {sorted(leaf_metrics)}. The recipe's "
                "metric key and the harness's per-sample metrics disagree.")
        expected = (raw.get("n-samples") or {}).get(leaf, {}).get("effective")
        if expected is not None and len(docs) != expected:
            raise ValueError(
                f"task {task!r} leaf {leaf!r}: captured {len(docs)} docs but "
                f"n-samples says {expected} effective - docs were dropped or "
                "duplicated in capture; the pooled mean cannot be trusted")
        for doc_id, rec in docs.items():
            items[f"{leaf}:{doc_id}"] = rec
    if not items:
        raise ValueError(f"task {task!r}: per-item pooling produced zero items")
    # D2 - the load-bearing control. Every gated accuracy metric aggregates by
    # (size-weighted) mean, so the pooled mean must reproduce the reported score.
    # isclose, never ==: group scores re-weight at each nesting level, a different
    # FP path than a flat pool - equal in exact arithmetic, differing in last ulps,
    # and an exact-equality hard-fail would kill a paid cert run on FP noise.
    reported = (raw.get("results") or {}).get(task, {}).get(metric_key)
    if reported is None:
        raise ValueError(
            f"task {task!r}: reported aggregate {metric_key!r} missing from "
            "results - cannot reconcile the per-item capture against the harness")
    pooled = sum(rec["v"] for rec in items.values()) / len(items)
    if not math.isclose(pooled, reported, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"task {task!r} metric {metric_key!r}: per-item reconciliation failed - "
            f"pooled mean {pooled!r} != reported {reported!r} over {len(items)} "
            "items. The capture is not the data the gate bands (wrong metric/filter, "
            "dropped docs, or an aggregation change); refusing to compute a paired SE")
    return items


def paired_stderr(base_items: dict, quant_items: dict, *, task: str) -> float:
    """scipy.stats.sem of the per-item deltas (D6), after the D8 join checks."""
    from scipy.stats import sem

    if set(base_items) != set(quant_items):
        only_b = len(set(base_items) - set(quant_items))
        only_q = len(set(quant_items) - set(base_items))
        raise ValueError(
            f"task {task!r}: doc sets differ between sides ({only_b} only-baseline, "
            f"{only_q} only-quantized of {len(base_items)}/{len(quant_items)}) - "
            "the paired test requires the IDENTICAL item set; refusing to intersect "
            "silently")
    mismatched = [k for k in base_items if base_items[k]["h"] != quant_items[k]["h"]]
    if mismatched:
        raise ValueError(
            f"task {task!r}: doc_hash mismatch between sides for {len(mismatched)} "
            f"docs (first: {mismatched[0]!r}) - the two evals did not see identical "
            "documents (dataset re-fetched between sides?); the pairing is invalid")
    deltas = [quant_items[k]["v"] - base_items[k]["v"] for k in sorted(base_items)]
    return float(sem(deltas, ddof=1))
