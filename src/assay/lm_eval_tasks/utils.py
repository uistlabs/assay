"""avg@K scoring for the assay-owned AIME task variants (aime24_avg / aime25_avg).

Referenced from the sibling YAMLs via lm-eval's `!function utils.<name>` (lm-eval
adds the task dir to sys.path when loading the YAML, exactly as the stock aime task
imports its own utils). Kept ASCII-only: values here reach the pod log and card.

Design: stock aime `process_results` scores results[0] only, and lm-eval applies a
default `take_first` when no filter is set, so `repeats:K` scores 1 of K. We add a
pass-through filter (keep all K) + this averaging `process_results`, reusing AIME's
own exact_match extractor per sample. lm-eval's `mean` aggregation then yields avg@K
with a native across-doc stderr that shrinks with K - the whole point of v0.4.1.
"""
from __future__ import annotations

from typing import List


def passthrough(resps, docs):
    """Custom filter (filters/custom.py CustomFilter) that keeps EVERY sample.

    `resps` is one list-of-K per doc; the default `take_first` would collapse each to
    its first element. Return all K unchanged so the averaging process_results sees the
    full sample set. Materialized to a list so a generator is not consumed upstream."""
    return [list(r) for r in resps]


def average_scores(scores: List[int]) -> dict:
    """avg@K for one doc: fraction of the K samples that were correct. Empty -> 0.0
    (never divide by zero on a rented pod mid-eval)."""
    if not scores:
        return {"exact_match": 0.0}
    return {"exact_match": sum(scores) / len(scores)}


def _score_one(doc: dict, response: str) -> int:
    """Score a single sample with AIME's OWN extractor (reused, not reinvented): try
    $...$, prefer a \\boxed{} answer, then AIME's is_equiv normalization. Imported from
    the stock aime task so we stay byte-identical to the extractor proven on R1 metal."""
    from lm_eval.tasks.aime.utils import (  # noqa: PLC0415
        is_equiv, last_boxed_only_string, remove_boxed,
    )
    indices = [pos for pos, ch in enumerate(response) if ch == "$"]
    answer = response if len(indices) <= 1 else response[indices[0] + 1:indices[-1]]
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        try:
            content = remove_boxed(boxed)
            if content is not None:
                answer = content
        except (AssertionError, IndexError):
            pass
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])
    return 1 if is_equiv(answer, target) else 0


def _as_samples(results):
    """Normalize the filtered results object to a flat list of K response strings.
    lm-eval's real per-doc shape for a single-instance-per-doc task (AIME) is the
    nested [[s1, ..., sK]] - one filtered list per doc - confirmed against the
    harness's evaluator and covered by test_avg_process_results_harness_nested_shape;
    the flat form is tolerated defensively (e.g. for direct/unit-test callers)."""
    if len(results) == 1 and isinstance(results[0], (list, tuple)):
        return list(results[0])
    return list(results)


def avg_process_results(doc: dict, results) -> dict:
    """process_results override: score ALL K samples, return the per-doc mean."""
    samples = _as_samples(results)
    scores = [_score_one(doc, r) for r in samples]
    return average_scores(scores)
