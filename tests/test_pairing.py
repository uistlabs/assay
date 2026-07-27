"""assay.pairing - per-item capture, pooling, and the paired SE (BITE 2, F-001).

Design under test: specs/2026-07-26-assay-paired-gating-design.md (D1-D8).

The fixture is a REAL lm-eval 0.4.12 payload (tiny model, CPU, mmlu + arc_easy,
limit=2, log_samples=True), trimmed to the keys the code consumes - not a synthetic
imitation, per D3. It carries the real group nesting (mmlu -> 4 subgroups -> 57
leaves), the real dual-metric samples (acc AND acc_norm under filter "none" - the
F-022 ambiguity), real doc_hash values, and a reported group score the pooled mean
must reproduce (D2).
"""
import json
import math
import pathlib

import pytest

from assay import pairing

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "lm_eval_samples_real.json"


def _raw():
    return json.loads(FIXTURE.read_text())


def _reduced_raw():
    """The raw dict as the parent sees it: reduced in-child, samples deleted (D5)."""
    raw = _raw()
    raw["assay_per_item"] = pairing.reduce_samples(raw)
    del raw["samples"]
    return raw


# --- reduce_samples (child side, D1/D4) --------------------------------------

def test_reduce_captures_both_metrics_distinctly():
    # F-022: arc_easy emits acc AND acc_norm under filter "none". Capturing one
    # while the gate bands the other is a silently wrong significance test, so the
    # schema must key by "<metric>,<filter>" and carry both.
    per = pairing.reduce_samples(_raw())
    arc = per["arc_easy"]
    assert set(arc) == {"acc,none", "acc_norm,none"}
    raw_samples = _raw()["samples"]["arc_easy"]
    for s in raw_samples:
        assert arc["acc,none"][str(s["doc_id"])]["v"] == float(s["acc"])
        assert arc["acc,none"][str(s["doc_id"])]["h"] == s["doc_hash"]


def test_reduce_doc_ids_are_strings():
    # D8 trap: after the _sanitize_raw/persist JSON round-trip, int doc_id keys
    # become strings. Normalize at capture so both sides of any join agree.
    per = pairing.reduce_samples(_raw())
    assert all(isinstance(k, str) for k in per["arc_easy"]["acc,none"])


def test_reduce_skips_non_scalar_metric_values():
    # D4: wikitext per-doc results are TUPLES (loglikelihood, words). A uniform
    # float() would crash the child after full paid generation. Scalars only;
    # non-scalars are excluded with zero plumbing.
    raw = {"samples": {"wikitext": [
        {"doc_id": 0, "doc_hash": "h0", "filter": "none",
         "metrics": ["word_perplexity"], "word_perplexity": (12.3, 45)},
    ]}}
    per = pairing.reduce_samples(raw)
    assert per["wikitext"] == {}


def test_reduce_averages_repeats_per_doc():
    # avg@K (aime): K sample entries per doc_id; the per-item value is the mean of
    # the K binaries (design "Verified sound" section).
    raw = {"samples": {"aime24_avg": [
        {"doc_id": 0, "doc_hash": "h0", "filter": "avg", "metrics": ["exact_match"], "exact_match": 1.0},
        {"doc_id": 0, "doc_hash": "h0", "filter": "avg", "metrics": ["exact_match"], "exact_match": 0.0},
        {"doc_id": 1, "doc_hash": "h1", "filter": "avg", "metrics": ["exact_match"], "exact_match": 1.0},
        {"doc_id": 1, "doc_hash": "h1", "filter": "avg", "metrics": ["exact_match"], "exact_match": 1.0},
    ]}}
    per = pairing.reduce_samples(raw)
    assert per["aime24_avg"]["exact_match,avg"] == {
        "0": {"v": 0.5, "h": "h0"}, "1": {"v": 1.0, "h": "h1"}}


def test_reduce_rejects_doc_hash_conflict_within_a_doc():
    # Two entries claiming the same doc_id with different content is corrupt input;
    # averaging over it would hide a real join hazard.
    raw = {"samples": {"t": [
        {"doc_id": 0, "doc_hash": "AAA", "filter": "avg", "metrics": ["m"], "m": 1.0},
        {"doc_id": 0, "doc_hash": "BBB", "filter": "avg", "metrics": ["m"], "m": 0.0},
    ]}}
    with pytest.raises(ValueError, match="doc_hash"):
        pairing.reduce_samples(raw)


def test_reduce_requires_doc_hash():
    # F-033: a harness bump that drops/renames doc_hash must fail loud. The old
    # `.get("doc_hash", "")` default degraded the D8 cross-side identical-inputs
    # check to "" == "" - always true, guard dead, no signal.
    raw = {"samples": {"t": [
        {"doc_id": 0, "filter": "none", "metrics": ["m"], "m": 1.0},
    ]}}
    with pytest.raises(ValueError, match="doc_hash"):
        pairing.reduce_samples(raw)


def test_reduce_requires_the_metrics_declaration():
    # The per-sample `metrics` list is the harness's own statement of which keys are
    # metric values. Its absence means the samples schema moved under us (the BITE 3
    # shape hazard) - fail loud, never guess from key names.
    raw = {"samples": {"t": [{"doc_id": 0, "doc_hash": "h", "filter": "none", "m": 1.0}]}}
    with pytest.raises((ValueError, KeyError), match="metrics"):
        pairing.reduce_samples(raw)


# --- collect_items (parent side, D2/D3/D8) -----------------------------------

def test_collect_plain_task():
    items = pairing.collect_items(_reduced_raw(), "arc_easy", "acc,none")
    assert set(items) == {"arc_easy:0", "arc_easy:1"}


def test_collect_group_pools_leaves_and_reconciles():
    # D3 recursion over the REAL nesting + D2: the pooled mean must reproduce the
    # reported group score (weight_by_size aggregation makes them identical up to
    # FP path differences).
    raw = _reduced_raw()
    items = pairing.collect_items(raw, "mmlu", "acc,none")
    assert len(items) == 114  # 57 leaves x limit=2
    pooled = sum(rec["v"] for rec in items.values()) / len(items)
    assert math.isclose(pooled, raw["results"]["mmlu"]["acc,none"],
                        rel_tol=1e-9, abs_tol=1e-12)
    # namespaced keys: leaf task + doc_id, so doc 0 of two leaves cannot collide
    assert "mmlu_abstract_algebra:0" in items


def test_collect_subgroup_recursion():
    # Resolution must work from ANY level, not just the top (mmlu_stem is itself a
    # group of 19 leaves in this harness version).
    raw = _reduced_raw()
    items = pairing.collect_items(raw, "mmlu_stem", "acc,none")
    assert len(items) == 38


def test_collect_reconciliation_catches_wrong_data():
    # The load-bearing control (D2): tamper ONE captured value and the pooled mean
    # no longer matches the reported aggregate - wrong-metric capture, wrong filter,
    # dropped docs, and ordering bugs all surface here.
    raw = _reduced_raw()
    key = next(iter(raw["assay_per_item"]["arc_easy"]["acc,none"]))
    raw["assay_per_item"]["arc_easy"]["acc,none"][key]["v"] += 0.5
    with pytest.raises(ValueError, match="reconcil"):
        pairing.collect_items(raw, "arc_easy", "acc,none")


def test_collect_count_assert_catches_dropped_docs():
    raw = _reduced_raw()
    del raw["assay_per_item"]["arc_easy"]["acc,none"]["0"]
    with pytest.raises(ValueError, match="n-samples"):
        pairing.collect_items(raw, "arc_easy", "acc,none")


def test_collect_missing_metric_key_names_what_exists():
    raw = _reduced_raw()
    with pytest.raises(ValueError, match="exact_match,avg"):
        pairing.collect_items(raw, "arc_easy", "exact_match,avg")


def test_collect_missing_payload_fails_loud_with_child_error():
    # D4: on a child-side reduction failure the eval persists WITHOUT per-item data
    # plus an error marker; the gate path must fail loud and surface that marker,
    # never silently fall back to the unpaired test.
    raw = _reduced_raw()
    del raw["assay_per_item"]
    raw["assay_per_item_error"] = "Traceback: boom in reduce"
    with pytest.raises(ValueError, match="boom in reduce"):
        pairing.collect_items(raw, "arc_easy", "acc,none")


# --- paired_stderr (gate side, D6/D8) ----------------------------------------

def _items(vals, hashes=None):
    return {str(i): {"v": float(v), "h": (hashes or {}).get(i, f"h{i}")}
            for i, v in enumerate(vals)}


def test_paired_stderr_matches_sem_by_hand():
    # deltas = [0, 0, -1, 0]: mean -0.25, var(ddof=1) 0.25, sd 0.5, sem 0.25.
    se = pairing.paired_stderr(_items([1, 0, 1, 0]), _items([1, 0, 0, 0]), task="t")
    assert se == pytest.approx(0.25)


def test_paired_stderr_zero_on_identical_sides():
    # D6 degenerate-safe: all-zero deltas give exactly 0.0, deterministically -
    # no RNG anywhere in a certification decision.
    se = pairing.paired_stderr(_items([1, 0, 1]), _items([1, 0, 1]), task="t")
    assert se == 0.0


def test_paired_stderr_refuses_differing_doc_sets():
    # D8: never silently intersect - a missing doc on one side means the join is
    # not the identical item set the whole method rests on.
    with pytest.raises(ValueError, match="doc"):
        pairing.paired_stderr(_items([1, 0, 1]), _items([1, 0]), task="t")


def test_paired_stderr_refuses_doc_hash_mismatch():
    # D8: covers an unpinned dataset re-fetched between sides with changed content
    # under identical ids (e.g. the aime24 HF dataset is not revision-pinned).
    base = _items([1, 0])
    quant = _items([1, 0], hashes={1: "DIFFERENT"})
    with pytest.raises(ValueError, match="doc_hash"):
        pairing.paired_stderr(base, quant, task="t")
