import json
import math
from pathlib import Path

import pytest

from assay.evaluate import _stderr_key, parse_results

FIXTURE = Path(__file__).parent / "fixtures" / "lm_eval_sample.json"


def test_parse_normalizes_lm_eval_shape():
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(
        raw,
        (("gsm8k", "exact_match,strict-match"), ("arc_challenge", "acc")),
        ("wikitext", "word_perplexity"),
    )
    assert out["arc_challenge"] == {"metric": "acc", "value": 0.601, "stderr": 0.014}
    assert out["gsm8k"] == {"metric": "exact_match,strict-match", "value": 0.802, "stderr": 0.011}
    assert out["wikitext"] == {"metric": "word_perplexity", "value": 9.87, "stderr": None}


def test_parse_perplexity_none_omits_entry():
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(raw, (("hellaswag", "acc"),), None)
    assert "wikitext" not in out
    assert out["hellaswag"] == {"metric": "acc", "value": 0.42, "stderr": 0.015}


def test_parse_fully_qualified_filter_key_is_exact():
    # A multi-filter task: the pair names the filter, so _pick must not grab the
    # other filter's value. Fixture 'gsm8k' has strict-match=0.802, flexible=<other>.
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(raw, (("gsm8k", "exact_match,strict-match"),), None)
    assert out["gsm8k"]["value"] == 0.802


def test_parse_missing_task_raises():
    raw = {"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}
    try:
        parse_results(raw, (("gsm8k", "exact_match,strict-match"), ("arc_challenge", "acc")), None)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_parse_picks_bare_acc_when_acc_norm_key_precedes_it():
    # "hellaswag" fixture entry lists acc_norm,none BEFORE acc,none (reversed
    # from the arc_challenge entry above). A looser predicate such as
    # `key.startswith(name)` (missing the comma anchor) would match
    # "acc_norm,none" first - since "acc_norm,none".startswith("acc") is
    # True - and wrongly return 0.55 instead of the bare-acc value 0.42.
    # This pins _pick to the comma-anchored match regardless of key order.
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(raw, (("hellaswag", "acc"),), None)
    assert out["hellaswag"] == {"metric": "acc", "value": 0.42, "stderr": 0.015}


def test_parse_task_present_metric_absent_raises_keyerror():
    # "arc_challenge" is present in results, but its metrics dict has no
    # "acc" (or "acc,<filter>") key at all - the requested metric is
    # genuinely absent. This must raise via _pick's own explicit
    # `raise KeyError(...)`, distinct from test_parse_missing_task_raises
    # above, which raises via the plain dict lookup `results[task]` when the
    # TASK itself is absent from results.
    raw = {"results": {"arc_challenge": {"exact_match,none": 0.5}}}
    try:
        parse_results(raw, (("arc_challenge", "acc"),), None)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_stderr_key_derivation():
    assert _stderr_key("exact_match,none") == "exact_match_stderr,none"
    assert _stderr_key("exact_match,flexible-extract") == "exact_match_stderr,flexible-extract"
    assert _stderr_key("word_perplexity") == "word_perplexity_stderr"


def test_parse_captures_stderr_when_present():
    raw = {"results": {
        "aime24": {"exact_match,none": 0.4667, "exact_match_stderr,none": 0.092},
        "wikitext": {"word_perplexity": 9.4, "word_perplexity_stderr": "N/A"},
    }}
    out = parse_results(raw, (("aime24", "exact_match,none"),), ("wikitext", "word_perplexity"))
    assert out["aime24"]["value"] == 0.4667
    assert out["aime24"]["stderr"] == 0.092
    assert out["wikitext"]["stderr"] is None    # lm-eval "N/A" -> None


def test_parse_stderr_none_when_absent():
    raw = {"results": {"mmlu": {"acc,none": 0.71}}}
    out = parse_results(raw, (("mmlu", "acc,none"),), None)
    assert out["mmlu"]["value"] == 0.71
    assert out["mmlu"]["stderr"] is None


def test_parse_stderr_none_when_present_but_null():
    # a stderr key that is present but JSON-null must normalize to None, not crash
    raw = {"results": {"aime24": {"exact_match,none": 0.4667, "exact_match_stderr,none": None}}}
    out = parse_results(raw, (("aime24", "exact_match,none"),), None)
    assert out["aime24"]["stderr"] is None


def test_parse_results_raises_on_non_finite_accuracy():
    raw = {"results": {"aime24_avg": {"exact_match,avg": float("nan")}}}
    with pytest.raises(ValueError, match=r"non-finite.*aime24_avg.*exact_match,avg"):
        parse_results(raw, (("aime24_avg", "exact_match,avg"),), None)


def test_parse_results_raises_on_non_finite_perplexity():
    raw = {"results": {"t": {"acc,none": 0.5},
                       "wikitext": {"word_perplexity,none": float("inf")}}}
    with pytest.raises(ValueError, match=r"non-finite.*wikitext.*word_perplexity"):
        parse_results(raw, (("t", "acc,none"),), ("wikitext", "word_perplexity"))
