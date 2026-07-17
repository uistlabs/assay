import json
from pathlib import Path

from assay.evaluate import parse_results

FIXTURE = Path(__file__).parent / "fixtures" / "lm_eval_sample.json"


def test_parse_normalizes_lm_eval_shape():
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(raw, ("gsm8k", "arc_challenge"), "wikitext")
    assert out["arc_challenge"] == {"metric": "acc", "value": 0.601}
    assert out["wikitext"] == {"metric": "word_perplexity", "value": 9.87}


def test_parse_gsm8k_resolves_exact_match_strict_match():
    # C1 regression test: gsm8k has NO "acc" key in real lm-eval output --
    # only exact_match,strict-match and exact_match,flexible-extract. The
    # old code hardcoded _pick(results["gsm8k"], "acc"), which would raise
    # KeyError against this real-shaped fixture (no key equals "acc" or
    # starts with "acc,"). The fixed code resolves gsm8k's metric via
    # config.metric_for("gsm8k") == "exact_match,strict-match" first, so it
    # reads the correct key instead of ever looking for "acc".
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(raw, ("gsm8k",), "wikitext")
    assert out["gsm8k"] == {"metric": "exact_match,strict-match", "value": 0.802}


def test_parse_missing_task_raises():
    raw = {"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}
    try:
        parse_results(raw, ("gsm8k", "arc_challenge"), "wikitext")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_parse_picks_bare_acc_when_acc_norm_key_precedes_it():
    # "hellaswag" fixture entry lists acc_norm,none BEFORE acc,none (reversed
    # from the arc_challenge entry above). A looser predicate such as
    # `key.startswith(name)` (missing the comma anchor) would match
    # "acc_norm,none" first -- since "acc_norm,none".startswith("acc") is
    # True -- and wrongly return 0.55 instead of the bare-acc value 0.42.
    # This pins _pick to the comma-anchored match regardless of key order.
    raw = json.loads(FIXTURE.read_text())
    out = parse_results(raw, ("hellaswag",), "wikitext")
    assert out["hellaswag"] == {"metric": "acc", "value": 0.42}


def test_parse_task_present_metric_absent_raises_keyerror():
    # "arc_challenge" is present in results, but its metrics dict has no
    # "acc" (or "acc,<filter>") key at all -- its resolved metric (the
    # default "acc") is genuinely absent. This must raise via _pick's own
    # explicit `raise KeyError(...)`, distinct from
    # test_parse_missing_task_raises above, which raises via the plain dict
    # lookup `results[task]` when the TASK itself is absent from results.
    # (gsm8k can't be used for this case anymore -- with the C1 fix its
    # resolved metric, exact_match,strict-match, really is present.)
    raw = {"results": {"arc_challenge": {"exact_match,none": 0.5}}}
    try:
        parse_results(raw, ("arc_challenge",), "wikitext")
        assert False, "expected KeyError"
    except KeyError:
        pass
