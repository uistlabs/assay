import dataclasses

import pytest

from assay.config import DEFAULT_GATE
from assay.recipes import RECIPES, Recipe, Calib, Eval, get_recipe, validate_recipe


def test_two_real_recipes_registered():
    assert "qwen2_5_7b_instruct" in RECIPES
    assert "r1_distill_qwen_7b" in RECIPES


def test_get_recipe_unknown_lists_valid_slugs():
    with pytest.raises(KeyError, match="qwen2_5_7b_instruct"):
        get_recipe("nope")


def test_qwen_recipe_is_chat_mode_with_default_gate():
    r = get_recipe("qwen2_5_7b_instruct")
    assert r.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert r.eval.mode == "chat"
    assert r.gate_or_default == DEFAULT_GATE
    assert ("gsm8k", "exact_match,flexible-extract") in r.eval.accuracy_tasks
    assert r.perplexity_task_name == "wikitext"


def test_r1_recipe_reasoning_shape():
    r = get_recipe("r1_distill_qwen_7b")
    assert r.base_model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert r.eval.mode == "chat"
    assert r.eval.system_prompt is None
    # The checkpoint's own chat_template appends "<think>\n" on add_generation_prompt
    # (which lm-eval sets in chat mode), so the recipe must NOT set a prefix or it
    # would double the tag. Verified against the live chat_template on metal-prep.
    assert r.eval.prompt_prefix is None
    assert r.eval.gen_kwargs["temperature"] == 0.6
    assert r.eval.gen_kwargs["top_p"] == 0.95
    assert ("gpqa_diamond_cot_zeroshot", "exact_match,flexible-extract") in r.eval.accuracy_tasks


def test_accuracy_task_names_property():
    r = get_recipe("qwen2_5_7b_instruct")
    assert r.accuracy_task_names == tuple(t for t, _ in r.eval.accuracy_tasks)


def test_all_registered_recipes_validate():
    for r in RECIPES.values():
        validate_recipe(r)  # must not raise


def _valid_eval():
    return Eval(accuracy_tasks=(("mmlu", "acc,none"),), perplexity=("wikitext", "word_perplexity"),
                mode="chat", gen_kwargs=None, system_prompt=None, prompt_prefix=None)


def test_validate_rejects_empty_accuracy_battery():
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "s", 8, 128),
               Eval((), None, "chat", None, None, None), None, ("nvfp4",))
    with pytest.raises(ValueError, match="at least one accuracy task"):
        validate_recipe(r)


def test_validate_rejects_bad_mode():
    ev = _valid_eval().__class__(**{**_valid_eval().__dict__, "mode": "banana"})
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "s", 8, 128), ev, None, ("nvfp4",))
    with pytest.raises(ValueError, match="mode must be"):
        validate_recipe(r)


def test_validate_rejects_prompt_prefix_in_completion_mode():
    ev = Eval((("mmlu", "acc"),), None, "completion", None, None, "<think>\n")
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "s", 8, 128), ev, None, ("nvfp4",))
    with pytest.raises(ValueError, match="prompt_prefix.*chat"):
        validate_recipe(r)


def test_validate_rejects_malformed_metric_key():
    ev = Eval((("mmlu", ""),), None, "chat", None, None, None)
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "s", 8, 128), ev, None, ("nvfp4",))
    with pytest.raises(ValueError, match="metric"):
        validate_recipe(r)


def test_validate_rejects_accuracy_metric_without_filter_comma():
    ev = Eval((("mmlu", "acc"),), None, "chat", None, None, None)
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "s", 8, 128), ev, None, ("nvfp4",))
    with pytest.raises(ValueError, match="must be fully qualified"):
        validate_recipe(r)


def test_validate_rejects_empty_base_model():
    r = Recipe("x", "", "NVFP4A16", Calib("d", "s", 8, 128), _valid_eval(), None, ("nvfp4",))
    with pytest.raises(ValueError, match="base_model"):
        validate_recipe(r)


def test_validate_rejects_empty_quant_scheme():
    r = Recipe("x", "b", "", Calib("d", "s", 8, 128), _valid_eval(), None, ("nvfp4",))
    with pytest.raises(ValueError, match="quant_scheme"):
        validate_recipe(r)


def test_validate_rejects_empty_calib_dataset():
    r = Recipe("x", "b", "NVFP4A16", Calib("", "s", 8, 128), _valid_eval(), None, ("nvfp4",))
    with pytest.raises(ValueError, match="calib.dataset"):
        validate_recipe(r)


def test_validate_rejects_empty_calib_split():
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "", 8, 128), _valid_eval(), None, ("nvfp4",))
    with pytest.raises(ValueError, match="calib.split"):
        validate_recipe(r)


def _r1():
    return RECIPES["r1_distill_qwen_7b"]


def test_r1_minerva_uses_math_verify():
    metrics = dict(_r1().eval.accuracy_tasks)
    assert metrics["minerva_math500"] == "math_verify,none"


def test_r1_gate_is_significance_gated():
    g = _r1().gate_or_default
    assert g.k_stderr == 2.0
    assert g.min_mean_retention is None
    assert g.max_single_drop_pts is None
    assert g.max_ppl_increase == 0.03


def test_r1_repeats_targets_aime_only():
    assert _r1().eval.repeats == {"aime24_avg": 16, "aime25_avg": 16}


def test_repeats_empty_by_default_validates():
    validate_recipe(RECIPES["qwen2_5_7b_instruct"])  # no repeats -> fine


def test_repeats_rejects_unknown_task():
    r = _r1()
    bad = dataclasses.replace(r, eval=dataclasses.replace(r.eval, repeats={"not_a_task": 4}))
    with pytest.raises(ValueError, match="repeats"):
        validate_recipe(bad)


def test_repeats_rejects_non_positive_k():
    r = _r1()
    bad = dataclasses.replace(r, eval=dataclasses.replace(r.eval, repeats={"aime24_avg": 0}))
    with pytest.raises(ValueError, match="int >= 1"):
        validate_recipe(bad)


def test_r1_recipe_uses_avgk_aime():
    from assay.recipes import get_recipe, validate_recipe
    r = get_recipe("r1_distill_qwen_7b")
    tasks = dict(r.eval.accuracy_tasks)
    assert "aime24_avg" in tasks and tasks["aime24_avg"] == "exact_match,avg"
    assert "aime25_avg" in tasks and tasks["aime25_avg"] == "exact_match,avg"
    assert "aime24" not in tasks and "aime25" not in tasks  # single-sample tasks gone
    assert r.eval.repeats == {"aime24_avg": 16, "aime25_avg": 16}
    validate_recipe(r)  # repeats keys must be accuracy tasks; must not raise


def test_validate_rejects_all_none_gate():
    # An EXPLICIT all-None GateThresholds is truthy, so gate_or_default does NOT
    # substitute DEFAULT_GATE; evaluate_gate then fires zero criteria and PASSES any
    # quant -> publish + DOI a false certification. gate=None (the DEFAULT_GATE path)
    # is fine; a criteria-free explicit gate must be rejected before any paid work.
    from assay.config import GateThresholds
    empty = GateThresholds(min_mean_retention=None, max_single_drop_pts=None,
                           max_ppl_increase=None, k_stderr=None)
    r = Recipe("x", "b", "NVFP4A16", Calib("d", "s", 8, 128), _valid_eval(), empty, ("nvfp4",))
    with pytest.raises(ValueError, match="gate"):
        validate_recipe(r)
