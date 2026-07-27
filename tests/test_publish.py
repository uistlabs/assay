import dataclasses

from assay import __version__
from assay.config import load_config
from assay.gate import evaluate_gate
from assay.publish import build_model_card, publish_if_passed


ACC = ("gsm8k",)
PPL = "wikitext"


def _res(passed: bool):
    base = {
        "gsm8k": {"metric": "exact_match,strict-match", "value": 0.80},
        "wikitext": {"metric": "word_perplexity", "value": 10.0},
    }
    good = {
        "gsm8k": {"metric": "exact_match,strict-match", "value": 0.799},
        "wikitext": {"metric": "word_perplexity", "value": 10.05},
    }
    bad = {
        "gsm8k": {"metric": "exact_match,strict-match", "value": 0.60},
        "wikitext": {"metric": "word_perplexity", "value": 10.0},
    }
    return evaluate_gate(base, good if passed else bad, ACC, PPL)


def _r1_sig_res():
    """R1-recipe significance GateResult: aime24_avg carries a wide combined stderr
    (avg@16, n=30) and minerva_math500 a tight one, plus perplexity. Built through
    evaluate_gate (as test_gate.py does) using the R1 recipe's own gate so it never
    drifts from the real thresholds.

    Deliberately NOT described as "power-limited" vs "binding": the gate is conjunctive,
    so a wide threshold means low resolving power on that task, not a lesser vote."""
    from tests._pairing_helpers import paired_items
    cfg = load_config({"ASSAY_RECIPE": "r1_distill_qwen_7b",
                       "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/R1-NVFP4A16"})
    gate = cfg.recipe.gate_or_default
    # Paired per-item data (BITE 2); paired SEs chosen to equal the combined SEs the
    # old per-side numbers produced, so the card's threshold prose keeps its scale.
    ab, aq = paired_items(0.50, 0.49, 0.1272792, prefix="aime")
    mb, mq = paired_items(0.84, 0.835, 0.0282843, prefix="min")
    base = {
        "aime24_avg": {"metric": "exact_match,avg", "value": 0.50, "stderr": 0.09, "items": ab},
        "minerva_math500": {"metric": "math_verify,none", "value": 0.84, "stderr": 0.02, "items": mb},
        "wikitext": {"metric": "word_perplexity", "value": 10.0},
    }
    quant = {
        "aime24_avg": {"metric": "exact_match,avg", "value": 0.49, "stderr": 0.09, "items": aq},
        "minerva_math500": {"metric": "math_verify,none", "value": 0.835, "stderr": 0.02, "items": mq},
        "wikitext": {"metric": "word_perplexity", "value": 10.05},
    }
    result = evaluate_gate(base, quant, ("aime24_avg", "minerva_math500"), "wikitext",
                           thresholds=gate)
    return cfg, result


class FakeApi:
    def __init__(self):
        self.uploaded = False

    def upload_folder(self, **kwargs):
        self.uploaded = True


def test_model_card_has_table_and_license():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "| task" in card
    assert "apache-2.0" in card.lower()
    assert "Qwen/Qwen2.5-7B-Instruct" in card
    assert card.isascii()


def test_model_card_names_the_pinned_base_revision():
    """F-027: 'quantization of X' with no revision names a moving target - two certs
    issued months apart can describe different weights while reading identically.
    The card must pin the exact upstream snapshot the certificate describes."""
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights",
                       "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    rev = cfg.recipe.base_revision
    assert rev  # live recipe is pinned; the card must surface it
    card = build_model_card(cfg, _res(True))
    assert rev[:12] in card
    assert f"https://huggingface.co/{cfg.recipe.base_model}/tree/{rev}" in card


def test_model_card_omits_revision_line_when_unpinned():
    # An unpinned recipe (pins not yet generated) must not render an empty pin -
    # a blank revision would read as a broken card, not a missing pin.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights",
                       "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    recipe = dataclasses.replace(cfg.recipe, base_revision="", base_files={})
    cfg = dataclasses.replace(cfg, recipe=recipe)
    card = build_model_card(cfg, _res(True))
    assert "/tree/" not in card
    assert "@ ``" not in card


def test_card_has_citation_section():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
                       "ASSAY_PIPELINE_URL": "https://github.com/uistlabs/assay"})
    card = build_model_card(cfg, _res(True))
    assert "## Citation" in card
    # BibTeX key + howpublished are derived from the repo, base cite from the recipe.
    assert "@misc{uistlabs_model_nvfp4a16," in card
    assert "howpublished = {\\url{https://huggingface.co/myorg/Model-NVFP4A16}}" in card
    assert "Please also cite the base model" in card
    assert "https://github.com/uistlabs/assay" in card  # pipeline url flows into the note
    assert card.isascii()


def test_card_tags_come_from_recipe_not_string_surgery():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "nvfp4a16" in card
    assert "8-bit" not in card  # this is 4-bit weights; the wrong tag must be gone


def test_card_has_generated_hardware_section():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "Hardware" in card
    assert "Marlin" in card and "sm_89" in card and "Turing" in card  # measured floor + Turing excluded
    assert "Blackwell" in card  # states it is NOT required


def test_card_hardware_section_requires_blackwell_for_activation_quant_scheme():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4", "ASSAY_QUANT_SCHEME": "NVFP4"})
    card = build_model_card(cfg, _res(True))
    assert "Blackwell" in card
    assert "not require Blackwell" not in card
    assert "no Blackwell required" not in card
    assert "requires" in card and "Blackwell" in card  # stated as a requirement, not an exclusion
    assert "Marlin" not in card  # the weight-only kernel path does not apply here


def test_card_chat_methodology_does_not_overclaim_real_usage():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})  # default recipe = chat
    card = build_model_card(cfg, _res(True))
    assert "reflect real usage" not in card.lower()  # overclaim for loglikelihood MC rows
    assert "read the deltas, not the absolute" in card  # honest framing
    assert "both sides are evaluated with identical settings" in card


def test_card_methodology_is_chat_mode_aware():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})  # qwen recipe, mode=chat
    card = build_model_card(cfg, _res(True))
    assert "chat" in card.lower()
    assert "raw-completion" not in card.lower()  # the old caveat must be gone in chat mode


def test_card_has_provenance_stamp():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert __version__ in card


def test_card_is_ascii():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert build_model_card(cfg, _res(True)).isascii()


def test_card_uses_single_hyphen_not_double():
    # Card PROSE must use ' - ', never the old double-hyphen style (which reads as a
    # CLI option-separator; org style is single ' - '). Covers both shapes: the R1
    # significance card (the "Gate: PASS - ..." verdict line + the no-stderr
    # perplexity placeholder in render_delta_table) and the default chat card (mode_note
    # + prose sections). The real ` --quantization ` CLI flag is backtick-wrapped, so a
    # spaced check correctly leaves it alone.
    #
    # NOTE: the needle is built from parts on purpose. A literal here is a MATCHING
    # string, not prose, and a repo-wide prose sweep would silently rewrite it and
    # invert this assertion into "the card contains no ' - ' at all" - which is how
    # this test broke during the 2026-07-24 sweep.
    needle = " " + ("-" * 2) + " "
    r1_cfg, r1_res = _r1_sig_res()
    assert needle not in build_model_card(r1_cfg, r1_res)
    default_cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert needle not in build_model_card(default_cfg, _res(True))


def test_r1_card_renders_significance_cert_without_crash():
    # R1 gate has min_mean_retention=None + max_single_drop_pts=None - the cert section
    # must render the ACTIVE thresholds only (significance + ppl), never format a None.
    cfg = load_config({"ASSAY_RECIPE": "r1_distill_qwen_7b",
                       "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/R1-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "None" not in card              # no NoneType formatted into the card
    # F-032: the headline criteria bullet must state the PAIRED basis; the deleted
    # unpaired construction ("of the baseline and quantized scores") is pinned OUT so
    # the stale wording cannot return without failing here.
    assert "standard error of the per-item score differences" in card
    assert "of the baseline and quantized scores" not in card
    assert "k=2" in card
    assert "No single accuracy task down more than" not in card  # point-drop bar NOT claimed
    assert "Perplexity increase" in card   # ppl bar IS claimed (0.03 set)
    assert card.isascii()


def test_publishes_when_gate_passes(tmp_path):
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    api = FakeApi()
    published = publish_if_passed(cfg, str(tmp_path), _res(True), token="t", api=api,
                                  dry_run=False)
    assert published is True
    assert api.uploaded is True


def test_does_not_publish_when_gate_fails(tmp_path):
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    api = FakeApi()
    published = publish_if_passed(cfg, str(tmp_path), _res(False), token="t", api=api,
                                  dry_run=False)
    assert published is False
    assert api.uploaded is False


def test_dry_run_writes_card_but_skips_upload(tmp_path):
    # Dry-run mode: the card build (exercises real card generation) must
    # still happen to verify the card renders, but the upload must
    # never fire - a dry run is not a publish.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    api = FakeApi()
    published = publish_if_passed(cfg, str(tmp_path), _res(True), token="t", api=api, dry_run=True)
    assert published is False
    assert api.uploaded is False
    assert (tmp_path / "README.md").exists()  # card still built (exercises card gen)


def test_r1_card_is_battery_generic_and_significance_shaped():
    # #4: the noise caveat is battery-generic (no gsm8k - R1 has no gsm8k task), and a
    # significance-gated card leads with the verdict line, not the noise-mean headline.
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "gsm8k" not in card                                  # stale caveat gone
    assert "a finite benchmark set is a sample" in card         # battery-generic caveat
    assert "Gate: PASS" in card                                 # significance verdict via render
    # Regression guard for requirement #1 (threading `gate` into render_delta_table):
    # a significance-gated card must show ONLY the verdict line, never the noise-mean
    # headline. If this fails, the render_delta_table(result) call was not changed to
    # render_delta_table(result, gate) inside build_model_card.
    assert "Mean accuracy retention:" not in card
    assert card.isascii()


# --- BITE 1: card generator honesty pass -------------------------------------------
# The published card claimed one task carried "the certification's binding, low-variance
# signal". The gate is CONJUNCTIVE (gate.py - any significant task fails the run), so no
# task binds and none is decorative. These tests pin the replacement claims, which are
# checkable against the card's own stderr column rather than asking for trust.

def test_card_drops_the_false_binding_signal_claim():
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "binding, low-variance signal" not in card
    assert "power-limited" not in card
    # ...and does not deny that noise can fail a sound quant, which it can at k=2.
    assert "cannot fail a sound quant" not in card


def test_card_states_per_task_fail_threshold_from_measured_stderr():
    # k * paired SE, in points, per scored task. aime24_avg: paired SE 0.12728,
    # k=2 -> 25.5 pts. minerva: paired SE 0.028284 -> 5.7 pts.
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "`aime24_avg` (avg@16, the mean of 16 samples per item) 25.5 pts" in card
    assert "`minerva_math500` 5.7 pts" in card
    assert "Per-task fail thresholds (this run)" in card
    # It is the threshold on the MEASURED drop. "Minimum detectable regression" / "rules
    # out a regression larger than X" would imply detection POWER: in power-analysis
    # usage a detectable effect is (z_alpha+z_beta)*SE, whereas a TRUE regression of
    # exactly k*SE clears this threshold only about half the time.
    assert "fails a task only when its measured drop exceeds" in card
    assert "does not assert that no regression exists below that size" in card
    for banned in ("minimum detectable", "rules out a regression", "could have caught"):
        assert banned not in card.lower()


def test_card_discloses_the_paired_test_without_spinning_it():
    # BITE 2 (D10): the card records WHICH test certified the run. The old unpaired
    # sqrt-combination text is gone with the formula it described; the paired test is
    # stated as construction, not characterized as a virtue.
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "paired" in card
    assert "per-item score differences" in card
    assert "sqrt(se_baseline^2 + se_quantized^2)" not in card
    # Never describe the old unpaired test as "conservative": it was conservative
    # against a false FAIL but LENIENT against a false PASS - the direction a
    # certification reader cares about.
    for spin in ("conservative", "worst case"):
        assert spin not in card.lower()


def test_card_states_the_one_sided_upper_bound_per_task():
    # The actual non-inferiority certificate (D10): measured drop + k*SE per task,
    # meaningful now that the SE is paired. aime: (0.01 + 2*0.1272792)*100 = 26.5
    # pts; minerva: (0.005 + 2*0.0282843)*100 = 6.2 pts.
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "upper confidence bound" in card
    assert "26.5" in card
    assert "6.2" in card


def test_card_states_false_alarm_rate_at_one_sig_fig_and_no_family_wise_number():
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    # NOMINAL rate from a normal approximation over small-n metrics: one significant
    # figure, because 3 would imply a calibration we have not verified.
    assert "nominal false-alarm rate of about 2% per task" in card
    assert "2.3%" not in card
    # No family-wise figure: it needs across-task independence, which is false (one
    # checkpoint -> positively correlated), so any such number would be wrong.
    assert "tasks share one checkpoint and are not independent" in card
    assert "4.5%" not in card and "8.8%" not in card


def test_card_keeps_the_perplexity_backstop_as_an_independent_bar():
    # Once the card admits it could miss a 20-pt aime drop, the ppl bar is what stops
    # the certification reading hollow. Derived from the recipe (perplexity set +
    # max_ppl_increase), so no task taxonomy is needed to state it.
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "perplexity criterion is an independent hard bar" in card
    assert "no more than 3%" in card


def test_repeat_protocol_derives_from_recipe_and_never_guesses_single_draw():
    """MODULARITY: a recipe whose repeats differ must produce a correct card with NO edit
    to publish.py. And a task the recipe does not repeat gets NO protocol label at all:
    "single draw" is itself a guessed classification - the recipe cannot tell one sampled
    draw from a greedy or loglikelihood-scored task."""
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "single draw" not in card
    assert "deterministic" not in card.lower()

    ev = dataclasses.replace(cfg.recipe.eval, repeats={"minerva_math500": 4})
    swapped = build_model_card(
        dataclasses.replace(cfg, recipe=dataclasses.replace(cfg.recipe, eval=ev)), result)
    assert "`minerva_math500` (avg@4, the mean of 4 samples per item)" in swapped
    assert "`aime24_avg` 25.5 pts" in swapped   # no longer repeated -> no label
    assert "avg@16" not in swapped              # nothing remembers the old recipe


def test_card_states_run_level_generation_settings_from_gen_kwargs():
    cfg, result = _r1_sig_res()   # R1 pins temperature 0.6 / top_p 0.95
    card = build_model_card(cfg, result)
    assert "Run-level generation settings: temperature 0.6, top_p 0.95" in card
    assert "do_sample" not in card and "max_gen_toks" not in card  # not sampling settings
    # A recipe pinning nothing must say so, not invent settings.
    ev = dataclasses.replace(cfg.recipe.eval, gen_kwargs=None)
    bare = build_model_card(
        dataclasses.replace(cfg, recipe=dataclasses.replace(cfg.recipe, eval=ev)), result)
    assert "No run-level sampling overrides were set" in bare


def test_usage_snippet_sampling_comes_from_the_recipe():
    # The snippet hardcoded temperature=0.7, top_p=0.8 on EVERY card, contradicting the
    # R1 recipe's own required 0.6/0.95 - handing the reader settings the certification
    # did not use.
    cfg, result = _r1_sig_res()
    assert "SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256)" in \
        build_model_card(cfg, result)
    # Nothing pinned -> no invented recommendation, just the length cap.
    qwen = build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}), _res(True))
    assert "SamplingParams(max_tokens=256)" in qwen
    assert "temperature=0.7" not in qwen


def test_single_scored_task_card_does_not_name_it_twice():
    """The latent bug the old note carried: with ONE scored task, max() and min() return
    the SAME delta, so the card named one task as both power-limited and binding."""
    cfg = load_config({"ASSAY_RECIPE": "r1_distill_qwen_7b",
                       "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/R1-NVFP4A16"})
    gate = cfg.recipe.gate_or_default
    from tests._pairing_helpers import paired_items
    mb, mq = paired_items(0.84, 0.835, 0.0282843, prefix="min")
    base = {"minerva_math500": {"metric": "math_verify,none", "value": 0.84, "stderr": 0.02, "items": mb},
            "wikitext": {"metric": "word_perplexity", "value": 10.0}}
    quant = {"minerva_math500": {"metric": "math_verify,none", "value": 0.835, "stderr": 0.02, "items": mq},
             "wikitext": {"metric": "word_perplexity", "value": 10.05}}
    result = evaluate_gate(base, quant, ("minerva_math500",), "wikitext", thresholds=gate)
    card = build_model_card(cfg, result)
    assert card.count("`minerva_math500` 5.7 pts") == 1
    assert "power-limited" not in card
    assert card.isascii()


def test_card_drops_unsupported_regularization_claim():
    # "quantization acting as mild regularization" was an unsupported causal claim, and
    # the v0.4->v0.5 pair showed that positive delta flipped sign (i.e. it was noise).
    cfg, result = _r1_sig_res()
    for card in (build_model_card(cfg, result),
                 build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}),
                                  _res(True))):
        assert "regulariz" not in card.lower()
        assert "no measurable loss" in card
        # "small benchmark sets vary run to run" is false for a greedy/loglikelihood
        # battery on a fixed stack; the real mechanism is finite-item sampling.
        assert "a finite benchmark set is a sample" in card


def test_card_scopes_the_claim_to_the_delta_on_every_recipe():
    cfg, result = _r1_sig_res()
    for card in (build_model_card(cfg, result),
                 build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}),
                                  _res(True))):
        assert "the certified quantity is the delta" in card
        assert "same software stack" in card
        assert "stored baseline" in card   # the published twin of the cached-baseline rejection


def test_rerun_variance_claim_only_appears_when_the_recipe_samples():
    # On a greedy / loglikelihood battery a rerun on the same stack reproduces the score
    # exactly, so claiming run-to-run movement would invent noise. The stack-drift clause
    # is unconditional - wikitext (deterministic) moved 0.46% across v0.4.0 -> v0.5.0.
    sig = build_model_card(*_r1_sig_res())
    assert "rerunning it on the same stack" in sig
    assert "up to roughly the standard errors shown" in sig   # a BOUND, not an equality
    qwen = build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}), _res(True))
    assert "rerunning it on the same stack" not in qwen
    assert "across harness or library versions" in qwen       # stack clause survives


def test_card_does_not_overclaim_what_the_certification_covers():
    # The card's most consequential sentence, and it was false: the gate measures
    # accuracy deltas on the listed benchmarks plus a perplexity ratio. It certifies
    # NOTHING about bias, safety, or behavior at large.
    cfg, result = _r1_sig_res()
    for card in (build_model_card(cfg, result),
                 build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}),
                                  _res(True))):
        assert "does not add or remove bias" not in card
        assert "faithfully reproduces the base model" not in card
        assert "certifies exactly that" not in card
        assert "does not measure bias, safety" in card
        assert "not guaranteed to preserve what was not measured" in card


def test_card_does_not_assert_unmeasured_w4a4_accuracy_behavior():
    # assay has never measured W4A4 accuracy; the one in-house datum is a perplexity
    # rejection. On a card closing with "actual measured numbers, not vendor estimates",
    # the comparison must be attributed and the real datum used.
    card = build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}), _res(True))
    assert "widely reported" in card and "we have not measured that ourselves" in card
    assert "failed the perplexity bar at +12.55%" in card
    assert "accept more degradation" not in card   # unmeasured comparative claim


def test_scheme_overview_bullets_track_the_actual_scheme():
    # These bullets were unconditional prose about weight-only 4-bit weights, so a W4A4
    # recipe got a card calling it weight-only - the same latent-falsehood class as the
    # single-task max()/min() bug.
    w4a4 = build_model_card(
        load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4", "ASSAY_QUANT_SCHEME": "NVFP4"}),
        _res(True))
    assert "quantizes **activations as well as weights**" in w4a4
    assert "(weight-only)." not in w4a4
    assert "Why weight-only" not in w4a4
    wo = build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}), _res(True))
    assert "Why weight-only" in wo


def test_card_license_comes_from_the_recipe_not_a_hardcoded_constant():
    # A quantization is a derivative work. publish.py hardcoded "apache-2.0", which
    # misdeclared the R1 card: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B is MIT upstream
    # (verified against the Hub 2026-07-26).
    r1 = build_model_card(*_r1_sig_res())
    assert "license: mit" in r1
    assert "license: apache-2.0" not in r1
    qwen = build_model_card(load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4A16"}), _res(True))
    assert "license: apache-2.0" in qwen   # correct for Qwen2.5-7B-Instruct


def test_mode_note_does_not_hardcode_a_battery_shape():
    # "most visibly on the multiple-choice tasks" described tasks the R1 battery does
    # not contain - the exact class of hardcode the modularity rule bans.
    card = build_model_card(*_r1_sig_res())
    assert "multiple-choice" not in card
    assert "read the deltas, not the absolute values" in card


def test_point_gated_card_gets_no_significance_prose():
    # Qwen is point-gated (k_stderr None): its flag threshold IS max_single_drop_pts,
    # already stated in the certification criteria. No MDR block, no false-flag rate.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "flags a task only when" not in card
    assert "by chance" not in card
    assert "No single accuracy task down more than" in card   # the real bar, unchanged


def test_qwen_card_unchanged_keeps_correct_gsm8k_caveat():
    # Qwen is point-gated (k_stderr is None): the generic-wording change (#4) still
    # applies, but it must NOT get the significance-only power-note bullet (#3), and
    # its mean-retention headline (a real gate criterion here) is unchanged.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "gsm8k" in card                             # correct for Qwen (gsm8k IS its task)
    assert "power-limited" not in card                 # point gate: no significance power line
    assert "Mean accuracy retention:" in card
    assert card.isascii()


def test_dry_run_is_keyword_only_and_required():
    """F-028: `publish_if_passed` is the single publish chokepoint and its `dry_run`
    defaulted to False - the LIVE, irreversible action. The production caller binds it
    explicitly today, so there is no active harm, but a future direct caller would get
    a real upload by omission. Publish-integrity bits are never defaulted, anywhere."""
    import inspect
    from assay.publish import publish_if_passed
    param = inspect.signature(publish_if_passed).parameters["dry_run"]
    assert param.default is inspect.Parameter.empty, "dry_run must be required"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, \
        "dry_run must be keyword-only so it can never be supplied positionally by accident"
