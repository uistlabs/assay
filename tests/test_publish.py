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
    """R1-recipe significance GateResult: aime24_avg carries the largest combined
    stderr (most power-limited, directional-only), minerva_math500 the smallest
    (binding, low-variance signal), plus perplexity. Built through evaluate_gate
    (as test_gate.py does) using the R1 recipe's own gate so it never drifts from
    the real thresholds."""
    cfg = load_config({"ASSAY_RECIPE": "r1_distill_qwen_7b",
                       "ASSAY_CHECKPOINT_REPO": "myorg/R1-NVFP4A16"})
    gate = cfg.recipe.gate_or_default
    base = {
        "aime24_avg": {"metric": "exact_match,avg", "value": 0.50, "stderr": 0.09},
        "minerva_math500": {"metric": "math_verify,none", "value": 0.84, "stderr": 0.02},
        "wikitext": {"metric": "word_perplexity", "value": 10.0},
    }
    quant = {
        "aime24_avg": {"metric": "exact_match,avg", "value": 0.49, "stderr": 0.09},
        "minerva_math500": {"metric": "math_verify,none", "value": 0.835, "stderr": 0.02},
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
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "| task" in card
    assert "apache-2.0" in card.lower()
    assert "Qwen/Qwen2.5-7B-Instruct" in card
    assert card.isascii()


def test_card_has_citation_section():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
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
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "nvfp4a16" in card
    assert "8-bit" not in card  # this is 4-bit weights; the wrong tag must be gone


def test_card_has_generated_hardware_section():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "Hardware" in card
    assert "Marlin" in card and "sm_89" in card and "Turing" in card  # measured floor + Turing excluded
    assert "Blackwell" in card  # states it is NOT required


def test_card_hardware_section_requires_blackwell_for_activation_quant_scheme():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/M-NVFP4", "ASSAY_QUANT_SCHEME": "NVFP4"})
    card = build_model_card(cfg, _res(True))
    assert "Blackwell" in card
    assert "not require Blackwell" not in card
    assert "no Blackwell required" not in card
    assert "requires" in card and "Blackwell" in card  # stated as a requirement, not an exclusion
    assert "Marlin" not in card  # the weight-only kernel path does not apply here


def test_card_chat_methodology_does_not_overclaim_real_usage():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})  # default recipe = chat
    card = build_model_card(cfg, _res(True))
    assert "reflect real usage" not in card.lower()  # overclaim for loglikelihood MC rows
    assert "read the deltas, not the absolute" in card  # honest framing
    assert "both sides are evaluated with identical settings" in card


def test_card_methodology_is_chat_mode_aware():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})  # qwen recipe, mode=chat
    card = build_model_card(cfg, _res(True))
    assert "chat" in card.lower()
    assert "raw-completion" not in card.lower()  # the old caveat must be gone in chat mode


def test_card_has_provenance_stamp():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert __version__ in card


def test_card_is_ascii():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
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
    default_cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert needle not in build_model_card(default_cfg, _res(True))


def test_r1_card_renders_significance_cert_without_crash():
    # R1 gate has min_mean_retention=None + max_single_drop_pts=None - the cert section
    # must render the ACTIVE thresholds only (significance + ppl), never format a None.
    cfg = load_config({"ASSAY_RECIPE": "r1_distill_qwen_7b",
                       "ASSAY_CHECKPOINT_REPO": "myorg/R1-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "None" not in card              # no NoneType formatted into the card
    assert "combined standard error" in card   # states the significance basis
    assert "k=2" in card
    assert "No single accuracy task down more than" not in card  # point-drop bar NOT claimed
    assert "Perplexity increase" in card   # ppl bar IS claimed (0.03 set)
    assert card.isascii()


def test_publishes_when_gate_passes(tmp_path):
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    api = FakeApi()
    published = publish_if_passed(cfg, str(tmp_path), _res(True), token="t", api=api)
    assert published is True
    assert api.uploaded is True


def test_does_not_publish_when_gate_fails(tmp_path):
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    api = FakeApi()
    published = publish_if_passed(cfg, str(tmp_path), _res(False), token="t", api=api)
    assert published is False
    assert api.uploaded is False


def test_dry_run_writes_card_but_skips_upload(tmp_path):
    # Dry-run mode: the card build (exercises real card generation) must
    # still happen to verify the card renders, but the upload must
    # never fire - a dry run is not a publish.
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    api = FakeApi()
    published = publish_if_passed(cfg, str(tmp_path), _res(True), token="t", api=api, dry_run=True)
    assert published is False
    assert api.uploaded is False
    assert (tmp_path / "README.md").exists()  # card still built (exercises card gen)


def test_r1_card_has_dynamic_power_line_and_no_stale_gsm8k():
    # #3/#4: significance-gated card names the actual highest/lowest-combined-stderr
    # tasks (power-limited vs binding) instead of a hardcoded, possibly-stale task
    # name, and the noise caveat is battery-generic (no gsm8k - R1 has no gsm8k task).
    cfg, result = _r1_sig_res()
    card = build_model_card(cfg, result)
    assert "gsm8k" not in card                                  # #4: stale caveat gone
    assert "small benchmark sets" in card                       # #4: battery-generic caveat
    assert "aime24_avg" in card and "power-limited" in card     # #3: names highest-se task
    assert "minerva_math500" in card                            # #3: names the binding task
    assert "Gate: PASS" in card                                 # significance verdict via render
    # Regression guard for requirement #1 (threading `gate` into render_delta_table):
    # a significance-gated card must show ONLY the verdict line, never the noise-mean
    # headline. If this fails, the render_delta_table(result) call was not changed to
    # render_delta_table(result, gate) inside build_model_card.
    assert "Mean accuracy retention:" not in card
    assert card.isascii()


def test_qwen_card_unchanged_keeps_correct_gsm8k_caveat():
    # Qwen is point-gated (k_stderr is None): the generic-wording change (#4) still
    # applies, but it must NOT get the significance-only power-note bullet (#3), and
    # its mean-retention headline (a real gate criterion here) is unchanged.
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    card = build_model_card(cfg, _res(True))
    assert "gsm8k" in card                             # correct for Qwen (gsm8k IS its task)
    assert "power-limited" not in card                 # point gate: no significance power line
    assert "Mean accuracy retention:" in card
    assert card.isascii()
