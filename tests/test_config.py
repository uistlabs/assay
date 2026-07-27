import pytest

from assay.config import (
    DEFAULT_GATE,
    GateThresholds,
    RunConfig,  # noqa: F401 - imported to assert the public name exists
    load_config,
    require_secret,
)


def _env(**kw):
    return {"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16", **kw}


def test_default_recipe_is_qwen():
    cfg = load_config(_env())
    assert cfg.recipe.slug == "qwen2_5_7b_instruct"
    assert cfg.recipe.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.recipe.calib.num_samples == 512
    assert cfg.recipe.perplexity_task_name == "wikitext"


def test_select_recipe_by_env():
    cfg = load_config(_env(ASSAY_RECIPE="r1_distill_qwen_7b"))
    assert cfg.recipe.slug == "r1_distill_qwen_7b"


def test_unknown_recipe_raises():
    with pytest.raises(KeyError, match="unknown recipe"):
        load_config(_env(ASSAY_RECIPE="nope"))


def test_scalar_env_overrides_recipe_field():
    cfg = load_config(_env(ASSAY_NUM_CALIB="8", ASSAY_BASE_MODEL="foo/bar"))
    assert cfg.recipe.calib.num_samples == 8
    assert cfg.recipe.base_model == "foo/bar"


def test_quant_scheme_default_and_override():
    # Default is the weight-only W4A16 variant after the W4A4 gate fail.
    assert load_config(_env()).recipe.quant_scheme == "NVFP4A16"
    assert load_config(_env(ASSAY_QUANT_SCHEME="NVFP4")).recipe.quant_scheme == "NVFP4"


def test_checkpoint_repo_required():
    with pytest.raises(ValueError, match="ASSAY_CHECKPOINT_REPO"):
        load_config({})


def test_weights_path_required_no_model_default():
    """F-015 amendment 3: the old default was a QWEN path for EVERY recipe, so an R1
    run that forgot ASSAY_WEIGHTS_PATH silently pointed at the Qwen volume instead of
    failing. Required-with-no-default turns that into a $0 pre-spend failure and
    removes a hardcoded model name from config."""
    with pytest.raises(ValueError, match="ASSAY_WEIGHTS_PATH"):
        load_config({"ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"})


def test_checkpoint_repo_read_from_env():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/My-Model-NVFP4A16"})
    assert cfg.checkpoint_repo == "myorg/My-Model-NVFP4A16"


def test_gpu_mem_util_default_and_override():
    assert load_config(_env()).gpu_mem_util == 0.85
    assert load_config(_env(ASSAY_GPU_MEM_UTIL="0.70")).gpu_mem_util == 0.70


def test_default_gate_thresholds():
    assert DEFAULT_GATE == GateThresholds(0.99, 2.0, 0.03)  # 0.03 loosened from 0.01 on metal


def test_config_never_holds_secrets():
    cfg = load_config(_env(RUNPOD_API_KEY="sekret", HF_TOKEN="sekret"))
    assert "sekret" not in repr(cfg)


def test_artifacts_dir_and_output_dir_are_separate():
    # I1/I2: ops artifacts (heartbeat, eval JSONs, delta table) must live OUTSIDE
    # the published checkpoint dir, so a publish (which uploads output_dir) can
    # never sweep them up. artifacts_dir and output_dir are SIBLINGS (neither
    # nests inside the other) so that publish_artifacts (which uploads
    # artifacts_dir wholesale to the private run-artifacts dataset) can never
    # sweep the multi-GB checkpoint into it either (SP1 whole-branch review fix).
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert cfg.artifacts_dir == "/runpod-volume/assay-out/artifacts"
    assert cfg.output_dir == "/runpod-volume/assay-out/checkpoint"
    assert cfg.heartbeat_path == "/runpod-volume/assay-out/artifacts/heartbeat.log"
    assert not cfg.heartbeat_path.startswith(cfg.output_dir)
    assert not cfg.output_dir.startswith(cfg.artifacts_dir + "/")
    assert not cfg.artifacts_dir.startswith(cfg.output_dir + "/")


def test_default_output_dir_and_artifacts_dir_are_siblings():
    # The structural invariant that prevents publish_artifacts from sweeping the
    # multi-GB checkpoint into the private run-artifacts upload: under the
    # DEFAULT config, output_dir and artifacts_dir must be siblings - neither
    # contains the other.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"})
    assert not cfg.output_dir.startswith(cfg.artifacts_dir + "/")
    assert not cfg.artifacts_dir.startswith(cfg.output_dir + "/")


def test_artifacts_dir_env_override():
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": "/tmp/ops",
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    assert cfg.artifacts_dir == "/tmp/ops"


def test_require_secret_present():
    assert require_secret({"HF_TOKEN": "abc"}, "HF_TOKEN") == "abc"


def test_require_secret_missing_raises():
    with pytest.raises(ValueError, match="HF_TOKEN is required"):
        require_secret({}, "HF_TOKEN")


# I2 guard: publish uploads output_dir wholesale to the public HF repo, so ops
# artifacts (heartbeat log, eval JSONs, delta table) must never live at or
# under output_dir. A stale ASSAY_OUTPUT_DIR reintroducing the old shared
# default must be rejected, not silently accepted.


def test_load_config_defaults_pass_the_i2_guard():
    # Safe layout: output_dir and artifacts_dir are siblings, so ops artifacts
    # (which live under artifacts_dir) are never inside output_dir, and the
    # checkpoint (in output_dir) is never inside artifacts_dir.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert cfg  # no raise


def test_output_dir_nested_in_artifacts_dir_raises():
    # The bidirectional half of the I2 guard (SP1 whole-branch review fix):
    # publish_artifacts uploads artifacts_dir wholesale to the PRIVATE
    # run-artifacts dataset, so a checkpoint nested inside artifacts_dir would
    # be swept into that upload every run, blowing the 300s timeout.
    with pytest.raises(ValueError, match="must not be nested inside"):
        load_config({
            "ASSAY_ARTIFACTS_DIR": "/vol/a",
            "ASSAY_OUTPUT_DIR": "/vol/a/checkpoint",
            "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        })


def test_stale_output_dir_equal_to_artifacts_dir_raises():
    # The exact regression this guard exists for: an operator's stale
    # ASSAY_OUTPUT_DIR pointed at the old shared default (== artifacts_dir).
    with pytest.raises(ValueError, match="published to the public HF repo"):
        load_config({
            "ASSAY_OUTPUT_DIR": "/runpod-volume/assay-out",
            "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        })


def test_output_dir_containing_heartbeat_path_raises():
    with pytest.raises(ValueError, match="published to the public HF repo"):
        load_config({
            "ASSAY_ARTIFACTS_DIR": "/tmp/ops",
            "ASSAY_HEARTBEAT": "/tmp/checkpoint/heartbeat.log",
            "ASSAY_OUTPUT_DIR": "/tmp/checkpoint",
            "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        })


def test_custom_disjoint_layout_does_not_raise():
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": "/tmp/ops",
        "ASSAY_HEARTBEAT": "/tmp/ops/heartbeat.log",
        "ASSAY_OUTPUT_DIR": "/tmp/checkpoint",
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    assert cfg.artifacts_dir == "/tmp/ops"
    assert cfg.output_dir == "/tmp/checkpoint"


def test_gate_thresholds_fields_are_optional():
    g = GateThresholds(min_mean_retention=None, max_single_drop_pts=None,
                       max_ppl_increase=0.03, k_stderr=2.0)
    assert g.min_mean_retention is None
    assert g.max_single_drop_pts is None
    assert g.max_ppl_increase == 0.03
    assert g.k_stderr == 2.0


def test_default_gate_unchanged_and_k_stderr_defaults_none():
    assert DEFAULT_GATE.min_mean_retention == 0.99
    assert DEFAULT_GATE.max_single_drop_pts == 2.0
    assert DEFAULT_GATE.max_ppl_increase == 0.03
    assert DEFAULT_GATE.k_stderr is None


def test_tier_defaults_to_cert():
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"})
    assert cfg.tier == "cert"
    assert cfg.eval_limit is None


def test_tier_dev_and_smoke_case_insensitive():
    base = {"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"}
    assert load_config({**base, "ASSAY_TIER": "dev"}).tier == "dev"
    assert load_config({**base, "ASSAY_TIER": "DEV"}).tier == "dev"
    assert load_config({**base, "ASSAY_TIER": "smoke"}).tier == "smoke"
    assert load_config({**base, "ASSAY_TIER": ""}).tier == "cert"  # empty -> full run


def test_tier_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="ASSAY_TIER"):
        load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", "ASSAY_TIER": "prod"})


def test_eval_limit_from_tier_and_override():
    base = {"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"}
    assert load_config({**base, "ASSAY_TIER": "dev"}).eval_limit == 50
    assert load_config({**base, "ASSAY_TIER": "smoke"}).eval_limit == 2
    assert load_config({**base, "ASSAY_TIER": "dev", "ASSAY_LIMIT": "10"}).eval_limit == 10


def test_eval_limit_one_refused():
    import pytest
    with pytest.raises(ValueError, match="limit == 1"):
        load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", "ASSAY_LIMIT": "1"})


def test_eval_limit_zero_refused():
    # Tightened alongside the pristine guard: an effective limit < 2 (not just the
    # literal 1) must be refused - 0/negative would evaluate nothing (or crash the
    # same significance-gate math as limit=1) while still matching cert tier.
    import pytest
    with pytest.raises(ValueError, match="limit == 0"):
        load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", "ASSAY_LIMIT": "0"})


def test_heartbeat_defaults_under_artifacts_dir():
    from assay.config import load_config
    cfg = load_config({
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        "ASSAY_ARTIFACTS_DIR": "/vol/assay-out/artifacts/pod-xyz",
    })
    assert cfg.heartbeat_path == "/vol/assay-out/artifacts/pod-xyz/heartbeat.log"


def test_explicit_heartbeat_still_wins():
    from assay.config import load_config
    cfg = load_config({
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        "ASSAY_ARTIFACTS_DIR": "/vol/art",
        "ASSAY_HEARTBEAT": "/custom/hb.log",
    })
    assert cfg.heartbeat_path == "/custom/hb.log"


def test_calib_override_breaks_pristine_even_on_cert_tier():
    # RED (the guard's reason to exist): a cert-tier run with a calib override reflects
    # a DIFFERENT quantization than the recipe - publishing it would mint a cert that
    # lies. Tier-gating alone (Task 1) misses this: tier is still "cert". Pristine catches it.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", "ASSAY_NUM_CALIB": "8"})
    assert cfg.tier == "cert"
    assert cfg.pristine is False


def test_pristine_true_only_for_unmodified_cert_run():
    assert load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"}).pristine is True


@pytest.mark.parametrize("var,val", [
    ("ASSAY_CALIB_DATASET", "wikitext"),
    ("ASSAY_CALIB_SPLIT", "train"),
    ("ASSAY_NUM_CALIB", "8"),
    ("ASSAY_MAX_SEQ_LEN", "1024"),
    ("ASSAY_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    ("ASSAY_QUANT_SCHEME", "NVFP4A16"),
    ("ASSAY_LIMIT", "10"),
    ("ASSAY_INJECT_STALL_AFTER", "60"),
])
def test_each_recognized_override_breaks_pristine(var, val):
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", var: val})
    assert cfg.pristine is False, var


def test_dev_and_smoke_tiers_are_not_pristine():
    base = {"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16"}
    assert load_config({**base, "ASSAY_TIER": "dev"}).pristine is False
    assert load_config({**base, "ASSAY_TIER": "smoke"}).pristine is False


def test_retired_assay_smoke_is_inert():
    # ASSAY_SMOKE is retired (MOVE 2): a stale one must NOT enable the old smoke tier -
    # it is simply ignored, so the run is a normal full cert (tier=cert, pristine=True).
    # Guards against a lingering reference silently resurrecting the old knob.
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", "ASSAY_SMOKE": "1"})
    assert cfg.tier == "cert"
    assert cfg.pristine is True


def test_every_recipe_override_key_breaks_pristine():
    """F-026: `_NONPRISTINE_VARS` used to be a hand-maintained duplicate of what
    `_apply_recipe_overrides` consumes, with the code comment admitting they had to be
    kept 'in lockstep'. Drift direction is FALSE-PASS: add an override, forget the
    other list, and an overridden run publishes as a pristine certificate.

    This is the structural guard - it derives the expectation from the override table
    itself, so a new override cannot be added without the pristine guard covering it.
    """
    from assay import config as cfgmod
    keys = cfgmod.recipe_override_keys()
    assert keys, "the override table must not be empty"
    for key in keys:
        cfg = load_config(_env(**{key: "1"}))
        assert cfg.pristine is False, (
            f"{key} overrides the recipe but did not break pristine - an overridden "
            "run could mint a real certificate")


def test_recipe_override_keys_match_what_is_actually_applied():
    """The table is the single source of truth: everything it names must actually
    change the resolved recipe, so a stale entry cannot linger and give false comfort."""
    from assay import config as cfgmod
    baseline = load_config(_env()).recipe
    for key in cfgmod.recipe_override_keys():
        overridden = load_config(_env(**{key: "7"})).recipe
        assert overridden != baseline, f"{key} is declared an override but changed nothing"


def test_pristine_true_only_on_a_clean_cert_run():
    assert load_config(_env()).pristine is True
    assert load_config(_env(ASSAY_TIER="dev")).pristine is False


def test_pristine_has_no_default_so_a_bypassing_caller_cannot_forget_it():
    """F-017: `pristine` gates whether a run may mint a REAL certificate. A default of
    True fails OPEN. A default of False fails safe but SILENTLY - a bypassing
    constructor that forgets the field quietly demotes a paid cert burn to dry-run,
    visible only in a log line nobody reads mid-incident. No default at all turns that
    into a TypeError at construction, in unit tests, at zero cost."""
    import dataclasses
    field = {f.name: f for f in dataclasses.fields(RunConfig)}["pristine"]
    assert field.default is dataclasses.MISSING, \
        "pristine must be required - a publish-integrity bit is never defaulted"
    assert field.default_factory is dataclasses.MISSING
