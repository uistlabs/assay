import pytest

from assay.config import (
    Config,
    MIN_MEAN_RETENTION,
    MAX_SINGLE_DROP_PTS,
    MAX_PPL_INCREASE,
    load_config,
    require_secret,
)


def test_defaults_match_spec():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.calib_dataset == "HuggingFaceH4/ultrachat_200k"
    assert cfg.num_calibration_samples == 512
    assert cfg.max_seq_length == 2048
    assert "gsm8k" in cfg.accuracy_tasks
    assert cfg.perplexity_task == "wikitext"


def test_checkpoint_repo_is_required():
    with pytest.raises(ValueError, match="ASSAY_CHECKPOINT_REPO"):
        load_config({})


def test_checkpoint_repo_read_from_env():
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/My-Model-NVFP4A16"})
    assert cfg.checkpoint_repo == "myorg/My-Model-NVFP4A16"


def test_artifacts_dir_and_output_dir_are_separate():
    # I1/I2: ops artifacts (heartbeat, eval JSONs, delta table) must live OUTSIDE
    # the published checkpoint dir, so a publish (which uploads output_dir) can
    # never sweep them up.
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert cfg.artifacts_dir == "/runpod-volume/assay-out"
    assert cfg.output_dir == "/runpod-volume/assay-out/checkpoint"
    assert cfg.heartbeat_path == "/runpod-volume/assay-out/heartbeat.log"
    assert not cfg.heartbeat_path.startswith(cfg.output_dir)
    assert cfg.output_dir.startswith(cfg.artifacts_dir)


def test_artifacts_dir_env_override():
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": "/tmp/ops",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    assert cfg.artifacts_dir == "/tmp/ops"


def test_env_overrides():
    cfg = load_config({
        "ASSAY_BASE_MODEL": "foo/bar",
        "ASSAY_NUM_CALIB": "8",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    assert cfg.base_model == "foo/bar"
    assert cfg.num_calibration_samples == 8


def test_gpu_mem_util_default_and_override():
    assert load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"}).gpu_mem_util == 0.85
    assert load_config({
        "ASSAY_GPU_MEM_UTIL": "0.70",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    }).gpu_mem_util == 0.70


def test_quant_scheme_default_and_override():
    # Default is the weight-only W4A16 variant after the W4A4 gate fail.
    assert load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"}).quant_scheme == "NVFP4A16"
    assert load_config({
        "ASSAY_QUANT_SCHEME": "NVFP4",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    }).quant_scheme == "NVFP4"


def test_gate_thresholds():
    assert MIN_MEAN_RETENTION == 0.99
    assert MAX_SINGLE_DROP_PTS == 2.0
    assert MAX_PPL_INCREASE == 0.03  # loosened from 0.01 on metal evidence


def test_config_never_holds_secrets():
    cfg = load_config({
        "RUNPOD_API_KEY": "sekret",
        "HF_TOKEN": "sekret",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    assert "sekret" not in repr(cfg)


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
    # Safe layout: output_dir is a subdir of artifacts_dir, so ops artifacts
    # (which live in artifacts_dir/heartbeat_path) are never inside output_dir.
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert cfg  # no raise


def test_stale_output_dir_equal_to_artifacts_dir_raises():
    # The exact regression this guard exists for: an operator's stale
    # ASSAY_OUTPUT_DIR pointed at the old shared default (== artifacts_dir).
    with pytest.raises(ValueError, match="published to the public HF repo"):
        load_config({
            "ASSAY_OUTPUT_DIR": "/runpod-volume/assay-out",
            "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        })


def test_output_dir_containing_heartbeat_path_raises():
    with pytest.raises(ValueError, match="published to the public HF repo"):
        load_config({
            "ASSAY_ARTIFACTS_DIR": "/tmp/ops",
            "ASSAY_HEARTBEAT": "/tmp/checkpoint/heartbeat.log",
            "ASSAY_OUTPUT_DIR": "/tmp/checkpoint",
            "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        })


def test_custom_disjoint_layout_does_not_raise():
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": "/tmp/ops",
        "ASSAY_HEARTBEAT": "/tmp/ops/heartbeat.log",
        "ASSAY_OUTPUT_DIR": "/tmp/checkpoint",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    assert cfg.artifacts_dir == "/tmp/ops"
    assert cfg.output_dir == "/tmp/checkpoint"
