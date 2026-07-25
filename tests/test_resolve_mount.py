"""resolve_mount rebases volume paths to wherever the weights actually mount.

Regression: assay is the first pod-based consumer of a RunPod network volume.
Pods can mount it at /workspace, not the /runpod-volume the config defaulted to -
so a hardcoded mount base stranded the job (weights not found -> fast fail ->
pod self-terminated ~6 min with no output)."""
from assay.config import RunConfig, resolve_mount
from assay.recipes import get_recipe

_BASE = dict(
    recipe=get_recipe("qwen2_5_7b_instruct"),
    checkpoint_repo="uistlabs/x",
    pipeline_url="",
    weights_path="/runpod-volume/qwen2.5-7b-instruct",
    artifacts_dir="/runpod-volume/assay-out/artifacts",
    output_dir="/runpod-volume/assay-out/checkpoint",
    heartbeat_path="/runpod-volume/assay-out/artifacts/heartbeat.log",
    gpu_mem_util=0.85,
)


def _cfg(**over):
    return RunConfig(**{**_BASE, **over})


def test_keeps_configured_paths_when_weights_present():
    cfg = _cfg()
    out = resolve_mount(cfg, exists=lambda p: p == cfg.weights_path)
    assert out is cfg  # unchanged


def test_rebases_all_volume_paths_to_workspace():
    cfg = _cfg()
    # weights are under /workspace, not the configured /runpod-volume
    out = resolve_mount(cfg, exists=lambda p: p == "/workspace/qwen2.5-7b-instruct")
    assert out.weights_path == "/workspace/qwen2.5-7b-instruct"
    assert out.artifacts_dir == "/workspace/assay-out/artifacts"
    assert out.output_dir == "/workspace/assay-out/checkpoint"
    assert out.heartbeat_path == "/workspace/assay-out/artifacts/heartbeat.log"
    # recipe/checkpoint_repo/gpu_mem_util are not paths - they pass through.
    assert out.recipe is cfg.recipe
    assert out.checkpoint_repo == cfg.checkpoint_repo
    assert out.gpu_mem_util == cfg.gpu_mem_util


def test_raises_when_weights_absent_everywhere():
    cfg = _cfg()
    try:
        resolve_mount(cfg, exists=lambda p: False)
    except FileNotFoundError as e:
        assert "qwen2.5-7b-instruct" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError when weights are nowhere")


def test_raises_for_custom_path_off_known_mounts():
    cfg = _cfg(weights_path="/data/qwen")
    try:
        resolve_mount(cfg, exists=lambda p: False)
    except FileNotFoundError as e:
        assert "ASSAY_WEIGHTS_PATH" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError for unknown-mount custom path")
