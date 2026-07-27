def test_version_is_0_5_0():
    import assay
    assert assay.__version__ == "0.6.1"


def test_quantize_imports_and_recipe_constructs():
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
    assert recipe.scheme == "NVFP4"


def test_assay_modules_import():
    import assay.quantize  # noqa: F401
    import assay.evaluate  # noqa: F401
    import assay.publish   # noqa: F401
    import assay.job       # noqa: F401


def test_module_entrypoint_invokes_main():
    """`python -m assay.job` (exactly what pod_entry.sh runs) must actually call
    main(). Regression: job.py defined main() but lacked the `if __name__ ==
    "__main__"` guard, so -m imported the module and exited 0 without running
    anything - on the pod the EXIT-trap teardown then self-terminated it, so the
    whole job silently no-op'd. This asserts the entrypoint is wired.

    Env is crafted so main() -> load_config() trips the I2 output_dir guard
    immediately, before any GPU/network work: if main() runs, the process exits
    non-zero with that error; with no guard, -m exits 0 silently."""
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "ASSAY_ARTIFACTS_DIR": "/tmp/assay-entrypoint-test",
        "ASSAY_HEARTBEAT": "/tmp/assay-entrypoint-test/hb.log",
        # heartbeat/artifacts nested in output_dir -> load_config raises ValueError
        "ASSAY_OUTPUT_DIR": "/tmp/assay-entrypoint-test",
        # required since B7 - must be set so load_config reaches the I2
        # guard this test asserts, instead of raising on the missing key first.
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "assay.job"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0, (
        "python -m assay.job exited 0 - main() was never invoked "
        "(missing `if __name__ == \"__main__\"` guard in job.py)"
    )
    assert "must not contain" in proc.stderr, proc.stderr
