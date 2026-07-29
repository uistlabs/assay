"""The reader payload is the money-touching surface of F-040: a wrong field strands
the reader (no observability) or violates the credential/pinning rules. Mirror of
test_launch_guard's posture: pin the REST CPU shape that worked first try 2026-07-27."""
import base64

import pytest

from assay.runpod_ctl import READER_IMAGE, build_reader_payload

ENV = {
    "RUNPOD_API_KEY": "rk-test",
    "HF_TOKEN": "hf-test",
}


def _payload(**over):
    kw = dict(main_pod_id="mainpod1", volume_id="vol1",
              dataset="org/run-artifacts",
              loop_b64=base64.b64encode(b"loop").decode(),
              snapshot_b64=base64.b64encode(b"snap").decode(),
              env=ENV)
    kw.update(over)
    return build_reader_payload(**kw)


def test_cpu_shape_no_gpu_fields():
    p = _payload()
    assert p["computeType"] == "CPU"
    assert p["vcpuCount"] == 2
    assert "gpuTypeIds" not in p and "gpuCount" not in p and "gpuTypeId" not in p
    assert p["cloudType"] == "SECURE"
    assert p["containerDiskInGb"] == 10
    assert p["networkVolumeId"] == "vol1"
    assert p["volumeMountPath"] == "/runpod-volume"
    assert p["dataCenterIds"] == ["EUR-IS-1"]  # REGION default
    assert p["name"] == "assay-reader-mainpod1"


def test_default_image_is_digest_pinned():
    # Full hex match so a forgotten paste of the Step-1 digest cannot pass.
    import re
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", READER_IMAGE.split("@")[1])
    assert _payload()["imageName"] == READER_IMAGE


def test_rejects_unpinned_image_override():
    with pytest.raises(ValueError, match="digest-pinned"):
        _payload(env={**ENV, "ASSAY_READER_IMAGE": "python:3.12-slim"})


def test_bootstrap_embeds_both_files():
    p = _payload()
    cmd = p["dockerStartCmd"]
    assert cmd[:2] == ["bash", "-lc"]
    boot = cmd[2]
    assert base64.b64encode(b"loop").decode() in boot
    assert base64.b64encode(b"snap").decode() in boot
    assert "reader_loop.sh" in boot and "reader_snapshot.py" in boot


def test_env_carries_config_and_secrets():
    p = _payload(env={**ENV, "ASSAY_READER_INTERVAL": "60",
                      "ASSAY_READER_TTL": "3600",
                      "ASSAY_ARTIFACTS_DIR": "/runpod-volume/custom"})
    e = p["env"]
    assert e["RUNPOD_API_KEY"] == "rk-test" and e["HF_TOKEN"] == "hf-test"
    assert e["ASSAY_READER_MAIN_POD_ID"] == "mainpod1"
    assert e["ASSAY_ARTIFACTS_DATASET"] == "org/run-artifacts"
    assert e["ASSAY_ARTIFACTS_DIR"] == "/runpod-volume/custom"  # R-10a passthrough
    assert e["ASSAY_READER_INTERVAL"] == "60"
    assert e["ASSAY_READER_TTL"] == "3600"


def test_artifacts_dir_defaults_to_pod_entry_default():
    # Must equal pod_entry.sh's own default base or the reader derives a wrong dir.
    assert _payload()["env"]["ASSAY_ARTIFACTS_DIR"] == "/runpod-volume/assay-out/artifacts"


def test_numeric_env_defaults_uncorrupted_when_unset():
    # Pin the three un-overridden defaults as strings, matching build_reader_payload's
    # env.get(..., "<default>") calls (review finding companion: the validation added
    # below must not disturb the happy path).
    e = _payload()["env"]
    assert e["ASSAY_READER_INTERVAL"] == "600"
    assert e["ASSAY_READER_TTL"] == "86400"
    assert e["ASSAY_READER_BOOT_ESCALATE_MIN"] == "30"


def test_rejects_non_numeric_interval():
    with pytest.raises(ValueError, match="ASSAY_READER_INTERVAL"):
        _payload(env={**ENV, "ASSAY_READER_INTERVAL": "24h"})


def test_rejects_non_numeric_ttl():
    with pytest.raises(ValueError, match="ASSAY_READER_TTL"):
        _payload(env={**ENV, "ASSAY_READER_TTL": "24h"})


def test_rejects_non_numeric_boot_escalate_min():
    with pytest.raises(ValueError, match="ASSAY_READER_BOOT_ESCALATE_MIN"):
        _payload(env={**ENV, "ASSAY_READER_BOOT_ESCALATE_MIN": "24h"})


def test_rejects_non_positive_numeric_env():
    # 0 or negative voids the TTL backstop / hammers HF just as badly as garbage text.
    with pytest.raises(ValueError, match="ASSAY_READER_TTL"):
        _payload(env={**ENV, "ASSAY_READER_TTL": "0"})
    with pytest.raises(ValueError, match="ASSAY_READER_INTERVAL"):
        _payload(env={**ENV, "ASSAY_READER_INTERVAL": "-5"})


def test_missing_secret_raises():
    with pytest.raises(Exception):
        _payload(env={"HF_TOKEN": "hf-test"})  # no RUNPOD_API_KEY
