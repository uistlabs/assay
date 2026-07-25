"""A cert-tier run that will publish for real should say so when the model card is
about to ship without a pipeline link. Dev/smoke tiers and dry-run stay quiet - the
warning must mark a real published artifact, not add noise to every test run.

Gated on `pristine` rather than on tier: job.py derives dry_run from it, so pristine
is exactly "this run publishes a REAL card". Deliberately a warning and not a hard
failure - a missing back-link degrades a card, it does not invalidate a cert."""

from assay.config import load_config


def _env(**kw):
    return {"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16", **kw}


def test_warns_when_pipeline_url_unset_on_live_cert_run(capsys):
    cfg = load_config(_env())
    assert cfg.pristine is True, "default env should be a live cert-tier run"
    assert cfg.pipeline_url == ""
    err = capsys.readouterr().err
    assert "ASSAY_PIPELINE_URL unset" in err
    assert "no pipeline link" in err


def test_silent_when_pipeline_url_is_set(capsys):
    cfg = load_config(_env(
        ASSAY_PIPELINE_URL="https://github.com/uistlabs/assay/tree/v0.5.1"))
    assert cfg.pipeline_url.endswith("/tree/v0.5.1")
    assert "ASSAY_PIPELINE_URL" not in capsys.readouterr().err


def test_silent_on_dev_tier(capsys):
    cfg = load_config(_env(ASSAY_TIER="dev"))
    assert cfg.pristine is False
    assert "ASSAY_PIPELINE_URL" not in capsys.readouterr().err
