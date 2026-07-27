"""assay.verify - in-pod, offline verification that the STAGED weights are the model
the recipe certifies (F-015).

Why this exists: ASSAY_WEIGHTS_PATH used to be trusted raw. A mis-staged volume
quantizes and publishes the WRONG model under a correct-looking card, and the run
still counts as pristine, so it can mint a real DOI cert. config.json alone is not an
acceptable floor: the two live base models are architecturally identical, so a partial
re-stage into a dirty directory can yield a chimera that LOADS cleanly - and the gate
is purely relative, with no absolute-score floor to object.

Contract under test:
- small pass (config.json / index / tokenizer_config.json) runs on EVERY tier;
  the shard pass runs only when the run can mint a real cert (pristine).
- ASSAY_BASE_MODEL present -> both passes SKIP, loudly. Safe by construction: that
  var breaks pristine, so the invariant holds stated positively - no run that can
  publish a real cert runs on unverified weights.
- Any mismatch/missing pin is a HARD fail (SystemExit), never a demotion to dry-run:
  a demoted run burns paid GPU hours producing plausible artifacts about the wrong
  model, and a dry run still builds a card.
"""
import dataclasses
import hashlib
import os

import pytest

from assay.config import load_config, recipe_override_keys
from assay.verify import verify_weights_shards, verify_weights_small


def _env(**kw):
    return {"ASSAY_WEIGHTS_PATH": "/vol/weights",
            "ASSAY_CHECKPOINT_REPO": "org/M-NVFP4A16", **kw}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _staged(tmp_path, files: dict[str, bytes]) -> str:
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    return str(tmp_path)


def _cfg(tmp_path, pins: dict[str, str], *, pristine=True, revision="e" * 40):
    cfg = load_config(_env())
    recipe = dataclasses.replace(cfg.recipe, base_revision=revision if pins else "",
                                 base_files=pins)
    return dataclasses.replace(cfg, recipe=recipe, weights_path=str(tmp_path),
                               pristine=pristine)


def test_base_model_override_breaks_pristine():
    """Amendment 1's safety rests entirely on this coupling: the skip is safe ONLY
    because ASSAY_BASE_MODEL is a pristine-breaking override, so a skipped run can
    never mint a real cert. If this ever fails, the skip is a false-cert path."""
    assert "ASSAY_BASE_MODEL" in recipe_override_keys()


def test_base_model_override_skips_both_passes_loudly(tmp_path, capsys):
    # weights dir deliberately EMPTY: a skip must not touch the filesystem.
    cfg = _cfg(tmp_path, {"config.json": "a" * 64})
    env = _env(ASSAY_BASE_MODEL="other/Model")
    verify_weights_small(cfg, env)
    verify_weights_shards(cfg, env)
    out = capsys.readouterr().out
    assert out.count("SKIPP") == 2
    assert "ASSAY_BASE_MODEL" in out


def test_unpinned_recipe_hard_fails_small_pass(tmp_path):
    cfg = _cfg(tmp_path, {})
    with pytest.raises(SystemExit, match="pin_base_files"):
        verify_weights_small(cfg, _env())


def test_small_pass_ok_ignores_shards_and_extra_files(tmp_path, capsys):
    good = b'{"arch": "qwen"}'
    tok = b'{"chat_template": "x"}'
    staged = _staged(tmp_path, {
        "config.json": good,
        "tokenizer_config.json": tok,
        "leftover.bin": b"junk from a previous stage",  # unpinned -> inert
    })
    pins = {
        "config.json": _sha(good),
        "tokenizer_config.json": _sha(tok),
        # Shard pinned but NOT staged: the small pass must not read shards at all.
        "model-00001-of-00002.safetensors": "b" * 64,
    }
    verify_weights_small(_cfg(staged, pins), _env())
    assert "OK" in capsys.readouterr().out


def test_small_pass_mismatch_hard_fails_naming_the_file(tmp_path):
    staged = _staged(tmp_path, {"config.json": b"tampered"})
    pins = {"config.json": _sha(b"expected")}
    with pytest.raises(SystemExit, match="config.json"):
        verify_weights_small(_cfg(staged, pins), _env())


def test_small_pass_missing_file_hard_fails(tmp_path):
    pins = {"config.json": _sha(b"expected")}
    with pytest.raises(SystemExit, match="config.json"):
        verify_weights_small(_cfg(tmp_path, pins), _env())


def test_shard_pass_verifies_only_shards(tmp_path, capsys):
    shard = b"\x00" * 1024
    staged = _staged(tmp_path, {"model-00001-of-00002.safetensors": shard})
    pins = {
        # config pinned but NOT staged: the shard pass must leave small files to
        # the small pass (which already ran before the GPU preflight).
        "config.json": "a" * 64,
        "model-00001-of-00002.safetensors": _sha(shard),
    }
    verify_weights_shards(_cfg(staged, pins), _env())
    assert "OK" in capsys.readouterr().out


def test_shard_pass_mismatch_hard_fails(tmp_path):
    staged = _staged(tmp_path, {"model-00001-of-00002.safetensors": b"wrong bytes"})
    pins = {"model-00001-of-00002.safetensors": _sha(b"right bytes")}
    with pytest.raises(SystemExit, match="model-00001-of-00002.safetensors"):
        verify_weights_shards(_cfg(staged, pins), _env())


def test_shard_pass_missing_shard_hard_fails(tmp_path):
    # The partial-re-stage chimera: index+config right, a shard absent/stale.
    pins = {"model-00001-of-00002.safetensors": _sha(b"x")}
    with pytest.raises(SystemExit, match="model-00001-of-00002.safetensors"):
        verify_weights_shards(_cfg(tmp_path, pins), _env())


def test_shard_pass_skipped_on_non_pristine_tiers(tmp_path, capsys):
    # Shards absent AND mismatched pins: must not matter - a non-pristine run
    # cannot mint a real cert, and the multi-minute hash pass is priced for cert
    # runs only. The skip still says so in the log.
    cfg = _cfg(tmp_path, {"model-00001-of-00002.safetensors": "c" * 64},
               pristine=False)
    verify_weights_shards(cfg, _env())
    out = capsys.readouterr().out
    assert "skip" in out.lower()


def test_small_pass_runs_even_when_not_pristine(tmp_path):
    # "Verify the small files on every tier": a dev-tier run on wrong weights is
    # still hours of wrong-model burn.
    staged = _staged(tmp_path, {"config.json": b"tampered"})
    pins = {"config.json": _sha(b"expected")}
    with pytest.raises(SystemExit, match="config.json"):
        verify_weights_small(_cfg(staged, pins, pristine=False), _env())
