"""scripts/pin_base_files.py - the authoring tool for a recipe's base-model identity
pins (F-015 amendment 4). The pins move the trust boundary from the staged volume to
the reviewed git recipe; this tool exists so the INDEPENDENT source (the Hub at a
pinned commit) is the easy path, not a hand-transcribed one.

Network calls are a thin wrapper; everything decision-shaped (which files get pinned,
how the snippet is rendered, how drift is reported) is pure and tested here."""
import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pin_base_files.py"


def _load():
    """Import the script by path - it lives in scripts/, not in the assay package."""
    spec = importlib.util.spec_from_file_location("pin_base_files", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_pin_files_selects_identity_files_and_shards():
    # F-031: the FULL tokenizer surface (tokenizer.json / vocab.json / merges.txt)
    # and generation_config.json are identity files - an earlier version of this
    # test asserted tokenizer.json was EXCLUDED, which documented the exact gap a
    # swapped-tokenizer chimera walked through.
    m = _load()
    names = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "README.md", "LICENSE", ".gitattributes", "figures/plot.png",
    ]
    picked = m.pick_pin_files(names)
    assert picked == [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]


def test_pick_pin_files_single_shard_no_index():
    # A single-file checkpoint has model.safetensors and no index; the picker must
    # not invent a missing index requirement.
    m = _load()
    picked = m.pick_pin_files(["config.json", "model.safetensors",
                               "tokenizer_config.json", "README.md"])
    assert picked == ["config.json", "model.safetensors", "tokenizer_config.json"]


def test_render_pins_emits_recipe_ready_snippet():
    m = _load()
    sha_a, sha_b = "a" * 64, "b" * 64
    out = m.render_pins("c" * 40, {"model.safetensors": sha_b, "config.json": sha_a})
    # Paste-ready for recipes.py: revision first, files sorted for a stable diff.
    assert f'base_revision="{"c" * 40}"' in out
    assert out.index('"config.json"') < out.index('"model.safetensors"')
    assert sha_a in out and sha_b in out


def test_diff_pins_empty_on_match():
    m = _load()
    files = {"config.json": "a" * 64}
    assert m.diff_pins("c" * 40, files, "c" * 40, dict(files)) == []


def test_diff_pins_reports_revision_and_content_drift():
    m = _load()
    old = {"config.json": "a" * 64, "gone.safetensors": "d" * 64}
    new = {"config.json": "b" * 64, "added.safetensors": "e" * 64}
    lines = "\n".join(m.diff_pins("c" * 40, old, "f" * 40, new))
    assert "revision" in lines
    assert "config.json" in lines          # content changed
    assert "gone.safetensors" in lines     # removed upstream
    assert "added.safetensors" in lines    # new upstream
