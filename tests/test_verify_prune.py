"""The prune verifier is the guard that stopped the image prune from lying.

The regression it exists for: the prune listed exact distribution names, `pip uninstall`
swallowed every miss with `|| true`, and packages whose names had drifted from the list
shipped as dead weight in every pull. The fix is a FAMILY scan, so the tests that matter
are the ones proving a renamed or newly-introduced sibling is still caught.
"""
import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "verify_prune.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_prune", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verify_prune = _load()


class FakeDist:
    """Minimal stand-in for importlib.metadata.Distribution."""

    def __init__(self, name):
        self.metadata = {"Name": name} if name is not None else None


@pytest.mark.parametrize("name", [
    # The three that actually shipped, and the exact reason the check is family-based:
    # each is a SIBLING of a name the prune list did contain.
    "nvidia-cutlass-dsl-libs-base",
    "tokenspeed-mla",
    "pynvvideocodec",
    # Plus a plausible future sibling nobody has listed yet.
    "flashinfer-jit",
])
def test_catches_the_siblings_an_exact_list_missed(name):
    assert verify_prune.survivors([name]) == [name]


@pytest.mark.parametrize("name", [
    "torch", "vllm", "transformers", "lm-eval", "compressed-tensors",
    "llmcompressor", "numpy", "datasets", "huggingface-hub", "runpod",
])
def test_does_not_flag_the_toolchain_we_keep(name):
    """A false positive here fails a correct build, so the load-bearing packages are
    pinned explicitly."""
    assert verify_prune.survivors([name]) == []


def test_normalization_folds_separators_and_case():
    """Distribution names appear with underscores, dots or mixed case depending on who
    wrote the metadata; the scan must see through all of it."""
    assert verify_prune.normalize("NVIDIA_Cutlass.DSL") == "nvidia-cutlass-dsl"
    assert verify_prune.survivors(["PyNvVideoCodec"]) == ["PyNvVideoCodec"]
    assert verify_prune.survivors(["TorchCodec"]) == ["TorchCodec"]


def test_results_are_deduplicated_and_sorted():
    """A stable build log is a diffable build log."""
    got = verify_prune.survivors(["tokenspeed-mla", "torchcodec", "tokenspeed-mla"])
    assert got == ["tokenspeed-mla", "torchcodec"]


def test_a_clean_environment_reports_nothing():
    assert verify_prune.survivors(["torch", "vllm", "numpy"]) == []


def test_malformed_distribution_without_a_name_is_skipped():
    """A dist with no metadata must not crash the build - it is not worth failing a
    multi-GB image over one unreadable package."""
    assert verify_prune.installed_names([FakeDist("torch"), FakeDist(None)]) == ["torch"]


def test_installed_names_reads_the_metadata_name():
    assert verify_prune.installed_names([FakeDist("a"), FakeDist("b")]) == ["a", "b"]


def test_failure_message_names_the_offenders_and_the_remedy():
    """Per the 2 AM rule: the message has to say what broke, what it costs, and the next
    action. A bare AssertionError would cost the next person an investigation."""
    msg = verify_prune.format_failure(["tokenspeed-mla", "pynvvideocodec"])
    # Collapse wrapping before matching phrases - the message is hard-wrapped for the
    # build log, so asserting on raw text would break every time a line is rewrapped.
    flat = " ".join(msg.split())
    assert "tokenspeed-mla" in flat and "pynvvideocodec" in flat
    assert "dead weight" in flat                 # what it costs
    assert "add each name above to prune" in flat.lower()   # the next action
    assert "deploy/Dockerfile" in flat           # where
    assert "BANNED_FAMILIES" in flat             # the escape hatch, if it is load-bearing


def test_main_passes_on_a_clean_tree(monkeypatch, capsys):
    monkeypatch.setattr(verify_prune, "installed_names", lambda *a, **k: ["torch", "vllm"])
    assert verify_prune.main() == 0
    assert "prune verified" in capsys.readouterr().out


def test_main_fails_and_explains_when_a_banned_family_survives(monkeypatch, capsys):
    monkeypatch.setattr(verify_prune, "installed_names",
                        lambda *a, **k: ["torch", "nvidia-cutlass-dsl-libs-base"])
    assert verify_prune.main() == 1
    assert "PRUNE FAILED" in capsys.readouterr().err


def test_every_banned_family_token_is_lowercase_and_normalized():
    """The tokens are compared against an already-normalized name, so a token containing
    an underscore or a capital could never match anything - a silently dead guard."""
    for token in verify_prune.BANNED_FAMILIES:
        assert token == verify_prune.normalize(token), f"{token!r} would never match"
