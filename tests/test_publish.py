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
