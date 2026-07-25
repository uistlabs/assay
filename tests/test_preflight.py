from assay.preflight import base_model_card_url, launch_reminder
from assay.recipes import get_recipe


def test_base_model_card_url():
    assert base_model_card_url("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") == \
        "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def test_launch_reminder_names_base_card_and_checklist():
    r = get_recipe("r1_distill_qwen_7b")
    msg = launch_reminder(r)
    assert "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" in msg
    assert "sampling" in msg.lower()          # points at the authoring checklist concerns
    assert msg.isascii()                       # shipped output stays 7-bit ASCII
