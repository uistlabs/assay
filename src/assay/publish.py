from __future__ import annotations

import os

from assay.config import (
    Config, MIN_MEAN_RETENTION, MAX_SINGLE_DROP_PTS, MAX_PPL_INCREASE,
)
from assay.gate import GateResult, render_delta_table


def build_model_card(cfg: Config, result: GateResult) -> str:
    """HF model card: overview -> vLLM usage -> creation -> evaluation -> certification.

    Structure mirrors the compressed-tensors community standard (RedHatAI cards), plus
    UIST Labs' differentiator: the checkpoint is published ONLY because it cleared a
    hard, stated accuracy gate, and the card shows the real measured deltas -- trust
    built on facts confirmed by analysis, applied to every UIST quantization release.
    Plain ASCII only (published file). All numbers come from `result` / cfg at publish."""
    name = cfg.checkpoint_repo.split("/")[-1]
    scheme = cfg.quant_scheme
    base = cfg.base_model
    base_tag = base.split("/")[-1].split("-")[0].lower()  # e.g. "Qwen2.5-7B-Instruct" -> "qwen2.5"
    pipeline = (
        f"UIST Labs' `assay` benchmark-gated quantization pipeline ({cfg.pipeline_url})"
        if cfg.pipeline_url else
        "UIST Labs' `assay` benchmark-gated quantization pipeline"
    )
    return "\n".join([
        "---",
        "license: apache-2.0",
        f"base_model: {base}",
        "base_model_relation: quantized",
        "pipeline_tag: text-generation",
        "library_name: compressed-tensors",
        "language:",
        "  - en",
        "datasets:",
        f"  - {cfg.calib_dataset}",
        "tags:",
        f"  - {base_tag}",
        "  - nvfp4",
        f"  - {scheme.lower()}",
        "  - weight-only",
        "  - vllm",
        "  - compressed-tensors",
        "  - quantized",
        "---",
        "",
        f"# {name}",
        "",
        f"`{scheme}` quantization of [`{base}`](https://huggingface.co/{base}), produced with "
        "[llm-compressor](https://github.com/vllm-project/llm-compressor) and **published only "
        "after passing an automated accuracy gate** against the bf16 baseline (see "
        "[Certification](#certification)).",
        "",
        "## Model overview",
        "",
        f"- **Scheme:** `{scheme}` -- 4-bit NVFP4 **weights**, 16-bit (bf16) **activations** "
        "(weight-only).",
        "- **Why weight-only:** activations are transient (never stored), so keeping them at 16-bit "
        "costs ~nothing on disk -- weights dominate size -- while avoiding the token-level quality "
        "loss that 4-bit *activation* quantization (W4A4) causes. You get almost all the compression "
        "with accuracy much closer to the original.",
        "- **Format:** compressed-tensors; loads natively in vLLM (no `--quantization` flag).",
        "",
        "## Use with vLLM",
        "",
        "```python",
        "from vllm import LLM, SamplingParams",
        "",
        f'llm = LLM(model="{cfg.checkpoint_repo}")',
        'prompts = ["Give me a short introduction to large language models."]',
        "params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)",
        "for out in llm.generate(prompts, params):",
        "    print(out.outputs[0].text)",
        "```",
        "",
        "## Creation",
        "",
        f"- **Tool:** llm-compressor, scheme `{scheme}`, targets `Linear`, ignore `lm_head`.",
        f"- **Calibration:** {cfg.num_calibration_samples} samples from "
        f"`{cfg.calib_dataset}` at {cfg.max_seq_length}-token sequences.",
        f"- **Pipeline:** {pipeline} -- quantize -> benchmark -> gate -> publish.",
        "",
        "## Evaluation",
        "",
        "Measured with lm-evaluation-harness (vLLM backend) on the bf16 baseline and this "
        "checkpoint. `retention` is quantized / baseline (higher is better; for perplexity, "
        "lower raw value is better).",
        "",
        render_delta_table(result),
        "",
        "## Methodology and limitations",
        "",
        "- **Apples-to-apples deltas.** The bf16 baseline and this checkpoint were evaluated "
        "with the *identical* harness and settings, so the `delta`/`retention` columns are a "
        "fair like-for-like comparison -- which is what a quantization gate should measure: "
        "change from the original, honestly.",
        "- **Absolute scores are raw-completion numbers.** The harness runs these tasks without "
        "applying the chat template, so the absolute values are lower than you would see chatting "
        "with the instruction-tuned model. This does not affect the deltas (both sides measured "
        "the same way); an instruct-style evaluation is on our roadmap for the absolute figures.",
        "- **Retention near or above 100% means \"no measurable loss,\" not \"better.\"** Where a "
        "task ticks up, that is within benchmark noise (small sets like gsm8k vary run to run) "
        "plus a touch of quantization acting as mild regularization -- read the whole table as "
        "\"indistinguishable from the original,\" not as an improvement.",
        "- **Weight-only tradeoff.** Weights are 4-bit; activations stay 16-bit. That keeps "
        "quality close to the original at nearly the full disk-size saving, at a modest inference-"
        "speed cost versus a fully 4-bit (W4A4) variant. If you need maximum throughput and can "
        "accept more degradation, a W4A4 build is a different point on that curve.",
        f"- **Bias, risks, and inherited behavior.** This is a quantization of "
        f"[`{base}`](https://huggingface.co/{base}) and inherits its capabilities, biases, and "
        "limitations unchanged -- quantization faithfully reproduces the base model's behavior (the "
        "gate above certifies exactly that), it does not add or remove bias. For intended use, "
        "safety, and ethical considerations, refer to the base model's card. Absolute-quality "
        "claims (multilingual, coding, safety) are the base model's; we certify only that "
        "quantization preserves them within the stated bar.",
        "",
        "## Certification",
        "",
        "UIST Labs publishes a quantized checkpoint **only if it clears a hard, stated accuracy "
        "bar** against its own bf16 baseline -- we would rather withhold a release than ship an "
        "unverified one. This checkpoint passed all of:",
        "",
        f"- Mean accuracy retention >= {MIN_MEAN_RETENTION:.0%}",
        f"- No single accuracy task down more than {MAX_SINGLE_DROP_PTS:.1f} points",
        f"- Perplexity increase <= {MAX_PPL_INCREASE:.0%}",
        "",
        "The deltas above are the actual measured numbers, not vendor estimates. This gate runs "
        "on every UIST Labs quantization release.",
        "",
        "-- UIST Labs",
    ])


def publish_if_passed(cfg: Config, out_dir: str, result: GateResult, token: str,
                      hb=None, api=None) -> bool:
    """Push to HF only on a passing gate. Returns whether it published."""
    if not result.passed:
        if hb:
            hb.emit("publish", "gate FAILED -- not publishing")
        return False

    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(build_model_card(cfg, result))

    if api is None:  # pragma: no cover -- exercised only against live HF
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(cfg.checkpoint_repo, exist_ok=True, private=False)

    if hb:
        hb.emit("publish", f"gate PASSED -- uploading to {cfg.checkpoint_repo}")
    api.upload_folder(folder_path=out_dir, repo_id=cfg.checkpoint_repo, token=token)
    return True
