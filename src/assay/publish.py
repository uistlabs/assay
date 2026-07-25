from __future__ import annotations

import os
from datetime import datetime, timezone

from assay import __version__
from assay.config import GateThresholds, RunConfig
from assay.gate import GateResult, render_delta_table

# Measured through assay's own gate. Bump ONLY as more arches are validated; never claim an
# arch we have not run. Turing (sm_75) is excluded: a known vLLM correctness bug makes NVFP4
# Marlin emit garbage there, so the nominal source gate (capability >= 75) overclaims.
_VALIDATED_ARCHES = "Ada (sm_89: e.g. L4, RTX 4090)"


def _is_weight_only(scheme: str) -> bool:
    """Weight-only NVFP4 schemes are named e.g. NVFP4A16 / W4A16 - the A16 marks
    16-bit activations. Anything else (e.g. plain NVFP4, W4A4) quantizes
    activations too and needs Blackwell native FP4, not the Marlin dequant path."""
    return "A16" in scheme.upper()


def _hardware_section(scheme: str) -> str:
    if _is_weight_only(scheme):
        return "\n".join([
            "## Hardware requirements",
            "",
            "This checkpoint uses **weight-only** NVFP4: it runs via vLLM's **FP4 Marlin** kernel "
            "and does **not** require Blackwell GPUs or native FP4 tensor cores. On non-Blackwell "
            "hardware the 4-bit weights are dequantized to 16-bit for the GEMM, trading some "
            "compute throughput for the smaller memory footprint (the KV cache and activations are "
            "16-bit either way, so the saving is on the linear weights).",
            "",
            f"**Validated (measured through assay's gate): {_VALIDATED_ARCHES}.** Other >= sm_80 "
            "(Ampere / Hopper) GPUs are expected to work by the same weight-only Marlin path but "
            "are **not yet independently validated here**. **Turing (sm_75) is excluded**: a known "
            "vLLM issue makes the NVFP4 Marlin GEMM emit incorrect output on Turing, so we do not "
            "claim it until that is fixed and measured.",
        ])
    return "\n".join([
        "## Hardware requirements",
        "",
        f"This checkpoint uses scheme `{scheme}`, which quantizes activations as well as weights.",
        "Native FP4 activation compute requires **Blackwell (SM100+) GPUs with native FP4 tensor",
        "cores**. This configuration is **not validated by assay's gate here** - the certified",
        "releases are weight-only (NVFP4A16); treat non-weight-only hardware support as unverified.",
    ])


def _citation_section(runcfg: RunConfig) -> str:
    """A BibTeX Citation block so the checkpoint is citable (and a DOI mint has a
    citation to anchor). Generated - not hand-added to the live card - so a re-cert
    regenerates it instead of silently dropping it. Year is the publish year."""
    name = runcfg.checkpoint_repo.split("/")[-1]
    base = runcfg.recipe.base_model
    base_short = base.split("/")[-1]
    key = "uistlabs_" + name.lower().replace("-", "_").replace(".", "_")
    year = datetime.now(timezone.utc).year
    pipe = f" ({runcfg.pipeline_url})" if runcfg.pipeline_url else ""
    return "\n".join([
        "## Citation",
        "",
        "If you use this checkpoint, please cite both this quantized release and the base model.",
        "",
        "```bibtex",
        f"@misc{{{key},",
        f"  title        = {{{name}: benchmark-gated NVFP4 quantization of {base_short}}},",
        "  author       = {{UIST Labs}},",
        f"  year         = {{{year}}},",
        "  publisher    = {Hugging Face},",
        f"  howpublished = {{\\url{{https://huggingface.co/{runcfg.checkpoint_repo}}}}},",
        f"  note         = {{Quantized and certified with the assay pipeline{pipe}; published "
        "only after passing an automated accuracy gate against the bf16 baseline.}",
        "}",
        "```",
        "",
        f"Please also cite the base model, [`{base}`](https://huggingface.co/{base}).",
    ])


def _certification_criteria(gate: GateThresholds) -> list[str]:
    """The certification bullets, one per ACTIVE threshold. A None field means that
    check is not part of this recipe's gate, so it is NOT claimed on the card - the
    card must describe the gate that actually ran. A significance-gated recipe
    (k_stderr set) states the significance basis, never a point-drop bar it did not use."""
    bullets: list[str] = []
    if gate.min_mean_retention is not None:
        bullets.append(f"- Mean accuracy retention >= {gate.min_mean_retention:.0%}")
    if gate.k_stderr is not None:
        bullets.append(
            f"- No statistically significant per-task accuracy regression: one-sided, a task "
            f"fails only if its drop exceeds k={gate.k_stderr:g} times the combined standard "
            f"error of the baseline and quantized scores")
    if gate.max_single_drop_pts is not None:
        bullets.append(f"- No single accuracy task down more than {gate.max_single_drop_pts:.1f} points")
    if gate.max_ppl_increase is not None:
        bullets.append(f"- Perplexity increase <= {gate.max_ppl_increase:.0%}")
    return bullets


def _power_note(result: GateResult, gate: GateThresholds) -> list[str]:
    """For significance recipes, name the most power-limited (highest combined stderr,
    directional-only) and the binding (lowest) accuracy task from the ACTUAL result, plus
    perplexity as the low-variance backstop. Dynamic -> self-healing, no hardcoded task
    names (the class of bug that left a stale 'gsm8k' caveat on a battery without gsm8k)."""
    if gate.k_stderr is None:
        return []
    scored = [d for d in result.accuracy_deltas if d.combined_stderr is not None]
    if not scored:
        return []
    weakest = max(scored, key=lambda d: d.combined_stderr)
    strongest = min(scored, key=lambda d: d.combined_stderr)
    ppl = " and perplexity" if result.perplexity_delta is not None else ""
    return [
        f"- **Per-task statistical power varies.** `{weakest.task}` has the largest combined "
        f"standard error ({weakest.combined_stderr:.3f}) and is power-limited - read its delta "
        f"as directional only. `{strongest.task}` ({strongest.combined_stderr:.3f}){ppl} carry "
        "the certification's binding, low-variance signal. The gate tests each task for a "
        "statistically significant regression, so an underpowered task cannot fail a sound quant "
        "on noise alone."
    ]


def build_model_card(runcfg: RunConfig, result: GateResult) -> str:
    """HF model card: overview -> vLLM usage -> creation -> evaluation -> certification.

    Structure mirrors the compressed-tensors community standard (RedHatAI cards), plus
    UIST Labs' differentiator: the checkpoint is published ONLY because it cleared a
    hard, stated accuracy gate, and the card shows the real measured deltas - trust
    built on facts confirmed by analysis, applied to every UIST quantization release.
    Plain ASCII only (published file). All numbers come from `result` / runcfg at publish."""
    recipe = runcfg.recipe
    name = runcfg.checkpoint_repo.split("/")[-1]
    scheme = recipe.quant_scheme
    base = recipe.base_model
    gate = recipe.gate_or_default
    pipeline = (
        f"UIST Labs' `assay` benchmark-gated quantization pipeline ({runcfg.pipeline_url})"
        if runcfg.pipeline_url else
        "UIST Labs' `assay` benchmark-gated quantization pipeline"
    )
    if recipe.eval.mode == "chat":
        # NOTE: do not claim the absolute scores "reflect real usage". True only for
        # generative tasks (e.g. gsm8k); the loglikelihood multiple-choice tasks
        # (arc/hellaswag/winogrande/mmlu) are scored by choice-loglikelihood, which a
        # chat template shifts but does not make "real usage". Say what is measured.
        mode_note = ("- **Chat-mode evaluation.** Tasks are evaluated with the model's chat "
                     "template applied. The template shifts the absolute scores on both the "
                     "baseline and the quantized model - most visibly on the multiple-choice "
                     "tasks - so read the deltas, not the absolute values. The comparison "
                     "stays valid because both sides are evaluated with identical settings.")
    else:
        mode_note = ("- **Absolute scores are raw-completion numbers.** The harness runs these "
                     "tasks without the chat template, so absolute values run lower than a chat "
                     "session; this does not affect the deltas (both sides measured identically).")
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
        f"  - {recipe.calib.dataset}",
        "tags:",
        *[f"  - {tag}" for tag in recipe.tags],
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
        f"- **Scheme:** `{scheme}` - 4-bit NVFP4 **weights**, 16-bit (bf16) **activations** "
        "(weight-only).",
        "- **Why weight-only:** activations are transient (never stored), so keeping them at 16-bit "
        "costs ~nothing on disk - weights dominate size - while avoiding the token-level quality "
        "loss that 4-bit *activation* quantization (W4A4) causes. You get almost all the compression "
        "with accuracy much closer to the original.",
        "- **Format:** compressed-tensors; loads natively in vLLM (no `--quantization` flag).",
        "",
        "## Use with vLLM",
        "",
        "```python",
        "from vllm import LLM, SamplingParams",
        "",
        f'llm = LLM(model="{runcfg.checkpoint_repo}")',
        'prompts = ["Give me a short introduction to large language models."]',
        "params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)",
        "for out in llm.generate(prompts, params):",
        "    print(out.outputs[0].text)",
        "```",
        "",
        _hardware_section(scheme),
        "",
        "## Creation",
        "",
        f"- **Tool:** llm-compressor, scheme `{scheme}`, targets `Linear`, ignore `lm_head`.",
        f"- **Calibration:** {recipe.calib.num_samples} samples from "
        f"`{recipe.calib.dataset}` at {recipe.calib.max_seq_len}-token sequences.",
        f"- **Pipeline:** {pipeline} - quantize -> benchmark -> gate -> publish.",
        "",
        "## Evaluation",
        "",
        "Measured with lm-evaluation-harness (vLLM backend) on the bf16 baseline and this "
        "checkpoint. `retention` is quantized / baseline (higher is better; for perplexity, "
        "lower raw value is better).",
        "",
        render_delta_table(result, gate),
        "",
        "## Methodology and limitations",
        "",
        "- **Apples-to-apples deltas.** The bf16 baseline and this checkpoint were evaluated "
        "with the *identical* harness and settings, so the `delta`/`retention` columns are a "
        "fair like-for-like comparison - which is what a quantization gate should measure: "
        "change from the original, honestly.",
        mode_note,
        *_power_note(result, gate),
        "- **Retention near or above 100% means \"no measurable loss,\" not \"better.\"** Where a "
        "task ticks up, that is within benchmark noise (small benchmark sets vary run to run) "
        "plus a touch of quantization acting as mild regularization - read the whole table as "
        "\"indistinguishable from the original,\" not as an improvement.",
        "- **Weight-only tradeoff.** Weights are 4-bit; activations stay 16-bit. That keeps "
        "quality close to the original at nearly the full disk-size saving, at a modest inference-"
        "speed cost versus a fully 4-bit (W4A4) variant. If you need maximum throughput and can "
        "accept more degradation, a W4A4 build is a different point on that curve.",
        f"- **Bias, risks, and inherited behavior.** This is a quantization of "
        f"[`{base}`](https://huggingface.co/{base}) and inherits its capabilities, biases, and "
        "limitations unchanged - quantization faithfully reproduces the base model's behavior (the "
        "gate above certifies exactly that), it does not add or remove bias. For intended use, "
        "safety, and ethical considerations, refer to the base model's card. Absolute-quality "
        "claims (multilingual, coding, safety) are the base model's; we certify only that "
        "quantization preserves them within the stated bar.",
        "",
        "## Certification",
        "",
        "UIST Labs publishes a quantized checkpoint **only if it clears a hard, stated accuracy "
        "bar** against its own bf16 baseline - we would rather withhold a release than ship an "
        "unverified one. This checkpoint passed all of:",
        "",
        *_certification_criteria(gate),
        "",
        "The deltas above are the actual measured numbers, not vendor estimates. This gate runs "
        "on every UIST Labs quantization release.",
        "",
        _citation_section(runcfg),
        "",
        f"Produced by assay v{__version__} - UIST Labs",
    ])


def publish_if_passed(runcfg: RunConfig, out_dir: str, result: GateResult, token: str,
                      hb=None, api=None, dry_run: bool = False) -> bool:
    """Push to HF only on a passing gate. Returns whether it published.
    dry_run (non-pristine run: any non-cert tier, or any runtime override on a cert-tier
    run): build the model card (exercises card generation) but never upload - returns
    False."""
    if not result.passed:
        if hb:
            hb.emit("publish", "gate FAILED - not publishing")
        return False

    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(build_model_card(runcfg, result))

    if dry_run:
        if hb:
            hb.emit("publish", "SMOKE dry-run - card built, upload skipped")
        return False

    if api is None:  # pragma: no cover - exercised only against live HF
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(runcfg.checkpoint_repo, exist_ok=True, private=False)

    if hb:
        hb.emit("publish", f"gate PASSED - uploading to {runcfg.checkpoint_repo}")
    api.upload_folder(folder_path=out_dir, repo_id=runcfg.checkpoint_repo, token=token)
    return True
