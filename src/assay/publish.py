from __future__ import annotations

import math
import os
from datetime import datetime, timezone

from assay import __version__
from assay.config import GateThresholds, RunConfig
from assay.gate import GateResult, render_delta_table
from assay.recipes import Recipe

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


def _scheme_bullets(scheme: str) -> list[str]:
    """Overview bullets for the ACTUAL scheme. These used to be unconditional prose
    describing weight-only 4-bit weights / 16-bit activations, so a W4A4 recipe would
    have been handed a card calling it weight-only - the same latent falsehood class as
    the single-task max()/min() bug. `_hardware_section` already branches on this
    predicate; the overview now does too.

    The W4A4 comparison is attributed rather than asserted. assay has never measured
    W4A4 accuracy: the one in-house datum is a perplexity rejection (+12.55%), which is
    real, narrow, and worth more than the folklore version on a card whose closing line
    is "actual measured numbers, not vendor estimates"."""
    if not _is_weight_only(scheme):
        return [
            f"- **Scheme:** `{scheme}` - quantizes **activations as well as weights**. This is "
            "not a weight-only checkpoint; see Hardware requirements.",
        ]
    return [
        f"- **Scheme:** `{scheme}` - 4-bit NVFP4 **weights**, 16-bit (bf16) **activations** "
        "(weight-only).",
        "- **Why weight-only:** activations are transient (never stored), so keeping them at "
        "16-bit costs ~nothing on disk - weights dominate size. Fully 4-bit *activation* "
        "quantization (W4A4) is widely reported to cost token-level quality; we have not "
        "measured that ourselves, and the one W4A4 candidate assay did gate failed the "
        "perplexity bar at +12.55% and was rejected. You get almost all the compression at a "
        "quality cost this card measures rather than estimates.",
    ]


def _citation_section(runcfg: RunConfig) -> str:
    """A BibTeX Citation block so the checkpoint is citable (and a DOI mint has a
    citation to anchor). Generated - not hand-added to the live card - so a re-cert
    regenerates it instead of silently dropping it. Year is the publish year."""
    name = runcfg.checkpoint_repo.split("/")[-1]
    base = runcfg.recipe.base_model
    base_short = base.split("/")[-1]
    key = "uist_labs_" + name.lower().replace("-", "_").replace(".", "_")
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
        # F-032: this bullet must describe the PAIRED test the gate actually runs
        # (BITE 2), in the same terms as the "How the standard error is computed"
        # section below - never the deleted unpaired combination.
        bullets.append(
            f"- No statistically significant per-task accuracy regression: one-sided "
            f"paired test, a task fails only if its drop exceeds k={gate.k_stderr:g} times "
            f"the standard error of the per-item score differences")
    if gate.max_single_drop_pts is not None:
        bullets.append(f"- No single accuracy task down more than {gate.max_single_drop_pts:.1f} points")
    if gate.max_ppl_increase is not None:
        bullets.append(f"- Perplexity increase <= {gate.max_ppl_increase:.0%}")
    return bullets


def _sampling_protocol(recipe: Recipe, task: str) -> str:
    """Repeat protocol for one accuracy task, DERIVED FROM THE RECIPE (`Eval.repeats`)
    and never from a task-name lookup - a new recipe with different repeats gets a
    correct card with no edit here.

    Returns "" for a task the recipe does not repeat, and that silence is deliberate.
    An earlier draft printed "single draw", which is itself a guessed classification:
    the recipe cannot distinguish one sampled draw from no sampling at all (a
    loglikelihood-scored or greedy task), so on a battery like Qwen's "single draw"
    would be false. assay has no task taxonomy, and swapping one false claim for a
    guessed one is not an honesty pass. The card states repeats, which the recipe
    really does declare, and nothing more."""
    k = recipe.eval.repeats.get(task, 1)
    return f" (avg@{k}, the mean of {k} samples per item)" if k > 1 else ""


def _generation_settings(recipe: Recipe) -> str | None:
    """The run's pinned sampling settings, or None if the recipe pins none (harness/task
    defaults then apply). `Eval.gen_kwargs` is global to the run, not per task, so this
    is stated once rather than attached to individual rows. Only genuine sampling knobs
    are surfaced - `do_sample` is a bool switch and `max_gen_toks` is a length cap,
    neither of which tells a reader anything about score variance."""
    gk = recipe.eval.gen_kwargs or {}
    shown = [f"{key} {gk[key]:g}" for key in ("temperature", "top_p", "top_k")
             if isinstance(gk.get(key), (int, float)) and not isinstance(gk.get(key), bool)]
    return ", ".join(shown) if shown else None


def _sampling_params_line(recipe: Recipe) -> str:
    """The usage snippet's SamplingParams, from the recipe's own `gen_kwargs` where it
    pins them, so the card cannot recommend settings the certification did not use.
    With nothing pinned, show only a length cap and leave sampling at the library's
    defaults rather than inventing a recommendation the recipe never made."""
    gk = recipe.eval.gen_kwargs or {}
    parts = [f"{key}={gk[key]:g}" for key in ("temperature", "top_p", "top_k")
             if isinstance(gk.get(key), (int, float)) and not isinstance(gk.get(key), bool)]
    return "params = SamplingParams(" + ", ".join(parts + ["max_tokens=256"]) + ")"


def _scope_note(recipe: Recipe, result: GateResult) -> str:
    """What the certification actually asserts: a same-run, same-stack DELTA, not an
    absolute score. The published twin of the cached-baseline rejection - a baseline
    measured on a different stack would fold stack drift into the retention number.

    Two clauses are conditional, because each is FALSE on some battery:
    - The rerun-variance clause only holds where the recipe actually enables sampling.
      On a greedy / loglikelihood battery, rerunning on the same stack reproduces the
      same score exactly, so claiming run-to-run movement would invent noise.
    - The magnitude cites the table's stderr column only when the gate ran in
      significance mode and that column is rendered; a point-gated card has no stderrs,
      so "the standard errors shown" would point at a number the reader cannot see.
    The stack-drift clause is unconditional - it is evidenced across v0.4.0 to v0.5.0,
    where wikitext (a DETERMINISTIC metric) moved 0.46%."""
    has_stderr = any(d.combined_stderr is not None for d in result.accuracy_deltas)
    rerun = ""
    if _generation_settings(recipe):
        # "up to roughly": the reported stderr is uncertainty over ITEM sampling, which
        # bounds same-item rerun variance rather than equalling it. Overstating our own
        # noise would be the same species of error as understating it.
        magnitude = ("up to roughly the standard errors shown in the table above"
                     if has_stderr else "by an amount set by the size of the benchmark set")
        rerun = (f" Because this battery samples its answers, rerunning it on the same stack "
                 f"moves absolute scores by {magnitude}.")
    return (
        "- **Scope: the certified quantity is the delta, not the absolute score.** Both sides "
        "were measured in the same run against the same software stack, which is the only "
        "condition under which these two columns are comparable." + rerun + " Absolute scores "
        "can move further across harness or library versions, so a number here will not "
        "necessarily reproduce elsewhere, while the delta under identical conditions is what "
        "was certified. For the same reason we do not compare against a stored baseline from "
        "an earlier run: that would fold stack drift into the measurement.")


def _statistical_notes(recipe: Recipe, result: GateResult, gate: GateThresholds) -> list[str]:
    """Significance-gate honesty block: what the gate could and could not have caught.

    Replaces the old "power varies" note, which named one task as power-limited and
    another as carrying "the certification's binding, low-variance signal". That was
    false: the gate is CONJUNCTIVE (gate.py - any significant task fails the run), so no
    task is binding and none is decorative. It also broke on a single-task battery, where
    max() and min() return the same delta and one task was named as both.

    What replaces it is checkable rather than reassuring: per task, the observed drop the
    gate would have flagged. That turns the card into a stated non-inferiority claim a
    reader can verify against the table's own stderr column."""
    if gate.k_stderr is None:
        # Point gates state their bar directly in the certification criteria
        # (max_single_drop_pts IS the flag threshold); nothing to add here.
        return []
    scored = [d for d in result.accuracy_deltas if d.combined_stderr is not None]
    if not scored:
        return []
    k = gate.k_stderr
    # Recipe order, NOT sorted by magnitude: ranking the tasks is what produced the
    # "binding signal" claim in the first place.
    per_task = "; ".join(
        f"`{d.task}`{_sampling_protocol(recipe, d.task)} {k * d.combined_stderr * 100.0:.1f} pts"
        for d in scored)
    upper_bounds = "; ".join(
        f"`{d.task}` {(-d.delta + k * d.combined_stderr) * 100.0:.1f} pts"
        for d in scored)
    gen = _generation_settings(recipe)
    gen_note = (
        # "the harness scores OTHER tasks by loglikelihood" would imply this battery
        # contains some - the same battery-shape implication as the multiple-choice
        # clause removed from mode_note. Stated as a conditional instead, so it is true
        # whatever the battery holds.
        f"- **Run-level generation settings: {gen}.** These apply where a task generates "
        "its answer, and have no effect on any task the harness scores by loglikelihood. "
        "The baseline and this checkpoint were measured with identical settings."
        if gen else
        "- **No run-level sampling overrides were set**; the harness's per-task defaults "
        "applied, identically to the baseline and this checkpoint.")
    # One-sided normal tail at k. math.erfc is the stdlib special function, not a
    # hand-rolled approximation. Rendered to ONE significant figure on purpose: it is a
    # NOMINAL rate from a normal approximation over small-n binary/averaged metrics, and
    # 3 significant figures would imply a calibration we have not verified.
    false_fail = 0.5 * math.erfc(k / math.sqrt(2.0))
    ppl_note = ""
    if result.perplexity_delta is not None and gate.max_ppl_increase is not None:
        ppl_note = (
            f" Separately, the perplexity criterion is an independent hard bar (increase no "
            f"more than {gate.max_ppl_increase:.0%}) that does not depend on this test at all.")
    return [
        # NOT "minimum detectable regression" and NOT "rules out a regression larger than
        # X". In power-analysis usage "detectable" means detected with stated POWER
        # (~(z_alpha+z_beta)*SE); k*SE is the decision threshold on the MEASURED drop, and
        # a true regression of exactly that size clears it only about half the time. The
        # honest claim is about the decision rule, so the wording stays on the decision rule.
        f"- **Per-task fail thresholds (this run).** The gate fails a task only when its "
        f"measured drop exceeds k={k:g} times the paired standard error of the per-item "
        f"score differences. On this run those thresholds were: {per_task}. A measured drop at "
        "or below a task's threshold passed as statistically indistinguishable from zero - so "
        "this certification does not assert that no regression exists below that size. "
        "Threshold width tracks each task's sampling noise, not its importance: a wide "
        "threshold means this run had limited resolving power on that task." + ppl_note,
        # The test's construction is disclosed rather than characterized (D10): the
        # card records WHICH test certified the run. Never describe the old unpaired
        # combination as "conservative" - it was conservative against a false FAIL but
        # lenient against a false PASS, the direction a certification reader cares about.
        "- **How the standard error is computed.** This is a paired test: both "
        "evaluations score the identical items, so the standard error is computed from "
        "the per-item score differences (quantized minus baseline, item by item) rather "
        "than by combining the two sides' independent standard errors. Pairing credits "
        "the correlation the two sides share through item difficulty; the per-side "
        "stderr column in the table is informative only. These are the thresholds the "
        "gate actually enforced.",
        # The actual non-inferiority certificate: only meaningful now that the SE is
        # paired. Negative values mean even the bound shows an improvement.
        "- **One-sided upper confidence bound on the true regression.** At the same "
        f"k={k:g}, measured drop + k*SE per task: {upper_bounds}. This is the "
        "non-inferiority claim this certification makes: the data are consistent with "
        "a true per-task regression of at most these sizes.",
        f"- **A sound quantization can still fail on noise.** The per-task test is one-sided "
        f"at k={k:g}, a nominal false-alarm rate of about {false_fail * 100.0:.0f}% per task, "
        "and every task is tested - so the run-level chance is higher than any single task's. "
        "We do not quote a run-level figure: the tasks share one checkpoint and are not "
        "independent. We accept those odds in the withholding direction, since a false alarm "
        "costs us a release while a missed regression would cost you a bad checkpoint.",
        gen_note,
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
        # The old text added "- most visibly on the multiple-choice tasks -", a
        # battery-shape hardcode of exactly the class the modularity rule bans: the R1
        # battery has NO multiple-choice tasks, so the live R1 card described tasks it
        # does not contain. The template-shift point stands without it.
        mode_note = ("- **Chat-mode evaluation.** Tasks are evaluated with the model's chat "
                     "template applied. The template shifts the absolute scores on both the "
                     "baseline and the quantized model, so read the deltas, not the absolute "
                     "values. The comparison stays valid because both sides are evaluated "
                     "with identical settings.")
    else:
        mode_note = ("- **Absolute scores are raw-completion numbers.** The harness runs these "
                     "tasks without the chat template, so absolute values run lower than a chat "
                     "session; this does not affect the deltas (both sides measured identically).")
    return "\n".join([
        "---",
        # From the RECIPE, never a constant: a quantization is a derivative work, so the
        # card must declare the base model's license. Hardcoding apache-2.0 misdeclared
        # the R1 checkpoint, whose base model is MIT.
        f"license: {recipe.license}",
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
        *_scheme_bullets(scheme),
        "- **Format:** compressed-tensors; loads natively in vLLM (no `--quantization` flag).",
        "",
        "## Use with vLLM",
        "",
        "```python",
        "from vllm import LLM, SamplingParams",
        "",
        f'llm = LLM(model="{runcfg.checkpoint_repo}")',
        'prompts = ["Give me a short introduction to large language models."]',
        # Derived, not hardcoded: the snippet used to show temperature=0.7/top_p=0.8 on
        # every card, which directly contradicted the R1 recipe's own required 0.6/0.95
        # - the card handed the reader settings the certification did not use.
        _sampling_params_line(recipe),
        "for out in llm.generate(prompts, params):",
        "    print(out.outputs[0].text)",
        "```",
        "",
        _hardware_section(scheme),
        "",
        "## Creation",
        "",
        # Only when the recipe carries identity pins: an empty revision would render
        # a broken-looking line, and the claim below is only true when the in-pod
        # verifier (assay.verify) had pins to check against.
        *([f"- **Base snapshot:** [`{base}` @ `{recipe.base_revision[:12]}`]"
           f"(https://huggingface.co/{base}/tree/{recipe.base_revision}) - upstream "
           "repos are mutable, so the certificate names the exact commit it describes; "
           "the staged weights were verified against the recipe's pinned sha256s "
           "before quantization."]
          if recipe.base_revision else []),
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
        _scope_note(recipe, result),
        *_statistical_notes(recipe, result, gate),
        # "small benchmark sets vary run to run" is false for a greedy/loglikelihood
        # battery on a fixed stack, where a rerun reproduces the score exactly. The real
        # and battery-independent mechanism is finite-item sampling noise.
        "- **Retention near or above 100% means \"no measurable loss,\" not \"better.\"** Where a "
        "task ticks up, that is sampling noise - a finite benchmark set is a sample, not the "
        "whole population - so read the whole table as \"indistinguishable from the original,\" "
        "not as an improvement.",
        "- **Weight-only tradeoff.** Weights are 4-bit; activations stay 16-bit. That keeps "
        "quality close to the original at nearly the full disk-size saving, at some inference-"
        "speed cost versus a fully 4-bit (W4A4) variant. We have not benchmarked a W4A4 build "
        "of this model, so we make no claim about where it lands on that curve.",
        # SCOPE OF THE CERTIFICATION - the card's most consequential sentence, and it was
        # false. The old text said quantization "faithfully reproduces the base model's
        # behavior (the gate above certifies exactly that), it does not add or remove
        # bias". The gate measures accuracy deltas on a handful of listed benchmarks plus
        # a perplexity ratio. It certifies nothing about bias, safety, or behavior at
        # large, and "does not add or remove bias" is an unmeasured empirical claim.
        # Overclaiming what a certification COVERS is worse than overclaiming its
        # strength: it invites reliance we did not earn.
        f"- **Bias, risks, and inherited behavior.** This is a quantization of "
        f"[`{base}`](https://huggingface.co/{base}) and inherits its capabilities, biases, and "
        "limitations. The gate above certifies accuracy retention on the listed benchmarks "
        "only; it does not measure bias, safety, or any behavior those benchmarks do not "
        "cover, and quantization is not guaranteed to preserve what was not measured. For "
        "intended use, safety, and ethical considerations, refer to the base model's card.",
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
                      hb=None, api=None, *, dry_run: bool) -> bool:
    """Push to HF only on a passing gate. Returns whether it published.
    dry_run (non-pristine run: any non-cert tier, or any runtime override on a cert-tier
    run): build the model card (exercises card generation) but never upload - returns
    False.

    dry_run is KEYWORD-ONLY and REQUIRED on purpose. It used to default to False - the
    live, irreversible action - so a caller that simply forgot it would upload for real.
    Publish-integrity bits are never defaulted; keyword-only additionally stops it being
    supplied positionally by accident as the parameter list grows."""
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
