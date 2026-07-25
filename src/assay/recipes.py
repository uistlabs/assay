from __future__ import annotations

from dataclasses import dataclass, field

from assay.config import DEFAULT_GATE, GateThresholds

_VALID_MODES = ("chat", "completion")


@dataclass(frozen=True)
class Calib:
    dataset: str
    split: str
    num_samples: int
    max_seq_len: int


@dataclass(frozen=True)
class Eval:
    # Each accuracy task carries its FULLY-QUALIFIED lm-eval metric key, incl. any
    # filter suffix (e.g. gpqa reports both exact_match,flexible-extract and
    # exact_match,strict-match - naming the filter removes the ambiguity).
    accuracy_tasks: tuple[tuple[str, str], ...]
    perplexity: tuple[str, str] | None      # (task, metric) or None to skip the ppl gate
    mode: str                                # "chat" | "completion"
    gen_kwargs: dict | None                  # sampling for generative tasks; global to the run
    system_prompt: str | None                # None => no system prompt (R1 requires this)
    prompt_prefix: str | None                # forced assistant-turn prefix, e.g. "<think>\n"
    # Targeted multi-sampling: task -> sample count K (avg@K). Empty => every task
    # single-sampled (today's behavior). Spend K only where stderr is large: aime
    # (n=30, ~9pt stderr) needs it; minerva/gpqa are already tight.
    repeats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe:
    slug: str
    base_model: str
    quant_scheme: str
    calib: Calib
    eval: Eval
    gate: GateThresholds | None              # None => DEFAULT_GATE
    tags: tuple[str, ...]

    @property
    def accuracy_task_names(self) -> tuple[str, ...]:
        return tuple(task for task, _ in self.eval.accuracy_tasks)

    @property
    def perplexity_task_name(self) -> str | None:
        return None if self.eval.perplexity is None else self.eval.perplexity[0]

    @property
    def gate_or_default(self) -> GateThresholds:
        return self.gate or DEFAULT_GATE


# Format/framework tags per HF convention; the supported-architecture story lives in
# the generated Hardware Requirements card section, not a GPU-name laundry list.
_NVFP4A16_TAGS = (
    "text-generation", "nvfp4", "nvfp4a16", "fp4", "e2m1", "weight-only",
    "compressed-tensors", "vllm", "quantized", "llm-compressor",
)

RECIPES: dict[str, Recipe] = {
    "qwen2_5_7b_instruct": Recipe(
        slug="qwen2_5_7b_instruct",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        quant_scheme="NVFP4A16",
        calib=Calib("HuggingFaceH4/ultrachat_200k", "train_sft", 512, 2048),
        eval=Eval(
            accuracy_tasks=(
                # flexible-extract, NOT strict-match: in chat mode the model answers
                # conversationally rather than in gsm8k's "#### <n>" completion format,
                # so strict-match under-extracts (measured on metal: baseline collapsed
                # to ~0.18 vs the true ~0.80). flexible-extract pulls the final number
                # out of chat-formatted output. The gate is apples-to-apples either way,
                # but the absolute score must reflect real chat-mode capability.
                ("gsm8k", "exact_match,flexible-extract"),
                ("arc_challenge", "acc,none"),
                ("hellaswag", "acc,none"),
                ("winogrande", "acc,none"),
                ("mmlu", "acc,none"),
            ),
            perplexity=("wikitext", "word_perplexity"),
            mode="chat",              # the re-cert upgrade: eval as the model is used
            gen_kwargs=None,
            system_prompt=None,
            prompt_prefix=None,
        ),
        gate=None,
        tags=_NVFP4A16_TAGS + ("qwen2", "qwen2.5", "conversational"),
    ),
    "r1_distill_qwen_7b": Recipe(
        slug="r1_distill_qwen_7b",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        quant_scheme="NVFP4A16",
        calib=Calib("HuggingFaceH4/ultrachat_200k", "train_sft", 512, 2048),
        eval=Eval(
            accuracy_tasks=(
                ("aime24_avg", "exact_match,avg"),
                ("aime25_avg", "exact_match,avg"),
                # math_verify (from lm-eval[math]) not exact_match: R1 emits
                # <think>...\boxed{} answers that the Minerva-prompt exact_match
                # extractor misses (scored 0.0000 both sides on metal); math_verify
                # is a robust symbolic verifier and read 0.84 in the same run.
                ("minerva_math500", "math_verify,none"),
                ("gpqa_diamond_cot_zeroshot", "exact_match,flexible-extract"),
            ),
            perplexity=("wikitext", "word_perplexity"),
            mode="chat",
            # DeepSeek card: temperature 0.6, top_p 0.95, no system prompt, and the
            # response must begin with "<think>\n". do_sample=True is required or the
            # task-YAML greedy default stays in effect. max_gen_toks matches aime24.yaml.
            gen_kwargs={"temperature": 0.6, "top_p": 0.95, "do_sample": True, "max_gen_toks": 32768},
            system_prompt=None,
            # prompt_prefix stays None ON PURPOSE. This checkpoint's own chat_template
            # ends with `{% if add_generation_prompt %}...<think>\n{% endif %}`, and
            # lm-eval sets add_generation_prompt=True in chat mode - so "<think>\n" is
            # ALREADY forced onto every generation prompt. Setting a prefix here would
            # DOUBLE it. The prompt_prefix mechanism remains (validated, unwired) for a
            # future checkpoint whose template does NOT auto-append. (Verified against
            # the live deepseek-ai/DeepSeek-R1-Distill-Qwen-7B chat_template.)
            prompt_prefix=None,
            # avg@16 on the two AIME sets. AIME is n=30, temperature 0.6, single-sample
            # combined stderr ~0.18 (~11-question gate threshold) - "nearly decorative".
            # Averaging K=16 samples per question shrinks the sampling-noise term ~1/sqrt(K)
            # toward the between-question floor, giving the significance gate real teeth.
            # WIRED via the assay-owned aime24_avg/aime25_avg tasks (lm_eval_tasks/) +
            # evaluate.py repeats injection; stock lm-eval 0.4.12 take_first cannot avg@K.
            # K applies to AIME only: minerva (n=500) and gpqa (n=198) are already
            # well-powered by their large n.
            repeats={"aime24_avg": 16, "aime25_avg": 16},
        ),
        # Significance-gated, not point-gated: tiny generative sets (aime n=30) make
        # max_single_drop_pts structurally wrong (one question = 3.3pt > 2pt). k=2 is
        # the standard "beyond the error bars" bar. ppl stays the strict weight-
        # preservation backstop (ratio-gated, low variance - R1 quant was +0.47%).
        gate=GateThresholds(
            min_mean_retention=None,
            max_single_drop_pts=None,
            max_ppl_increase=0.03,
            k_stderr=2.0,
        ),
        tags=_NVFP4A16_TAGS + ("deepseek", "deepseek-r1", "reasoning", "conversational"),
    ),
    # --- TEMPLATE: copy this block, rename the slug, edit the fields. ---
    # Adding a model = adding one Recipe here (reviewable diff). Keep metric keys
    # FULLY-QUALIFIED (include the lm-eval filter suffix). Set mode="chat" for
    # instruct/reasoning models. Leave gate=None to inherit DEFAULT_GATE.
    "template_example": Recipe(
        slug="template_example",
        base_model="org/YourModel-Instruct",
        quant_scheme="NVFP4A16",
        calib=Calib("HuggingFaceH4/ultrachat_200k", "train_sft", 512, 2048),
        eval=Eval(
            accuracy_tasks=(("mmlu", "acc,none"),),
            perplexity=("wikitext", "word_perplexity"),
            mode="chat",
            gen_kwargs=None,
            system_prompt=None,
            prompt_prefix=None,
        ),
        gate=None,
        tags=_NVFP4A16_TAGS,
    ),
}


def get_recipe(slug: str) -> Recipe:
    try:
        return RECIPES[slug]
    except KeyError:
        raise KeyError(
            f"unknown recipe {slug!r}; valid slugs: {', '.join(sorted(RECIPES))}"
        ) from None


def validate_recipe(recipe: Recipe) -> None:
    """Cheap structural validation - catch a malformed recipe in milliseconds,
    before any paid GPU work. Does NOT hit the network (task existence is proven
    on metal)."""
    if not recipe.base_model:
        raise ValueError(
            f"recipe {recipe.slug!r}: base_model is required (got empty/unset - check "
            "ASSAY_BASE_MODEL if this was an env override)")
    if not recipe.quant_scheme:
        raise ValueError(
            f"recipe {recipe.slug!r}: quant_scheme is required (got empty/unset - check "
            "ASSAY_QUANT_SCHEME if this was an env override)")
    if not recipe.calib.dataset:
        raise ValueError(
            f"recipe {recipe.slug!r}: calib.dataset is required (got empty/unset - check "
            "ASSAY_CALIB_DATASET if this was an env override)")
    if not recipe.calib.split:
        raise ValueError(
            f"recipe {recipe.slug!r}: calib.split is required (got empty/unset - check "
            "ASSAY_CALIB_SPLIT if this was an env override)")
    ev = recipe.eval
    if not ev.accuracy_tasks:
        raise ValueError(f"recipe {recipe.slug!r}: needs at least one accuracy task")
    if ev.mode not in _VALID_MODES:
        raise ValueError(f"recipe {recipe.slug!r}: mode must be one of {_VALID_MODES}, got {ev.mode!r}")
    if ev.prompt_prefix is not None and ev.mode != "chat":
        raise ValueError(
            f"recipe {recipe.slug!r}: prompt_prefix requires mode='chat' (it is a "
            "forced assistant-turn prefix rendered through the chat template)")
    for task, metric in ev.accuracy_tasks:
        if not task or not metric:
            raise ValueError(f"recipe {recipe.slug!r}: empty task/metric in accuracy_tasks: {(task, metric)!r}")
        if "," not in metric:
            raise ValueError(
                f"recipe {recipe.slug!r}: accuracy metric key {metric!r} for task {task!r} "
                "must be fully qualified with its lm-eval filter suffix (e.g. 'acc,none', "
                "'exact_match,strict-match')")
    for rtask, rk in ev.repeats.items():
        if rtask not in recipe.accuracy_task_names:
            raise ValueError(
                f"recipe {recipe.slug!r}: repeats key {rtask!r} is not one of the "
                f"recipe's accuracy tasks {recipe.accuracy_task_names}")
        if not isinstance(rk, int) or rk < 1:
            raise ValueError(
                f"recipe {recipe.slug!r}: repeats[{rtask!r}] must be an int >= 1, got {rk!r}")
    if ev.perplexity is not None:
        ptask, pmetric = ev.perplexity
        if not ptask or not pmetric:
            raise ValueError(f"recipe {recipe.slug!r}: perplexity must be (task, metric) or None")
    if not recipe.tags:
        raise ValueError(f"recipe {recipe.slug!r}: at least one tag is required for the model card")
    # A criteria-free gate is a false-certification hazard: an EXPLICIT all-None
    # GateThresholds is truthy, so gate_or_default does NOT substitute DEFAULT_GATE,
    # and evaluate_gate then fires zero criteria and PASSES any quantization -> the
    # checkpoint publishes and is DOI-minted on no evidence. gate=None is the supported
    # "use DEFAULT_GATE" path; a criteria-free explicit gate must be rejected here.
    if recipe.gate is not None and not any((
            recipe.gate.min_mean_retention is not None,
            recipe.gate.max_single_drop_pts is not None,
            recipe.gate.max_ppl_increase is not None,
            recipe.gate.k_stderr is not None)):
        raise ValueError(
            f"recipe {recipe.slug!r}: gate has no active criteria (all thresholds are "
            "None). A criteria-free gate would PASS any quantization and publish a false "
            "certification. Set at least one threshold, or use gate=None for DEFAULT_GATE.")
