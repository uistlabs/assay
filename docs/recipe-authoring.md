# Authoring an assay recipe

A recipe (`src/assay/recipes.py`) is code-as-config: it is the human's encoding of the
base model's evaluation protocol. Before a paid run, walk this checklist against the
base model's Hugging Face card. Encode each item in the recipe, or note the deliberate
deviation in a comment.

- **Sampling.** Does the card specify temperature / top_p / greedy? Set `eval.gen_kwargs`
  to match. A reasoning model evaluated greedily (or a greedy model sampled) is a
  different distribution than the one the card reports.
- **avg@K.** Does the card report avg@K (or pass@K) on any small generative benchmark
  (e.g. AIME, n=30)? A single-sample score there is dominated by noise. Set
  `eval.repeats={task: K}` and use the assay-owned avg@K task variant (see
  `src/assay/lm_eval_tasks/`); stock lm-eval single-samples.
- **Chat template.** Instruct/reasoning models must be evaluated in chat mode
  (`eval.mode = "chat"`). Check whether the template auto-appends a reasoning prefix
  (e.g. `<think>`); if so, leave `eval.prompt_prefix = None` or it doubles.
- **System prompt.** Some cards require NO system prompt (R1). Set `eval.system_prompt`
  accordingly.
- **Task + metric keys.** Verify each `(task, metric)` pair against the pinned lm-eval
  wheel (unzip + grep) - an extractor mismatch scores 0.0 silently (minerva needed
  `math_verify`, not `exact_match`).
- **Gate mode.** Choose a point gate (large well-powered MC sets) or the significance
  gate (`k_stderr`, for small high-variance generative sets). The gate must match the
  battery's statistical power.
