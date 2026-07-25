"""Tier-1 structural smoke: prove the eval->gate path is sound WITHOUT a GPU, a model,
or the network, in the PROD IMAGE. Runs as a pytest, a Dockerfile build step, and an
in-pod startup step (assay.job.main) - one path, three call sites, so 'green in the
test' and 'works on metal' cannot diverge (the v0.4.1 lesson).

Coupled to lm-eval 0.4.12's config loader: `lm_eval.tasks._yaml_loader.load_yaml`
resolves `!function` tags by importing the referenced module via a file path
(`_import_func_in_yml` -> `_load_module_with_cache` -> `importlib.util.spec_from_file_location`),
giving the callable a PATH __module__ - exactly the object that crashed the parent's
spawn-unpickle. VERIFIED against the installed 0.4.12 wheel
(.venv/lib/python3.12/site-packages/lm_eval/tasks/_yaml_loader.py): there is no
`lm_eval.utils.load_yaml_config` in this version (utils.py has no such name; it only
defines the `!function` YAML constructor `import_function`, which the real task-build
path does NOT use - `_yaml_loader.py` installs its own `!function` constructor via
`yaml.add_constructor`). The real, used-in-production entry point is
`lm_eval.tasks._yaml_loader.load_yaml(path, resolve_func=True)` - called positionally,
not `load_yaml_config(yaml_path=...)` - a "pure data-loading helper" (its own
docstring) with no task/group/tag semantics and no dataset/network access; it is the
exact function `_factory.py`'s `_load_full_config` calls before a Task is ever built.
Confirmed live against both shipped YAMLs: returns a dict keyed by the YAML's own top-
level keys (task/process_results/filter_list/metric_list/...), `!function utils.X`
resolved to a real callable whose `__module__` is the absolute file path of
lm_eval_tasks/utils.py (not `lm_eval.tasks...`, since our task dir lives outside
lm_eval's own `tasks/` tree) - confirming the crash mechanism this module guards
against. A future lm-eval bump is a deliberate forward-test event (v0.5.x SP5), and
this failing loudly at build time is the correct signal, not a bug. ASCII only
(reaches the build log)."""
from __future__ import annotations

import glob
import multiprocessing
import os
import sys


def _assay_task_yamls() -> list[str]:
    from assay.evaluate import assay_task_dir
    return sorted(glob.glob(os.path.join(assay_task_dir(), "*_avg.yaml")))


def _roundtrip_child(conn, yaml_paths: list[str], stub_results: dict) -> None:
    """Runs in a fresh SPAWN interpreter, mirroring _eval_child: this process imports the
    external task functions BY PATH (registering their path-module HERE, as the real eval
    child does), embeds them in a raw-shaped payload, sanitizes, and sends. The parent
    NEVER imported those path-modules - so if _sanitize_raw fails to strip the callables,
    the parent's recv() raises ModuleNotFoundError, exactly the production crash."""
    try:
        from lm_eval.tasks._yaml_loader import load_yaml  # noqa: PLC0415
        from assay.evaluate import _sanitize_raw  # noqa: PLC0415

        configs = {}
        for path in yaml_paths:
            cfg = load_yaml(path)  # resolves !function -> path-named callables, no network
            task = cfg["task"]
            # Exercise the extractor on ONE synthetic sample: a boxed answer that must
            # score 1 -> a finite mean. Catches a broken process_results (the class that
            # made minerva read 0.0000 both sides in RUN 2).
            pr = cfg["process_results"]
            doc = {"Problem": "x", "Answer": "42"}
            scored = pr(doc, [["The answer is \\boxed{42}."]])
            if not scored or not all(v == v for v in scored.values()):  # NaN-safe finite check
                raise RuntimeError(f"process_results for {task} returned {scored!r}")
            # Embed the real path-named callables in the raw-shaped payload, where lm-eval
            # puts them, so _sanitize_raw must strip them for the parent recv to survive.
            configs[task] = {
                "filter_list": [{"filter": [{"filter_fn": cfg["filter_list"][0]["filter"][0]["filter_fn"]}]}],
                "process_results": pr,
            }
        raw = {"results": stub_results, "configs": configs}
        conn.send(("ok", _sanitize_raw(raw)))
    except BaseException as exc:  # noqa: BLE001
        import traceback
        conn.send(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def _check_recipe(recipe) -> None:
    from assay.recipes import validate_recipe
    from assay.evaluate import parse_results
    from assay.gate import evaluate_gate, render_delta_table

    validate_recipe(recipe)
    ev = recipe.eval

    # Only the assay-owned avg tasks carry external path-named callables; those are the
    # ones that must survive the spawn boundary. Match them to the shipped YAMLs by name.
    all_yamls = {os.path.splitext(os.path.basename(p))[0]: p for p in _assay_task_yamls()}
    avg_paths = [all_yamls[t] for t in recipe.accuracy_task_names if t in all_yamls]

    # Synthetic, FINITE stub numbers for every task in the recipe, so parse_results +
    # evaluate_gate + render_delta_table run against this recipe's real (task, metric)
    # mapping and real GateThresholds - catching the v0.4 None-threshold card-crash class.
    stub_results: dict = {}
    for task, metric in ev.accuracy_tasks:
        stub_results[task] = {metric: 0.5, _stderr_of(metric): 0.01}
    if ev.perplexity is not None:
        ptask, pmetric = ev.perplexity
        stub_results[ptask] = {pmetric + ",none": 9.5, pmetric + "_stderr,none": 0.1}

    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_roundtrip_child, args=(send, avg_paths, stub_results))
    proc.start()
    send.close()
    try:
        status, payload = recv.recv()  # ModuleNotFoundError here == the production crash
    finally:
        recv.close()
        proc.join()
    if status != "ok":
        raise RuntimeError(f"tier-1 spawn round-trip failed for {recipe.slug}:\n{payload}")

    base = parse_results(payload, ev.accuracy_tasks, ev.perplexity)
    quant = parse_results(payload, ev.accuracy_tasks, ev.perplexity)
    result = evaluate_gate(base, quant, recipe.accuracy_task_names,
                           recipe.perplexity_task_name, recipe.gate_or_default)
    render_delta_table(result, recipe.gate_or_default)  # must not raise for this recipe's gate


def _stderr_of(metric: str) -> str:
    """The stderr key lm-eval would emit beside `metric` (mirrors evaluate._stderr_key)."""
    if "," in metric:
        name, filt = metric.split(",", 1)
        return f"{name}_stderr,{filt}"
    return f"{metric}_stderr"


def tier1_structural(recipes=None) -> None:
    """Raise on the first structural failure across all real recipes (template excluded)."""
    from assay.recipes import RECIPES
    recipes = recipes if recipes is not None else [
        r for slug, r in RECIPES.items() if slug != "template_example"]
    for recipe in recipes:
        _check_recipe(recipe)


def main() -> int:
    try:
        tier1_structural()
    except BaseException as exc:  # noqa: BLE001
        print(f"FATAL: tier-1 smoke FAILED: {exc}", file=sys.stderr)
        return 1
    print("tier-1 smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
