import pytest

from assay.config import load_config
from assay.job import run_job, Deps

_INFRA_KWARGS = {"persist_path", "watchdog_factory"}


def _science(kw: dict) -> dict:
    """Drop the per-run infra plumbing so a test can assert on the eval SETTINGS
    (which baseline and quantized must share) without the persist paths (which
    differ per eval by design)."""
    return {k: v for k, v in kw.items() if k not in _INFRA_KWARGS}


def _make_runcfg():
    """Minimal RunConfig from default recipe + dummy paths for testing."""
    return load_config({
        "ASSAY_HEARTBEAT": "/tmp/heartbeat.log",
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "test/Model-NVFP4A16",
        "ASSAY_ARTIFACTS_DIR": "/tmp/artifacts",
        "ASSAY_OUTPUT_DIR": "/tmp/output",
        "ASSAY_WEIGHTS_PATH": "/tmp/weights",
    })


def _passing_gate(*_a, **_k):
    from assay.gate import evaluate_gate
    base = {
        "gsm8k": {"metric": "exact_match,strict-match", "value": 0.80},
        "wikitext": {"metric": "word_perplexity", "value": 10.0},
    }
    good = {
        "gsm8k": {"metric": "exact_match,strict-match", "value": 0.799},
        "wikitext": {"metric": "word_perplexity", "value": 10.02},
    }
    return evaluate_gate(base, good, ("gsm8k",), "wikitext")


def _mk_deps(calls):
    return Deps(
        quantize=lambda recipe, mp, out, hb: calls.append("quantize") or out,
        run_eval=lambda mp, tasks, hb, gmu, **kw: calls.append(f"eval:{mp}:{gmu}") or {},
        parse=lambda raw, acc, ppl, **kw: {},
        gate=lambda base, quant, acc, ppl, thr: calls.append("gate") or _passing_gate(),
        publish=lambda runcfg, out, res, hb: calls.append("publish") or True,
    )


def test_happy_path_runs_all_stages_in_order(tmp_path):
    # ASSAY_HEARTBEAT override: Heartbeat.__init__ eagerly os.makedirs()'s its
    # dirname, and load_config({})'s default heartbeat_path is the pod-only
    # /runpod-volume/... mount - not writable (or even creatable) on a dev
    # box with no such volume. Production is fine (RunPod pre-mounts the
    # volume); the unit test needs a real writable path, same as every other
    # module's tests use tmp_path for Heartbeat.
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    run_job(cfg, {}, _mk_deps(calls))
    assert calls.index("quantize") < calls.index("gate") < calls.index("publish")


def test_run_eval_receives_configured_gpu_mem_util(tmp_path):
    # run_job must thread cfg.gpu_mem_util into BOTH eval calls - the knob
    # exists so an operator can squeeze the eval engine under residual GPU
    # usage (vLLM v1 hard-fails startup when device-wide free memory is below
    # gpu_mem_util * total); a call site that silently falls back to the
    # run_eval default would make the env override a no-op.
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_GPU_MEM_UTIL": "0.70",
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    run_job(cfg, {}, _mk_deps(calls))
    evals = [c for c in calls if c.startswith("eval:")]
    assert len(evals) == 2
    assert all(c.endswith(":0.7") for c in evals)


def test_run_eval_receives_recipe_derived_chat_kwargs_identically(tmp_path):
    # eval_kwargs is derived once from recipe.eval (mode/gen_kwargs/system_prompt)
    # and must reach BOTH run_eval calls identically - baseline and quantized
    # must be evaluated under the same chat-mode/sampling settings, or a gate
    # delta could reflect a settings drift instead of the quantization.
    from assay.evaluate import assay_task_dir
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    seen = []

    def _run_eval(mp, tasks, hb, gmu, **kwargs):
        seen.append(kwargs)
        return {}

    deps = _mk_deps([])._replace(run_eval=_run_eval)
    run_job(cfg, {}, deps)
    assert len(seen) == 2
    assert _science(seen[0]) == _science(seen[1])
    # qwen2_5_7b_instruct (the default recipe): mode="chat", no sampling overrides,
    # no avg@K repeats. include_path/repeats forwarded per Task 2 (lm-eval
    # TaskManager + per-recipe repeats wiring): run_job always threads the
    # shipped task dir + recipe.eval.repeats ({} here) into both eval calls.
    assert _science(seen[0]) == {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        # Point-gated recipe (DEFAULT_GATE, no k_stderr): no per-item capture, so
        # log_samples stays off and no new failure surface appears (D7/F-025).
        "capture_per_item": False,
        "gen_kwargs": None,
        "system_instruction": None,
        "include_path": assay_task_dir(),
        "repeats": {},
        # ASSAY_TIER unset -> runcfg.tier is "cert" -> eval_limit stays None, but
        # run_job always threads limit=limit as its own kwarg (Task 6).
        "limit": None,
    }


def test_smoke_scales_pipeline_on_both_eval_calls(tmp_path):
    # SP1 unit 2's core promise: ASSAY_TIER=smoke scales the REAL pipeline down so a
    # cheap pre-burn smoke genuinely predicts the burn. This pins the junction that
    # connects the tier to behavior - WITHOUT it, a refactor dropping the tier-profile
    # application stays green while an operator's "cheap smoke" silently runs the full
    # avg@16 battery for hours on a rented GPU. limit MUST be 2 (not 1: n=1 yields an
    # "N/A" stderr that crashes a significance-gated recipe - see gate.py /
    # _pick_stderr) and gen_kwargs MUST override max_gen_toks=8, applied IDENTICALLY to
    # baseline and quantized.
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
        "ASSAY_TIER": "smoke",
    })
    assert cfg.tier == "smoke"
    seen = []

    def _run_eval(mp, tasks, hb, gmu, **kwargs):
        seen.append(kwargs)
        return {}

    run_job(cfg, {}, _mk_deps([])._replace(run_eval=_run_eval))
    assert len(seen) == 2
    assert _science(seen[0]) == _science(seen[1])  # baseline and quantized scaled identically
    assert seen[0]["limit"] == 2
    assert seen[0]["gen_kwargs"] == {"max_gen_toks": 8}


def test_smoke_preserves_recipe_gen_kwargs_and_overrides_max_gen_toks(tmp_path):
    # When a recipe carries its own gen_kwargs (sampling), smoke must MERGE, overriding
    # only max_gen_toks - not clobber the recipe's temperature/top_p, or the smoke path
    # would diverge from the real generative settings it is meant to rehearse.
    from dataclasses import replace
    from assay.recipes import RECIPES
    base = RECIPES["qwen2_5_7b_instruct"]
    recipe = replace(base, eval=replace(base.eval, gen_kwargs={"temperature": 0.6, "top_p": 0.95}))
    cfg = replace(_make_runcfg_writable(tmp_path), recipe=recipe, tier="smoke", eval_limit=2)
    seen = []
    run_job(cfg, {}, _mk_deps([])._replace(run_eval=lambda mp, t, hb, gmu, **kw: seen.append(kw) or {}))
    assert seen[0]["gen_kwargs"] == {"temperature": 0.6, "top_p": 0.95, "max_gen_toks": 8}


def test_dev_tier_scales_avg_k_and_gen_len(tmp_path):
    # dev tier: avg@8 (aime 16->8), max_gen_toks capped to 4096, limit 50 - applied
    # IDENTICALLY to baseline and quantized, and never RAISING a task's K.
    from dataclasses import replace
    from assay.recipes import RECIPES
    base = RECIPES["r1_distill_qwen_7b"]
    cfg = replace(_make_runcfg_writable(tmp_path), recipe=base, tier="dev", eval_limit=50)
    seen = []
    run_job(cfg, {}, _mk_deps([])._replace(
        run_eval=lambda mp, t, hb, gmu, **kw: seen.append(kw) or {}))
    assert seen[0]["limit"] == 50
    assert seen[0]["gen_kwargs"]["max_gen_toks"] == 4096
    assert seen[0]["repeats"] == {"aime24_avg": 8, "aime25_avg": 8}
    assert _science(seen[0]) == _science(seen[1])


def _make_runcfg_writable(tmp_path):
    """A RunConfig with tmp writable paths (Heartbeat eagerly makedirs its dir)."""
    return load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "ASSAY_OUTPUT_DIR": str(tmp_path / "checkpoint"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })


def test_default_deps_threads_dry_run_from_tier(tmp_path, monkeypatch):
    # default_deps is the real wiring (pragma: no cover) and its publish lambda binds
    # dry_run=(not runcfg.pristine) - the full provenance guard (Task 2). Tier alone
    # drives pristine here (dev/smoke are never pristine, cert with no overrides is),
    # so this still exercises the tier -> pristine -> dry_run path through the single
    # publish chokepoint. If it regresses to a bare dry_run=False (or drops the
    # kwarg), a smoke run whose 2-sample gate passes uploads a "certified" checkpoint
    # built from limit=2 numbers.
    # Pin it: non-cert tier -> dry_run True, cert tier -> dry_run False.
    import assay.publish as publish_mod
    from assay.job import default_deps
    captured = {}
    monkeypatch.setattr(publish_mod, "publish_if_passed",
                        lambda *a, **k: captured.update(k) or True)
    for tier in ("smoke", "cert"):
        cfg = load_config({
            "ASSAY_HEARTBEAT": str(tmp_path / "hb.log"),
            "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
            "ASSAY_TIER": tier,
        })
        # HF_TOKEN must be present: the publish lambda resolves it via require_secret.
        default_deps({"HF_TOKEN": "tok"}).publish(cfg, "out", object(), None)
        assert captured["dry_run"] is (tier != "cert")


def test_publish_dry_run_reflects_pristine(tmp_path, monkeypatch):
    # Positive chokepoint coverage (spec s8, the I-3 lesson): the single publish
    # chokepoint's dry_run must FOLLOW pristine, not tier - this is what lets a
    # calib override under cert tier (still pristine=False) get caught where Task
    # 1's tier-only gate would have missed it.
    import assay.job as job_mod
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    seen = {}

    def _fake_publish_if_passed(runcfg, out, result, token, hb, dry_run=False):
        seen["dry_run"] = dry_run
        return True

    monkeypatch.setattr("assay.publish.publish_if_passed", _fake_publish_if_passed)

    # pristine cert run -> real publish (dry_run False)
    cfg = _make_runcfg_writable(tmp_path)  # tier defaults cert, pristine True
    deps = job_mod.default_deps({"HF_TOKEN": "dummy-token"})
    deps.publish(cfg, cfg.output_dir, object(), None)
    assert seen["dry_run"] is False

    # non-pristine run -> forced dry-run
    from dataclasses import replace
    deps.publish(replace(cfg, pristine=False), cfg.output_dir, object(), None)
    assert seen["dry_run"] is True


def test_run_job_writes_durable_traceback_on_failure(tmp_path):
    # A stage crash must leave the traceback + a disk/mem snapshot on the volume
    # BEFORE teardown deletes the pod - the fix for the "fails once, tells you
    # nothing" forensics gap. The exception must still propagate and teardown fire.
    art = tmp_path / "art"
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": str(art),
        "ASSAY_OUTPUT_DIR": str(tmp_path / "checkpoint"),  # sibling, not nested under art
        "ASSAY_HEARTBEAT": str(art / "heartbeat.log"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    deps = _mk_deps(calls)._replace(
        quantize=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk boom")))
    with pytest.raises(RuntimeError, match="disk boom"):
        run_job(cfg, {}, deps)
    tb = (art / "traceback.txt").read_text()
    assert "RuntimeError" in tb and "disk boom" in tb
    assert "root_disk" in tb and "MemAvailable" in tb  # resource snapshot appended


def test_run_job_does_not_self_terminate(tmp_path):
    # Teardown is pod_entry.sh's job now (single EXIT trap): run_job must NOT
    # terminate, but it must still complete its forensics/artifact writes.
    art = tmp_path / "art"
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": str(art),
        "ASSAY_OUTPUT_DIR": str(tmp_path / "checkpoint"),  # sibling, not nested under art
        "ASSAY_HEARTBEAT": str(art / "heartbeat.log"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    run_job(cfg, {}, _mk_deps(calls))
    assert "terminate" not in calls
    assert (art / "delta-table.md").exists()  # forensics/artifacts still written


def _run_job_with_real_parse_and_gate(tmp_path, *, gate_passes: bool):
    """Wire run_job with run_eval returning realistic raw lm-eval dicts and the
    REAL parse_results/evaluate_gate (not fakes), so the I1/I2 durable-artifact
    writes (json.dumps of raw eval output, render_delta_table of a real
    GateResult) are exercised end-to-end, for both a passing and a failing gate."""
    from dataclasses import replace

    from assay.evaluate import parse_results

    artifacts_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "checkpoint"  # sibling of artifacts_dir, not nested under it
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": str(artifacts_dir),
        "ASSAY_OUTPUT_DIR": str(output_dir),
        "ASSAY_HEARTBEAT": str(artifacts_dir / "heartbeat.log"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    # Trim the recipe's eval battery to gsm8k only: the raw dicts below carry
    # just gsm8k + wikitext, and the task list is recipe-owned now (the old
    # ASSAY_ACC_TASKS env knob was retired with the Config -> RunConfig split).
    cfg = replace(cfg, recipe=replace(
        cfg.recipe, eval=replace(
            cfg.recipe.eval,
            accuracy_tasks=(("gsm8k", "exact_match,strict-match"),),
        ),
    ))

    base_raw = {"results": {
        "gsm8k": {"exact_match,strict-match": 0.80},
        "wikitext": {"word_perplexity": 10.0},
    }}
    if gate_passes:
        quant_raw = {"results": {
            "gsm8k": {"exact_match,strict-match": 0.799},
            "wikitext": {"word_perplexity": 10.02},
        }}
    else:
        quant_raw = {"results": {
            "gsm8k": {"exact_match,strict-match": 0.50},
            "wikitext": {"word_perplexity": 10.0},
        }}

    calls = []

    def _run_eval(model_path, tasks, hb, gpu_mem_util, **kwargs):
        calls.append(f"eval:{model_path}")
        return quant_raw if model_path == "nvfp4-dir" else base_raw

    deps = Deps(
        quantize=lambda recipe, mp, out, hb: calls.append("quantize") or "nvfp4-dir",
        run_eval=_run_eval,
        parse=parse_results,
        gate=None,  # falls back to the real evaluate_gate, same as production wiring
        publish=lambda runcfg, out, res, hb: calls.append("publish") or True,
    )

    result = run_job(cfg, {}, deps)
    assert result.passed is gate_passes
    return artifacts_dir, result


def test_durable_artifacts_written_on_gate_pass(tmp_path):
    # I1/I2: eval JSONs + delta table land in artifacts_dir (never output_dir),
    # so they survive the run and never ride along into a published checkpoint.
    artifacts_dir, _ = _run_job_with_real_parse_and_gate(tmp_path, gate_passes=True)
    assert (artifacts_dir / "eval-baseline.json").exists()
    assert (artifacts_dir / "eval-nvfp4.json").exists()
    assert (artifacts_dir / "delta-table.md").exists()


def test_durable_artifacts_written_on_gate_fail(tmp_path):
    # I1's whole point: a gate FAIL must still leave the eval numbers + delta
    # table on disk, not just in memory where a FAIL run loses them forever.
    artifacts_dir, result = _run_job_with_real_parse_and_gate(tmp_path, gate_passes=False)
    assert not result.passed
    assert (artifacts_dir / "eval-baseline.json").exists()
    assert (artifacts_dir / "eval-nvfp4.json").exists()
    assert (artifacts_dir / "delta-table.md").exists()
    assert "FAIL" in (artifacts_dir / "delta-table.md").read_text()


def test_run_identity_lines_names_version_recipe_and_tasks():
    from assay import __version__, job
    runcfg = _make_runcfg()
    lines = job.run_identity_lines(runcfg, {"ASSAY_BUILD_SHA": "abc1234"})
    text = "\n".join(lines)
    assert __version__ in text
    assert "abc1234" in text
    assert runcfg.recipe.slug in text
    assert "gsm8k" in text            # an accuracy task
    assert "exact_match,flexible-extract" in text  # its fully-qualified metric


def test_run_eval_receives_distinct_persist_paths_and_watchdog_factory(tmp_path):
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "hb.log"),
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "art"),
        "ASSAY_OUTPUT_DIR": str(tmp_path / "out"),
        "ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    seen = []
    run_job(cfg, {"ASSAY_RAW_LOG": "/tmp/raw.log"},
            _mk_deps([])._replace(run_eval=lambda mp, t, hb, gmu, **kw: seen.append(kw) or {}))
    assert len(seen) == 2
    persists = {kw["persist_path"] for kw in seen}
    assert len(persists) == 2  # baseline and quant persist to DIFFERENT files
    assert all(str(tmp_path / "art") in p for p in persists)
    assert all(callable(kw["watchdog_factory"]) for kw in seen)
    # NOTE: run_job does NOT pass raw_log_path to run_eval - the _wd_factory closure
    # already closes over raw_log (from env["ASSAY_RAW_LOG"]) via build_eval_watchdog,
    # so run_eval has no raw_log_path param (dropped in the Task 3 review).


def test_run_job_keys_pairing_off_k_stderr(tmp_path):
    """D7/F-025: per-item capture and collection run EXACTLY when the recipe's gate
    consults an SE. The R1 recipe (k_stderr) gets capture_per_item/collect_items
    True; the Qwen point gate gets False - so Qwen cert runs gain no new hard-fail
    surface (and no log_samples payload) for a test their gate never runs."""
    seen = {}

    def _deps():
        return Deps(
            quantize=lambda recipe, mp, out, hb: out,
            run_eval=lambda mp, tasks, hb, gmu, **kw:
                seen.setdefault("eval", []).append(kw.get("capture_per_item")) or {},
            parse=lambda raw, acc, ppl, **kw:
                seen.setdefault("parse", []).append(kw.get("collect_items")) or {},
            gate=lambda base, quant, acc, ppl, thr: _passing_gate(),
            publish=lambda runcfg, out, res, hb: True,
        )

    for recipe, expected in (("r1_distill_qwen_7b", True), ("qwen2_5_7b_instruct", False)):
        seen.clear()
        cfg = load_config({
            "ASSAY_RECIPE": recipe,
            "ASSAY_HEARTBEAT": str(tmp_path / "hb.log"),
            "ASSAY_WEIGHTS_PATH": "/tmp/weights",
            "ASSAY_CHECKPOINT_REPO": "test/Model-NVFP4A16",
            "ASSAY_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "ASSAY_OUTPUT_DIR": str(tmp_path / "output"),
        })
        run_job(cfg, {}, _deps())
        assert seen["eval"] == [expected, expected], recipe   # both eval sides
        assert seen["parse"] == [expected, expected], recipe  # both parse calls
