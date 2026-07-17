import pytest

from assay.config import load_config
from assay.job import run_job, Deps


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
        quantize=lambda cfg, mp, out, hb: calls.append("quantize") or out,
        run_eval=lambda mp, tasks, hb, gmu: calls.append(f"eval:{mp}:{gmu}") or {},
        parse=lambda raw, acc, ppl: {},
        gate=lambda base, quant, acc, ppl: calls.append("gate") or _passing_gate(),
        publish=lambda cfg, out, res, hb: calls.append("publish") or True,
        terminate=lambda: calls.append("terminate"),
    )


def test_happy_path_runs_all_stages_in_order(tmp_path):
    # ASSAY_HEARTBEAT override: Heartbeat.__init__ eagerly os.makedirs()'s its
    # dirname, and load_config({})'s default heartbeat_path is the pod-only
    # /runpod-volume/... mount -- not writable (or even creatable) on a dev
    # box with no such volume. Production is fine (RunPod pre-mounts the
    # volume); the unit test needs a real writable path, same as every other
    # module's tests use tmp_path for Heartbeat.
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    run_job(cfg, {}, _mk_deps(calls))
    assert calls.index("quantize") < calls.index("gate") < calls.index("publish")
    assert calls[-1] == "terminate"


def test_run_eval_receives_configured_gpu_mem_util(tmp_path):
    # run_job must thread cfg.gpu_mem_util into BOTH eval calls -- the knob
    # exists so an operator can squeeze the eval engine under residual GPU
    # usage (vLLM v1 hard-fails startup when device-wide free memory is below
    # gpu_mem_util * total); a call site that silently falls back to the
    # run_eval default would make the env override a no-op.
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_GPU_MEM_UTIL": "0.70",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    run_job(cfg, {}, _mk_deps(calls))
    evals = [c for c in calls if c.startswith("eval:")]
    assert len(evals) == 2
    assert all(c.endswith(":0.7") for c in evals)


def test_terminate_always_called_on_stage_error(tmp_path):
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    deps = _mk_deps(calls)
    deps = deps._replace(quantize=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        run_job(cfg, {}, deps)
    except RuntimeError:
        pass
    assert calls[-1] == "terminate"  # teardown fired despite the crash


def test_run_job_writes_durable_traceback_on_failure(tmp_path):
    # A stage crash must leave the traceback + a disk/mem snapshot on the volume
    # BEFORE teardown deletes the pod -- the fix for the "fails once, tells you
    # nothing" forensics gap. The exception must still propagate and teardown fire.
    art = tmp_path / "art"
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": str(art),
        "ASSAY_OUTPUT_DIR": str(art / "checkpoint"),
        "ASSAY_HEARTBEAT": str(art / "heartbeat.log"),
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    deps = _mk_deps(calls)._replace(
        quantize=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk boom")))
    with pytest.raises(RuntimeError, match="disk boom"):
        run_job(cfg, {}, deps)
    tb = (art / "traceback.txt").read_text()
    assert "RuntimeError" in tb and "disk boom" in tb
    assert "root_disk" in tb and "MemAvailable" in tb  # resource snapshot appended
    assert calls[-1] == "terminate"  # teardown still fired after forensics


def test_terminate_called_when_heartbeat_construction_fails(tmp_path, monkeypatch):
    # Heartbeat.__init__ eagerly os.makedirs()'s its path; if the volume isn't
    # mounted (or hiccups) that raises OSError. Heartbeat() is constructed INSIDE
    # run_job's try block, but `hb` is still None when the constructor raises, so
    # the finally block's `if hb is not None` guard skips the teardown emit.
    # deps.terminate() must still fire unconditionally -- an orphaned pod burns
    # money regardless of why the heartbeat couldn't be constructed.
    class _ExplodingHeartbeat:
        def __init__(self, *a, **k):
            raise OSError("heartbeat volume not mounted")

    monkeypatch.setattr("assay.job.Heartbeat", _ExplodingHeartbeat)
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    try:
        run_job(cfg, {}, _mk_deps(calls))
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError from Heartbeat construction to propagate")
    assert calls == ["terminate"]


def test_terminate_called_when_teardown_emit_fails(tmp_path, monkeypatch):
    # A heartbeat that works fine during the run but fails to emit on
    # teardown (disk full mid-run, mount dropped during teardown) must never
    # prevent deps.terminate() from running.
    class _FlakyTeardownHeartbeat:
        def __init__(self, *a, **k):
            self._step = 0

        def emit(self, stage, message=""):
            if stage == "teardown":
                raise OSError("disk full")
            self._step += 1

    monkeypatch.setattr("assay.job.Heartbeat", _FlakyTeardownHeartbeat)
    cfg = load_config({
        "ASSAY_HEARTBEAT": str(tmp_path / "heartbeat.log"),
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })
    calls = []
    result = run_job(cfg, {}, _mk_deps(calls))
    assert result.passed
    assert calls[-1] == "terminate"  # teardown emit blew up but terminate still ran


def _run_job_with_real_parse_and_gate(tmp_path, *, gate_passes: bool):
    """Wire run_job with run_eval returning realistic raw lm-eval dicts and the
    REAL parse_results/evaluate_gate (not fakes), so the I1/I2 durable-artifact
    writes (json.dumps of raw eval output, render_delta_table of a real
    GateResult) are exercised end-to-end, for both a passing and a failing gate."""
    from assay.evaluate import parse_results

    artifacts_dir = tmp_path / "artifacts"
    output_dir = artifacts_dir / "checkpoint"
    cfg = load_config({
        "ASSAY_ARTIFACTS_DIR": str(artifacts_dir),
        "ASSAY_OUTPUT_DIR": str(output_dir),
        "ASSAY_HEARTBEAT": str(artifacts_dir / "heartbeat.log"),
        "ASSAY_ACC_TASKS": "gsm8k",
        "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16",
    })

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

    def _run_eval(model_path, tasks, hb, gpu_mem_util):
        calls.append(f"eval:{model_path}")
        return quant_raw if model_path == "nvfp4-dir" else base_raw

    deps = Deps(
        quantize=lambda cfg, mp, out, hb: calls.append("quantize") or "nvfp4-dir",
        run_eval=_run_eval,
        parse=parse_results,
        gate=None,  # falls back to the real evaluate_gate, same as production wiring
        publish=lambda cfg, out, res, hb: calls.append("publish") or True,
        terminate=lambda: calls.append("terminate"),
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
