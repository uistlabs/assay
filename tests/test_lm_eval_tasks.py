import pytest

from assay.lm_eval_tasks import utils


def test_passthrough_keeps_all_k():
    # stock take_first would reduce each doc to r[0]; the avg variant must keep all K
    resps = [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    out = utils.passthrough(resps, docs=[{}, {}])
    assert [list(r) for r in out] == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]


def test_average_scores_is_fraction_correct():
    assert utils.average_scores([1, 1, 1, 0]) == {"exact_match": 0.75}
    assert utils.average_scores([0, 0]) == {"exact_match": 0.0}
    assert utils.average_scores([1]) == {"exact_match": 1.0}


def test_average_scores_empty_is_zero_not_crash():
    # defensive: an empty sample list must not ZeroDivisionError mid-run
    assert utils.average_scores([]) == {"exact_match": 0.0}


def test_avg_process_results_real_extractor():
    # Requires the real AIME extractor; skipped where lm-eval is absent (local dev).
    pytest.importorskip("lm_eval.tasks.aime.utils")
    from assay.lm_eval_tasks import utils
    doc = {"Problem": "x", "Answer": 42}
    # 3 correct boxed answers, 1 wrong => avg@4 = 0.75
    results = [
        "reasoning ... \\boxed{42}",
        "\\boxed{42}",
        "the answer is $42$",
        "\\boxed{41}",
    ]
    assert utils.avg_process_results(doc, results) == {"exact_match": 0.75}


def test_avg_process_results_harness_nested_shape():
    # The real lm-eval shape into process_results for a single-instance-per-doc task is
    # [[s1, ..., sK]] (one filtered list per doc); _as_samples must unwrap it. Same
    # 3-of-4-correct inputs as the flat test -> avg@4 = 0.75.
    pytest.importorskip("lm_eval.tasks.aime.utils")
    from assay.lm_eval_tasks import utils
    doc = {"Problem": "x", "Answer": 42}
    results = [["reasoning ... \\boxed{42}", "\\boxed{42}",
                "the answer is $42$", "\\boxed{41}"]]
    assert utils.avg_process_results(doc, results) == {"exact_match": 0.75}


def test_avg_tasks_register_under_include_path():
    # Guards the cross-root include pitfall: the inlined (self-contained) YAMLs must
    # appear in the task index when loaded via include_path (how Task 2 loads them).
    pytest.importorskip("lm_eval")
    import os
    import assay
    from lm_eval.tasks import TaskManager
    task_dir = os.path.join(os.path.dirname(assay.__file__), "lm_eval_tasks")
    tm = TaskManager(include_path=task_dir)
    assert {"aime24_avg", "aime25_avg"} <= set(tm.all_tasks)


def test_avgk_repeats_yields_k_instances_per_doc():
    # The load-bearing statistical feature (avg@16) is today proven only on a paid
    # metal burn. Prove on CPU that task.set_config("repeats", K) - the exact call
    # evaluate._eval_child makes for a repeats-override task - makes lm-eval run K
    # generations for a single doc, and that the assay-owned "avg" passthrough filter
    # keeps all K (stock take_first would collapse them to 1, silently breaking
    # avg@K). Retires the v0.4.1 lesson class: this mechanism must be pinned
    # somewhere that runs on every CI pass, not just discovered mid-burn.
    #
    # lm-eval 0.4.12 detail: build_all_requests() builds exactly ONE Instance per
    # doc for a generate_until task (instance.repeats == K carried in its metadata);
    # the K-fold clone-and-dispatch to the LM happens later, inside
    # evaluator.evaluate() (lm_eval/evaluator.py: `cloned_reqs.extend([req] *
    # req.repeats)`). So "K per doc" is not observable on task.instances - it is
    # observable in the per-doc resps/filtered_resps that simple_evaluate returns.
    # Drive the real entrypoint (simple_evaluate, same one _eval_child calls) with
    # lm-eval's built-in "dummy" model (returns a fixed string, no GPU/network) so
    # this stays a fast, deterministic CPU assertion.
    pytest.importorskip("lm_eval")
    import huggingface_hub.errors
    import requests
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager, get_task_dict
    from assay.evaluate import assay_task_dir

    # The aime24_avg task loads a non-vendored dataset (Maxwell-Jia/AIME_2024, 30
    # rows, unauthenticated, <1s on normal egress). This must stay a real network
    # call - a fresh CI runner has no HF cache, so forcing HF_HUB_OFFLINE here
    # would turn "no cache yet" into a permanent red build. Only a genuine
    # Hub-unreachable condition (outage, no egress) should skip, not fail.
    tm = TaskManager(include_path=assay_task_dir())
    try:
        td = get_task_dict(["aime24_avg"], tm)
        task = td["aime24_avg"]
        task.set_config("repeats", 4)

        raw = simple_evaluate(
            model="dummy",
            tasks=[task],
            limit=1,
            task_manager=tm,
            log_samples=True,
            verbosity="ERROR",
        )
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            huggingface_hub.errors.OfflineModeIsEnabled) as e:
        # Narrowly scoped to "Hub unreachable" only: DNS/egress-down surfaces as
        # requests.exceptions.ConnectionError/Timeout, and an explicit
        # HF_HUB_OFFLINE=1 surfaces as huggingface_hub.errors.OfflineModeIsEnabled.
        # A config bug (e.g. a typo'd dataset_path) reaches the Hub, gets a
        # definite answer, and raises datasets.exceptions.DatasetNotFoundError -
        # an OSError/FileNotFoundError subclass, deliberately NOT caught here, so
        # it propagates and fails the test loud instead of silently skipping.
        pytest.skip(f"AIME dataset unreachable (network/Hub down): {e}")

    doc0 = raw["samples"]["aime24_avg"][0]
    # 4 repeats of a single doc -> 4 raw generations dispatched to the LM.
    assert len(doc0["resps"][0]) == 4, f"expected 4 repeats, got {doc0['resps']}"
    # The custom "avg" filter (utils.passthrough) must keep all 4 post-filter -
    # this is the exact bug class stock take_first would reintroduce.
    assert len(doc0["filtered_resps"][0]) == 4, (
        f"expected passthrough to keep all 4, got {doc0['filtered_resps']}")
