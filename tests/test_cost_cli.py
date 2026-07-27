import json

from assay.cost.collect import RECORD_NAME, build_record, main, preflight_line, read_record

POD = {
    "id": "pod-1",
    "costPerHr": 0.99,
    "uptimeSeconds": 3600,
    "containerDiskInGb": 40,
    "volumeInGb": 0,
    "desiredStatus": "RUNNING",
    "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 5090"},
    "gpuCount": 1,
}

ENV = {
    "RUNPOD_POD_ID": "pod-1",
    "ASSAY_RECIPE": "qwen2_5_7b_instruct",
    "ASSAY_TIER": "cert",
    "ASSAY_CHECKPOINT_REPO": "uist-labs/Qwen2.5-7B-Instruct-NVFP4A16",
    "ASSAY_BUILD_SHA": "abc1234",
    "ASSAY_GPU_TYPE": "NVIDIA GeForce RTX 5090",
}


class FakeApi:
    def __init__(self, pod=None, gpu=None):
        self._pod, self._gpu = pod, gpu

    def get_pod(self, pod_id):
        return self._pod

    def get_gpu(self, gpu_id, gpu_quantity=1):
        return self._gpu


def test_build_record_has_the_spec_shape():
    rec = build_record(env=ENV, pod=POD, basis=None, gate_result="pass",
                       finalized=True, rc="0", now=1753449600.0)
    assert rec["schema_version"] == 1
    assert set(rec) >= {"schema_version", "run", "basis", "provider", "rates",
                        "marginal_usd", "outcome", "finalized"}
    assert rec["run"]["pod_id"] == "pod-1"
    assert rec["run"]["recipe"] == "qwen2_5_7b_instruct"
    assert rec["run"]["tier"] == "cert"
    assert rec["run"]["build_sha"] == "abc1234"
    assert rec["run"]["base_model"]  # resolved from the recipe
    assert rec["provider"]["cost_per_hr"] == 0.99
    assert rec["provider"]["gpu_display_name"] == "NVIDIA GeForce RTX 5090"
    assert rec["marginal_usd"]["gpu"] == 0.99
    assert rec["outcome"] == "pass"
    assert rec["finalized"] is True
    assert rec["basis"]["rate_source"] == "provider"


def test_build_record_marks_catalog_fallback():
    # The pod probe failed; the rate came from the gpuTypes catalog instead. The
    # reconciler must be able to tell a measured number from an inferred one.
    rec = build_record(env=ENV, pod=None, basis=None, gate_result=None,
                       finalized=False, catalog_price=0.99, now=1000.0)
    assert rec["basis"]["rate_source"] == "catalog"
    assert rec["provider"]["cost_per_hr"] == 0.99


def test_build_record_marks_unknown_when_no_rate_at_all():
    rec = build_record(env=ENV, pod=None, basis=None, gate_result=None,
                       finalized=False, catalog_price=None, now=1000.0)
    assert rec["basis"]["rate_source"] == "unknown"
    assert rec["marginal_usd"]["total"] == 0.0


def test_build_record_embeds_the_rate_table_version():
    # Self-contained record: a recomputation years later must be deterministic
    # even after RunPod changes prices.
    rec = build_record(env=ENV, pod=POD, basis=None, gate_result="pass",
                       finalized=True, rc="0", now=1000.0)
    assert rec["rates"]["rate_table_version"] == "2026-07-25"


def test_build_record_is_pure_with_respect_to_rc():
    # rc is an explicit parameter, not module state, because the host-side
    # reconciler re-runs this same function over stored records. Same inputs must
    # always give the same outcome, in any order.
    kw = dict(env=ENV, pod=POD, basis=None, gate_result="fail", finalized=True,
              now=1000.0)
    assert build_record(rc="0", **kw)["outcome"] == "gate_fail"
    assert build_record(rc="", **kw)["outcome"] == "infra_fail"
    assert build_record(rc="0", **kw)["outcome"] == "gate_fail"


def test_begin_writes_an_in_progress_record(tmp_path):
    rc = main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=POD), now=1000.0)
    assert rc == 0
    rec = read_record(str(tmp_path))
    assert rec["outcome"] == "in_progress"
    assert rec["finalized"] is False
    assert rec["basis"]["began_at_unix"] == 1000.0
    assert rec["basis"]["uptime_seconds_at_begin"] == 3600


def test_begin_succeeds_even_when_the_probe_fails(tmp_path):
    # A cost failure must never cost a run: begin returns 0 regardless.
    rc = main(["begin", str(tmp_path)], env=ENV,
              api=FakeApi(pod=None, gpu={"securePrice": 0.99}), now=1000.0)
    assert rc == 0
    assert read_record(str(tmp_path))["basis"]["rate_source"] == "catalog"


def test_finalize_sets_the_terminal_outcome(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("GATE PASSED\n")
    main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=POD), now=1000.0)
    rc = main(["finalize", str(tmp_path), "--rc", "0", "--log", str(log)],
              env=ENV, api=FakeApi(pod=POD), now=4600.0)
    assert rc == 0
    rec = read_record(str(tmp_path))
    assert rec["outcome"] == "pass"
    assert rec["finalized"] is True
    assert rec["marginal_usd"]["total"] > 0


def test_finalize_with_empty_rc_is_infra_fail(tmp_path):
    # pod_entry.sh passes ${rc:-}; empty means the EXIT trap fired before rc was
    # assigned, i.e. the pod died mid-run.
    log = tmp_path / "run.log"
    log.write_text("nothing\n")
    main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=POD), now=1000.0)
    main(["finalize", str(tmp_path), "--rc", "", "--log", str(log)],
         env=ENV, api=FakeApi(pod=POD), now=4600.0)
    assert read_record(str(tmp_path))["outcome"] == "infra_fail"


def test_finalize_with_gate_failed_marker_is_gate_fail(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("GATE FAILED\n")
    main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=POD), now=1000.0)
    main(["finalize", str(tmp_path), "--rc", "0", "--log", str(log)],
         env=ENV, api=FakeApi(pod=POD), now=4600.0)
    assert read_record(str(tmp_path))["outcome"] == "gate_fail"


def test_finalize_accrues_locally_when_the_pod_is_already_gone(tmp_path):
    # The pod terminated between begin and finalize. Provider truth is
    # unavailable, so uptime accrues from the basis: 3600 at begin + 1000 elapsed.
    log = tmp_path / "run.log"
    log.write_text("GATE PASSED\n")
    main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=POD), now=1000.0)
    main(["finalize", str(tmp_path), "--rc", "0", "--log", str(log)],
         env=ENV, api=FakeApi(pod=None), now=2000.0)
    rec = read_record(str(tmp_path))
    assert rec["provider"]["uptime_seconds"] == 4600.0
    assert rec["outcome"] == "pass"


def test_finalize_preserves_gpu_identity_when_the_pod_is_gone(tmp_path):
    # Reproduces the exact failure: begin probes a non-default pod (H200,
    # gpuCount 2, $3.99/hr) and the pod becomes unqueryable before finalize.
    # finalize must not silently rewrite the run's actual hardware to the
    # env/default RTX 5090 guess - gpu_display_name and gpu_count are the
    # predictor join keys the record exists to carry, and this is masked today
    # only because every current run happens to BE a 5090.
    h200_pod = {
        "id": "pod-h200",
        "costPerHr": 3.99,
        "uptimeSeconds": 3600,
        "containerDiskInGb": 40,
        "volumeInGb": 0,
        "desiredStatus": "RUNNING",
        "machine": {"gpuDisplayName": "NVIDIA H200"},
        "gpuCount": 2,
    }
    log = tmp_path / "run.log"
    log.write_text("GATE PASSED\n")
    main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=h200_pod), now=1000.0)
    begun = read_record(str(tmp_path))
    assert begun["provider"]["gpu_display_name"] == "NVIDIA H200"
    assert begun["provider"]["gpu_count"] == 2
    assert begun["basis"]["gpu_display_name"] == "NVIDIA H200"
    assert begun["basis"]["gpu_count"] == 2

    main(["finalize", str(tmp_path), "--rc", "0", "--log", str(log)],
         env=ENV, api=FakeApi(pod=None), now=2000.0)
    rec = read_record(str(tmp_path))
    assert rec["provider"]["gpu_display_name"] == "NVIDIA H200"
    assert rec["provider"]["gpu_count"] == 2
    assert rec["provider"]["cost_per_hr"] == 3.99
    assert rec["basis"]["rate_source"] == "provider"


def test_finalize_without_a_prior_begin_still_writes(tmp_path):
    # begin may have failed or timed out. finalize must still record what it can.
    log = tmp_path / "run.log"
    log.write_text("GATE PASSED\n")
    rc = main(["finalize", str(tmp_path), "--rc", "0", "--log", str(log)],
              env=ENV, api=FakeApi(pod=POD), now=2000.0)
    assert rc == 0
    assert read_record(str(tmp_path))["outcome"] == "pass"


def test_finalize_no_begin_and_pod_unqueryable_yields_unknown(tmp_path):
    # The honest "we cannot cost this run" signal: no prior begin AND the pod
    # probe returns None (terminated pod is unqueryable). rate_source is "unknown"
    # and totals are zero - recovery comes from account-level spend, not a
    # confidently wrong number.
    log = tmp_path / "run.log"
    log.write_text("GATE PASSED\n")
    rc = main(["finalize", str(tmp_path), "--rc", "0", "--log", str(log)],
              env=ENV, api=FakeApi(pod=None, gpu=None), now=2000.0)
    assert rc == 0
    rec = read_record(str(tmp_path))
    assert rec["basis"]["rate_source"] == "unknown"
    assert rec["marginal_usd"]["total"] == 0.0
    assert rec["finalized"] is True


def test_unwritable_artifacts_dir_still_returns_zero(tmp_path):
    # The strongest form of "cost can never cost a run": even total I/O failure
    # exits 0 so pod_entry.sh's `|| echo` is never even reached.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    rc = main(["begin", str(blocker / "sub")], env=ENV, api=FakeApi(pod=POD),
              now=1000.0)
    assert rc == 0


def test_bad_argv_returns_nonzero():
    # Operator error IS worth a nonzero exit - it is a wiring bug, not a runtime
    # degradation, and it should surface in the pod log.
    assert main([], env=ENV) == 2
    assert main(["bogus", "/tmp"], env=ENV) == 2


def test_help_returns_zero(capsys):
    # A junior admin at 2 AM asking for help should not see a failure signal.
    # argparse exits with code 0 for --help; we catch that and return 0 to make
    # it clear that asking for syntax is a successful operation.
    rc = main(["--help"], env=ENV)
    assert rc == 0
    captured = capsys.readouterr()
    assert "begin" in captured.out or "finalize" in captured.out


def test_record_contains_no_secret_values(tmp_path):
    # Nothing in the pod env may leak into an uploaded artifact.
    env = {**ENV, "RUNPOD_API_KEY": "RPKEY_SECRET_abc", "HF_TOKEN": "hf_SECRET_xyz"}
    main(["begin", str(tmp_path)], env=env, api=FakeApi(pod=POD), now=1000.0)
    raw = (tmp_path / RECORD_NAME).read_text()
    assert "RPKEY_SECRET_abc" not in raw
    assert "hf_SECRET_xyz" not in raw


def test_record_on_disk_is_valid_ascii_json(tmp_path):
    main(["begin", str(tmp_path)], env=ENV, api=FakeApi(pod=POD), now=1000.0)
    raw = (tmp_path / RECORD_NAME).read_bytes()
    assert all(b < 128 for b in raw)
    assert json.loads(raw.decode("ascii"))["schema_version"] == 1


def test_preflight_line_reports_the_secure_rate():
    gpu = {"securePrice": 0.99, "communityPrice": 0.69,
           "lowestPrice": {"uninterruptablePrice": 0.69}}
    line = preflight_line(ENV, api=FakeApi(gpu=gpu))
    assert "0.99" in line
    assert "0.69" not in line  # the community rate must never be quoted here
    assert "SECURE" in line


def test_preflight_line_names_the_gpu_and_projects_a_three_hour_run():
    # The junior-admin-at-2AM affordance: see the number BEFORE the spend.
    gpu = {"securePrice": 0.99}
    line = preflight_line(ENV, api=FakeApi(gpu=gpu))
    assert "RTX 5090" in line
    assert "2.97" in line  # 0.99 * 3h


def test_preflight_line_degrades_without_a_price():
    line = preflight_line(ENV, api=FakeApi(gpu=None))
    assert "unavailable" in line.lower()


def test_preflight_line_is_ascii():
    line = preflight_line(ENV, api=FakeApi(gpu={"securePrice": 0.99}))
    assert all(ord(c) < 128 for c in line)


def test_preflight_line_makes_reference_framing_explicit():
    # The projection must not read as a prediction when no predictor exists yet.
    # Reference framing must be explicit and adjacent to the figure - a 2 AM
    # operator reads stderr once, acts decisively, and never re-reads the line.
    gpu = {"securePrice": 0.99}
    line = preflight_line(ENV, api=FakeApi(gpu=gpu))
    # An unmissable marker, not a skimmable one: "e.g." is too easy to read past
    # when a concrete dollar figure sits right next to it.
    assert "REFERENCE ONLY" in line
    assert "not a prediction of this run" in line
    # And the REASON, because "reference only" invites "why?" and the answer is
    # what stops the number being repeated back as a quote.
    assert "no duration predictor" in line
    # The rate itself IS real - the disclaimer must not undermine that.
    assert "$0.99/hr" in line
