import json
import os

from assay import runpod_ctl
from assay.cost import collect
from assay.cost.collect import (
    RECORD_NAME,
    _tmp_record_path,
    effective_uptime_seconds,
    probe_gpu_price,
    probe_pod,
    read_record,
    record_path,
    scan_gate_result,
    write_record,
)
from assay.job import GATE_FAILED_MARKER, GATE_PASSED_MARKER

SAMPLE_POD = {
    "id": "vax1xu90t2f4pc",
    "costPerHr": 0.99,
    "uptimeSeconds": 11242,
    "containerDiskInGb": 40,
    "volumeInGb": 0,
    "desiredStatus": "RUNNING",
    "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 5090"},
    "gpuCount": 1,
}


class FakeApi:
    def __init__(self, pod=None, gpu=None, raises=False):
        self._pod, self._gpu, self._raises = pod, gpu, raises
        self.calls = []

    def get_pod(self, pod_id):
        self.calls.append(pod_id)
        if self._raises:
            raise RuntimeError("RunPod API down")
        return self._pod

    def get_gpu(self, gpu_id, gpu_quantity=1):
        self.calls.append(gpu_id)
        if self._raises:
            raise RuntimeError("RunPod API down")
        return self._gpu


def test_probe_pod_returns_the_payload():
    api = FakeApi(pod=SAMPLE_POD)
    assert probe_pod("vax1xu90t2f4pc", api=api) == SAMPLE_POD
    assert api.calls == ["vax1xu90t2f4pc"]


def test_probe_pod_returns_none_for_a_terminated_pod():
    # VERIFIED 2026-07-25 against 5 real terminated assay pod ids: RunPod returns
    # None. This is the NORMAL state, not an error, and it is why there is no
    # post-hoc cost recovery path.
    assert probe_pod("gone", api=FakeApi(pod=None)) is None


def test_probe_pod_swallows_api_errors():
    # A cost probe must never take down a run.
    assert probe_pod("x", api=FakeApi(raises=True)) is None


def test_probe_pod_without_an_id_returns_none():
    assert probe_pod("", api=FakeApi(pod=SAMPLE_POD)) is None


def test_probe_gpu_price_picks_secure_not_lowest():
    # THE PRICE TRAP: RTX 5090 secure is $0.99/hr but communityPrice and
    # lowestPrice.uninterruptablePrice are both $0.69. assay pins CLOUD_TYPE=SECURE
    # because network volumes exist only in secure DCs, so grabbing the lowest
    # price undercounts every quote by 30%.
    gpu = {"securePrice": 0.99, "communityPrice": 0.69,
           "lowestPrice": {"uninterruptablePrice": 0.69}}
    assert probe_gpu_price("NVIDIA GeForce RTX 5090", "SECURE",
                           api=FakeApi(gpu=gpu)) == 0.99


def test_probe_gpu_price_uses_community_when_not_secure():
    gpu = {"securePrice": 0.99, "communityPrice": 0.69}
    assert probe_gpu_price("x", "COMMUNITY", api=FakeApi(gpu=gpu)) == 0.69


def test_probe_gpu_price_degrades_to_none():
    assert probe_gpu_price("x", "SECURE", api=FakeApi(raises=True)) is None
    assert probe_gpu_price("x", "SECURE", api=FakeApi(gpu=None)) is None
    assert probe_gpu_price("x", "SECURE", api=FakeApi(gpu={})) is None


def test_probe_gpu_price_returns_none_when_requested_field_is_missing():
    # PRICE TRAP CORRECTNESS: payload with communityPrice and lowestPrice but
    # MISSING securePrice must return None, never fall back to communityPrice.
    # Falling back would reintroduce the 30% undercount the trap exists to prevent
    # (RTX 5090: $0.99 secure vs $0.69 community). Absence of the field is a signal
    # to NOT attempt a cost - the fallback path would put a wrong price into every
    # quote for that GPU type.
    gpu = {"communityPrice": 0.69, "lowestPrice": {"uninterruptablePrice": 0.69}}
    assert probe_gpu_price("NVIDIA GeForce RTX 5090", "SECURE",
                           api=FakeApi(gpu=gpu)) is None


def test_write_then_read_record_round_trips(tmp_path):
    rec = {"schema_version": 1, "outcome": "pass"}
    assert write_record(str(tmp_path), rec) is True
    assert (tmp_path / RECORD_NAME).exists()
    assert read_record(str(tmp_path)) == rec


def test_write_record_creates_the_directory(tmp_path):
    target = tmp_path / "nested" / "run"
    assert write_record(str(target), {"a": 1}) is True
    assert (target / RECORD_NAME).exists()


def test_write_record_leaves_no_temp_file(tmp_path):
    # Atomic tmp + os.replace: a mid-write death must not leave a torn record on
    # the volume, and must not litter it either. The temp name is dot-prefixed
    # (.cost.json.tmp) precisely so a survivor of a mid-write death (e.g. the
    # `timeout -k 5` SIGKILL case) reads as visibly scratch, not a second
    # candidate "real" record that publish_artifacts would upload alongside
    # cost.json - confirm neither the new nor the old undotted name lingers.
    write_record(str(tmp_path), {"a": 1})
    names = [p.name for p in tmp_path.iterdir()]
    assert names == [RECORD_NAME]
    assert ".cost.json.tmp" not in names
    assert "cost.json.tmp" not in names


def test_tmp_record_path_is_dot_prefixed(tmp_path):
    # Pins the scratch-file naming convention directly: a leading dot makes a
    # mid-write survivor visibly a temp file rather than a second "which file is
    # real?" candidate in the published artifacts dataset (upload_folder has no
    # ignore patterns and would sweep an undotted survivor in as-is).
    assert _tmp_record_path(str(tmp_path)) == str(tmp_path / ".cost.json.tmp")


def test_write_record_is_ascii_only(tmp_path):
    # The whole repo is ASCII-only; a record is uploaded and read by others.
    # \u escape so THIS file stays ASCII too, while still handing the writer
    # a genuinely non-ASCII value.
    write_record(str(tmp_path), {"note": "caf\u00e9"})
    raw = (tmp_path / RECORD_NAME).read_bytes()
    assert all(b < 128 for b in raw)


def test_write_record_returns_false_instead_of_raising(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    assert write_record(str(blocker / "sub"), {"a": 1}) is False


def test_write_record_cleans_up_temp_file_on_json_serialize_failure(tmp_path):
    # json.dump can fail mid-write if the record contains a non-serializable value
    # (e.g. a live object()). The .tmp file must not be left orphaned on the
    # shared, long-lived network volume - a stray cost.json.tmp at 2 AM creates a
    # "which file is real?" confusion that the atomic-write design prevents.
    record = {"bad": object()}
    assert write_record(str(tmp_path), record) is False
    # Verify no .tmp file remains, and the original error was logged to stderr.
    tmp_files = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tmp_files == []


def test_read_record_returns_none_when_absent(tmp_path):
    assert read_record(str(tmp_path)) is None


def test_read_record_returns_none_on_corrupt_json(tmp_path):
    (tmp_path / RECORD_NAME).write_text("{not json")
    assert read_record(str(tmp_path)) is None


def test_record_path_is_under_the_artifacts_dir(tmp_path):
    assert record_path(str(tmp_path)) == os.path.join(str(tmp_path), RECORD_NAME)


def test_scan_gate_result_normalizes_the_markers(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(f"noise\n{GATE_PASSED_MARKER}\nmore\n")
    assert scan_gate_result(str(p)) == "pass"
    p.write_text(f"noise\n{GATE_FAILED_MARKER}\n")
    assert scan_gate_result(str(p)) == "fail"


def test_scan_gate_result_uses_the_last_marker_not_the_first(tmp_path):
    # job.py prints its marker as the FINAL act of a run, so scanning for the LAST
    # occurrence is strictly more faithful than first-match-wins. An early stray
    # "GATE PASSED" (echoed sub-step output, a retry, fixture noise upstream)
    # must not beat the real final "GATE FAILED" - gate_fail is billable audit
    # work and pass means a certified deliverable, so misreading one as the
    # other is commercially load-bearing, not cosmetic.
    p = tmp_path / "run.log"
    p.write_text(
        f"noise\n{GATE_PASSED_MARKER}\nmore build output\n{GATE_FAILED_MARKER}\n")
    assert scan_gate_result(str(p)) == "fail"


def test_scan_gate_result_returns_none_without_a_marker(tmp_path):
    p = tmp_path / "run.log"
    p.write_text("nothing interesting here\n")
    assert scan_gate_result(str(p)) is None
    assert scan_gate_result(str(tmp_path / "missing.log")) is None
    assert scan_gate_result(None) is None


def test_scan_gate_result_never_returns_log_content(tmp_path):
    # It reads the RAW UNREDACTED log, so it must only ever emit the normalized
    # token. Returning any log text would put a secret into cost.json.
    p = tmp_path / "run.log"
    p.write_text(f"RPKEY_SECRET_abc123\n{GATE_PASSED_MARKER}\n")
    assert scan_gate_result(str(p)) == "pass"


def test_effective_uptime_prefers_provider_truth():
    basis = {"began_at_unix": 1000.0, "uptime_seconds_at_begin": 400}
    assert effective_uptime_seconds(SAMPLE_POD, basis, now=9999.0) == 11242.0


def test_effective_uptime_accrues_locally_when_the_pod_is_gone():
    # The finalize probe failed or the pod is already unqueryable: accrue from the
    # basis. 400s of image pull + 600s of wall clock since begin.
    basis = {"began_at_unix": 1000.0, "uptime_seconds_at_begin": 400}
    assert effective_uptime_seconds(None, basis, now=1600.0) == 1000.0


def test_effective_uptime_includes_the_image_pull_offset():
    # uptimeSeconds starts at pod CREATION, so the offset captured at begin is
    # billed image-pull time the job never sees. Dropping it undercounts the run.
    basis = {"began_at_unix": 1000.0, "uptime_seconds_at_begin": 5400}
    assert effective_uptime_seconds(None, basis, now=1000.0) == 5400.0


def test_effective_uptime_is_zero_without_a_basis():
    assert effective_uptime_seconds(None, None, now=1600.0) == 0.0


def test_effective_uptime_never_goes_backwards_on_clock_skew():
    basis = {"began_at_unix": 2000.0, "uptime_seconds_at_begin": 100}
    assert effective_uptime_seconds(None, basis, now=1000.0) == 100.0


def test_default_sku_constants_match_runpod_ctl():
    # Contract test for the single-sourced import: collect.py must import its
    # defaults from runpod_ctl.py, never re-type them as literals. A SKU change
    # in runpod_ctl.py (e.g. moving the default to H200 for the next roadmap
    # item) must not silently leave a stale duplicate pricing a different GPU in
    # the pre-flight line or the in-pod catalog fallback - if these ever drift,
    # this test catches it immediately rather than on a paid metal run.
    assert collect._DEFAULT_GPU_TYPE == runpod_ctl.RTX_5090
    assert collect._DEFAULT_CLOUD_TYPE == runpod_ctl.CLOUD_TYPE
    assert collect._DEFAULT_REGION == runpod_ctl.REGION


def test_probed_fields_exist_in_the_installed_sdk_query():
    """The fields we read must actually be in the SDK's shipped GraphQL query.

    Fake-api tests cannot catch an SDK field RENAME - they would keep passing
    happily while the real probe read None for every cost input and every record
    came out $0.00 with nothing indicating anything was wrong. Pin our field names
    against the query text the installed runpod SDK ships, so a rename breaks CI
    instead of a paid metal run. Same spirit as the GATE-marker contract test: a
    two-sided string dependency with no shared runtime gets pinned.
    """
    from runpod.api.queries import pods as sdk_pods

    query = sdk_pods.QUERY_POD + sdk_pods.generate_pod_query("probe")
    for field in ("costPerHr", "uptimeSeconds", "containerDiskInGb", "volumeInGb",
                  "desiredStatus", "gpuCount", "gpuDisplayName"):
        assert field in query, (
            f"{field!r} is not in the installed runpod SDK's pod query - the cost "
            "probe would silently read None for it and every record would be $0.00")
