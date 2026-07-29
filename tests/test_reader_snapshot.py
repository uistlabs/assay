"""reader_snapshot.py runs IN-POD with no assay package: these tests load it by file
path (verify_prune pattern) and pin the behaviors a sleeping operator depends on -
rollover-safe heartbeat age, honest boot-grace wording, and the split-brain
escalation (R-10b)."""
import importlib.util
import pathlib
from datetime import datetime, timezone

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "reader_snapshot.py"


def _load():
    spec = importlib.util.spec_from_file_location("reader_snapshot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load()


def _utc(h, m, s):
    return datetime(2026, 7, 27, h, m, s, tzinfo=timezone.utc)


# heartbeat lines are "HH:MM:SS [NNN] stage | message" (heartbeat.py emit format)

def test_heartbeat_age_simple():
    assert rs.heartbeat_age_seconds("12:00:00 [001] eval", _utc(12, 0, 30)) == 30


def test_heartbeat_age_midnight_rollover():
    # Line stamped 23:58:00, now 00:02:00 - age is 4 min, not negative/86k.
    assert rs.heartbeat_age_seconds("23:58:00 [017] eval", _utc(0, 2, 0)) == 240


def test_heartbeat_age_garbage_line_is_none():
    assert rs.heartbeat_age_seconds("not a heartbeat", _utc(1, 0, 0)) is None
    assert rs.heartbeat_age_seconds("", _utc(1, 0, 0)) is None


def test_heartbeat_age_fresh_line_after_now_clamps_to_zero():
    # F-041 (07-28 drill, 3/12 live cycles): `now` is stamped ~2 REST round-trips
    # BEFORE the file read, so a tick landing in that gap postdates `now` and
    # (now - line) % 86400 published 86399.0 - a fresh heartbeat reading as a day
    # old to the 2 AM operator. Seconds-in-the-future must clamp to 0.0.
    assert rs.heartbeat_age_seconds("12:00:01 [001] eval", _utc(12, 0, 0)) == 0.0
    assert rs.heartbeat_age_seconds("12:00:04 [002] eval", _utc(12, 0, 0)) == 0.0


def test_heartbeat_age_midnight_rollover_still_real():
    # The clamp must not swallow genuine just-under-24h staleness semantics for
    # the rollover case the mod-24h design exists for (4 min stale across 00:00).
    assert rs.heartbeat_age_seconds("23:58:00 [017] eval", _utc(0, 2, 0)) == 240


def test_read_tail_bounded(tmp_path):
    p = tmp_path / "stdout.log"
    p.write_bytes(b"x" * 300_000)
    text, size = rs.read_tail(str(p), max_bytes=200_000)
    assert size == 300_000 and len(text) == 200_000


def test_read_tail_missing_file(tmp_path):
    text, size = rs.read_tail(str(tmp_path / "absent.log"))
    assert text is None and size == 0


def test_status_waiting_for_boot():
    s = rs.build_status(main_pod={"desiredStatus": "RUNNING",
                                  "lastStartedAt": "2026-07-27T12:00:00Z"},
                        main_status="RUNNING", dir_exists=False,
                        heartbeat_lines=[], stdout_bytes=0,
                        now=_utc(12, 10, 0), escalate_after_min=30)
    assert s["state"] == "waiting-for-main-pod-boot"
    assert s["main_pod_status"] == "RUNNING"


def test_status_escalates_split_brain_after_threshold():
    # RUNNING 45 min, dir never appeared -> escalate wording names BOTH candidate
    # causes (slow image pull vs /workspace mount split-brain), R-10b.
    s = rs.build_status(main_pod={"desiredStatus": "RUNNING",
                                  "lastStartedAt": "2026-07-27T12:00:00Z"},
                        main_status="RUNNING", dir_exists=False,
                        heartbeat_lines=[], stdout_bytes=0,
                        now=_utc(12, 45, 0), escalate_after_min=30)
    assert s["state"] == "no-artifacts-dir-escalated"
    assert "split-brain" in s["note"] and "pull" in s["note"]


def test_status_normal_running():
    s = rs.build_status(main_pod={"desiredStatus": "RUNNING",
                                  "lastStartedAt": "2026-07-27T12:00:00Z"},
                        main_status="RUNNING", dir_exists=True,
                        heartbeat_lines=["12:30:00 [042] eval | task 3/7"],
                        stdout_bytes=12345, now=_utc(12, 30, 20),
                        escalate_after_min=30)
    assert s["state"] == "running"
    assert s["heartbeat_age_seconds"] == 20
    assert s["phase"] == "12:30:00 [042] eval | task 3/7"
    assert s["stdout_bytes"] == 12345


def test_status_main_unknown_on_rest_failure():
    # A transient REST failure must degrade to 'unknown', never crash or finalize.
    s = rs.build_status(main_pod=None, main_status="unknown", dir_exists=True,
                        heartbeat_lines=[], stdout_bytes=0,
                        now=_utc(1, 0, 0), escalate_after_min=30)
    assert s["main_pod_status"] == "unknown"
    assert s["state"] == "running"


# ---- Task 3: cycle behavior ----
import ast
import json


class FakeHfApi:
    def __init__(self, fail_times: int = 0):
        self.commits = []
        self._fail = fail_times

    def create_commit(self, *, repo_id, repo_type, operations, commit_message):
        if self._fail > 0:
            self._fail -= 1
            raise RuntimeError("hub 429")
        self.commits.append({"repo_id": repo_id, "repo_type": repo_type,
                             "ops": operations, "msg": commit_message})


def _env(tmp_path, **over):
    e = {
        "RUNPOD_API_KEY": "rk", "HF_TOKEN": "hf", "RUNPOD_POD_ID": "readerpod",
        "ASSAY_READER_MAIN_POD_ID": "mainpod1",
        "ASSAY_ARTIFACTS_DATASET": "org/run-artifacts",
        "ASSAY_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        "ASSAY_READER_TTL": "86400",
        "ASSAY_READER_BOOT_ESCALATE_MIN": "30",
    }
    e.update(over)
    return e


def _mk_run_dir(tmp_path):
    d = tmp_path / "artifacts" / "mainpod1"
    d.mkdir(parents=True)
    (d / "stdout.log").write_text("line1\nline2\n")
    (d / "heartbeat.log").write_text("12:00:00 [001] quantize | start\n")
    return d


def _rest(pods: dict):
    """rest_get stub: pods maps pod_id -> dict | None (404) | Exception."""
    def get(path, key):
        pod_id = path.rsplit("/", 1)[-1]
        v = pods[pod_id]
        if isinstance(v, Exception):
            raise v
        return v
    return get


RUNNING = {"desiredStatus": "RUNNING", "lastStartedAt": "2026-07-27T12:00:00Z",
           "createdAt": "2026-07-27T11:00:00Z"}


def test_cycle_running_one_atomic_commit(tmp_path):
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": RUNNING, "readerpod": RUNNING}),
                      now=_utc(12, 30, 0))
    assert rc == rs.EXIT_CONTINUE
    assert len(api.commits) == 1  # R-7: ONE create_commit per cycle, atomic
    paths = [op.path_in_repo for op in api.commits[0]["ops"]]
    assert "runs/mainpod1/live/status.json" in paths
    assert "runs/mainpod1/live/stdout.log" in paths
    assert "runs/mainpod1/live/heartbeat.log" in paths
    assert not any("traceback" in p for p in paths)  # absent file not uploaded


def test_cycle_includes_traceback_when_present(tmp_path):
    d = _mk_run_dir(tmp_path)
    (d / "traceback.txt").write_text("boom")
    api = FakeHfApi()
    rs.run_cycle(_env(tmp_path), hf_api=api,
                 rest_get=_rest({"mainpod1": RUNNING, "readerpod": RUNNING}),
                 now=_utc(12, 30, 0))
    assert any(op.path_in_repo == "runs/mainpod1/live/traceback.txt"
               for op in api.commits[0]["ops"])


def test_cycle_uploads_empty_but_readable_heartbeat(tmp_path):
    # "Readable" means uploaded, even at 0 bytes (boot window): an operator
    # must see present-but-empty, not absent.
    d = _mk_run_dir(tmp_path)
    (d / "heartbeat.log").write_text("")
    api = FakeHfApi()
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": RUNNING, "readerpod": RUNNING}),
                      now=_utc(12, 30, 0))
    assert rc == rs.EXIT_CONTINUE
    paths = [op.path_in_repo for op in api.commits[0]["ops"]]
    assert "runs/mainpod1/live/heartbeat.log" in paths


def test_cycle_main_pod_404_finalizes_with_done_marker(tmp_path):
    # F-046: TWO consecutive gone observations finalize (a single one is the
    # boot race, see the dedicated tests below).
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    state = tmp_path / "state"
    gone = _rest({"mainpod1": None, "readerpod": RUNNING})
    rc1 = rs.run_cycle(_env(tmp_path), hf_api=api, rest_get=gone,
                       now=_utc(13, 0, 0), state_dir=str(state))
    assert rc1 == rs.EXIT_CONTINUE
    rc2 = rs.run_cycle(_env(tmp_path), hf_api=api, rest_get=gone,
                       now=_utc(13, 1, 0), state_dir=str(state))
    assert rc2 == rs.EXIT_DONE
    paths = [op.path_in_repo for op in api.commits[-1]["ops"]]
    assert "runs/mainpod1/live/READER_DONE" in paths  # R-15 distinct marker


def test_single_gone_observation_never_finalizes(tmp_path):
    # F-046 (drill 2, VERIFIED on metal): the reader's first cycle 26s after the
    # writer's creation got one "gone" answer while the writer was RUNNING and
    # heartbeating - and published "burn over" for a live pod. One gone
    # observation is a boot/propagation race until confirmed.
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": None, "readerpod": RUNNING}),
                      now=_utc(13, 0, 0), state_dir=str(tmp_path / "state"))
    assert rc == rs.EXIT_CONTINUE
    status = json.loads([op for op in api.commits[0]["ops"]
                         if op.path_in_repo.endswith("status.json")][0]
                        .path_or_fileobj.getvalue())
    assert status["state"] == "main-pod-not-visible"
    assert "confirm" in status["note"].lower()
    paths = [op.path_in_repo for op in api.commits[0]["ops"]]
    assert not any("READER_DONE" in p for p in paths)


def test_gone_then_alive_resets_the_confirmation_counter(tmp_path):
    # gone -> alive -> gone must be two SEPARATE single observations, not a
    # confirmed pair: any alive sighting resets.
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    state = str(tmp_path / "state")
    rc1 = rs.run_cycle(_env(tmp_path), hf_api=api,
                       rest_get=_rest({"mainpod1": None, "readerpod": RUNNING}),
                       now=_utc(13, 0, 0), state_dir=state)
    rc2 = rs.run_cycle(_env(tmp_path), hf_api=api,
                       rest_get=_rest({"mainpod1": RUNNING, "readerpod": RUNNING}),
                       now=_utc(13, 1, 0), state_dir=state)
    rc3 = rs.run_cycle(_env(tmp_path), hf_api=api,
                       rest_get=_rest({"mainpod1": None, "readerpod": RUNNING}),
                       now=_utc(13, 2, 0), state_dir=state)
    assert (rc1, rc2, rc3) == (rs.EXIT_CONTINUE, rs.EXIT_CONTINUE,
                               rs.EXIT_CONTINUE)


def test_corrupt_gone_counter_treated_as_zero(tmp_path):
    # Unreadable state must delay finalize (safe direction), never crash.
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    state = tmp_path / "state"
    state.mkdir()
    (state / "gone_count").write_text("not a number")
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": None, "readerpod": RUNNING}),
                      now=_utc(13, 0, 0), state_dir=str(state))
    assert rc == rs.EXIT_CONTINUE


def test_final_commit_bounded_retry_then_gives_up(tmp_path):
    # R-9b: final snapshot retries (races the main pod's own upload), then gives
    # up so the loop proceeds to self-delete - billing stops even if HF is down.
    _mk_run_dir(tmp_path)
    slept = []
    api = FakeHfApi(fail_times=rs.FINAL_COMMIT_ATTEMPTS)  # every attempt fails
    state = tmp_path / "state"
    state.mkdir()
    (state / "gone_count").write_text("1")  # F-046: prior gone already observed
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": None, "readerpod": RUNNING}),
                      now=_utc(13, 0, 0), sleep=slept.append,
                      state_dir=str(state))
    assert rc == rs.EXIT_DONE
    assert len(slept) == rs.FINAL_COMMIT_ATTEMPTS - 1
    assert api.commits == []


def test_ttl_expiry_finalizes_with_ttl_marker(tmp_path):
    # R-4: TTL anchors on the READER pod's createdAt from REST, so container
    # restarts cannot reset it. R-15: marker distinct from READER_DONE.
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    old = {**RUNNING, "createdAt": "2026-07-26T10:00:00Z"}  # reader created >24h ago
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": RUNNING, "readerpod": old}),
                      now=_utc(13, 0, 0))
    assert rc == rs.EXIT_TTL
    paths = [op.path_in_repo for op in api.commits[0]["ops"]]
    assert "runs/mainpod1/live/READER_TTL_EXPIRED" in paths


def test_transient_rest_failure_continues(tmp_path):
    # Network blip polling the main pod: status 'unknown', NOT finalize (only a
    # definitive 404/terminal may end the reader).
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": OSError("timeout"),
                                      "readerpod": RUNNING}),
                      now=_utc(12, 30, 0))
    assert rc == rs.EXIT_CONTINUE
    status = json.loads([op for op in api.commits[0]["ops"]
                         if op.path_in_repo.endswith("status.json")][0]
                        .path_or_fileobj.getvalue())
    assert status["main_pod_status"] == "unknown"


def test_midrun_commit_failure_is_swallowed(tmp_path):
    # Mid-run cycles self-heal next cycle (R-9): a failed commit logs + continues.
    _mk_run_dir(tmp_path)
    api = FakeHfApi(fail_times=1)
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": RUNNING, "readerpod": RUNNING}),
                      now=_utc(12, 30, 0))
    assert rc == rs.EXIT_CONTINUE


# ---- F-044: live REST timestamps are Go-format, not ISO (captured 2026-07-28) ----
# GET /v1/pods/<id> returned createdAt "2026-07-28 17:36:57.73 +0000 UTC" and
# lastStartedAt "2026-07-28 17:36:57.724 +0000 UTC" on the drill writer. The ISO
# fixtures above stay (the GraphQL fallback path returns ISO-like strings); these
# pin that the LIVE format also works - fromisoformat alone fail-opened the TTL
# backstop and the boot escalation on every real cycle.

LIVE_RUNNING = {"desiredStatus": "RUNNING",
                "lastStartedAt": "2026-07-28 17:36:57.724 +0000 UTC",
                "createdAt": "2026-07-28 17:36:57.73 +0000 UTC"}


def test_parse_pod_timestamp_both_formats():
    iso = rs._parse_pod_timestamp("2026-07-27T11:00:00Z")
    go = rs._parse_pod_timestamp("2026-07-28 17:36:57.73 +0000 UTC")
    assert iso is not None and iso.tzinfo is not None
    assert go is not None and go.tzinfo is not None
    assert (go.hour, go.minute, go.second) == (17, 36, 57)
    assert rs._parse_pod_timestamp("not a time") is None
    assert rs._parse_pod_timestamp("") is None


def test_ttl_expiry_fires_with_live_rest_timestamp_format(tmp_path):
    # The 07-28 drill's exact failure shape: TTL anchor must not fail-open on the
    # real REST format (reader created >24h before `now`).
    _mk_run_dir(tmp_path)
    api = FakeHfApi()
    old = {**LIVE_RUNNING, "createdAt": "2026-07-26 10:00:00.12 +0000 UTC"}
    rc = rs.run_cycle(_env(tmp_path), hf_api=api,
                      rest_get=_rest({"mainpod1": LIVE_RUNNING, "readerpod": old}),
                      now=_utc(13, 0, 0))
    assert rc == rs.EXIT_TTL


def test_escalation_fires_with_live_rest_timestamp_format():
    s = rs.build_status(main_pod={"desiredStatus": "RUNNING",
                                  "lastStartedAt": "2026-07-27 12:00:00.5 +0000 UTC"},
                        main_status="RUNNING", dir_exists=False,
                        heartbeat_lines=[], stdout_bytes=0,
                        now=_utc(12, 45, 0), escalate_after_min=30)
    assert s["state"] == "no-artifacts-dir-escalated"


# ---- F-042: REST-only transport 403'd origin-dependently for the whole 07-28
# drill window while GraphQL worked in-pod (diag pod, runs/_diag/rest403/).
# _rest_get falls back to the GraphQL pod query on non-404 failures. ----

class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status = 200

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(calls, gql_pod):
    import urllib.error

    def urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if "rest.runpod.io" in url:
            raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)
        assert "api.runpod.io/graphql" in url
        assert req.get_header("Authorization", "").startswith("Bearer ")
        return _FakeHTTPResponse({"data": {"pod": gql_pod}})
    return urlopen


def test_rest_get_403_falls_back_to_graphql(monkeypatch):
    calls = []
    pod = {"desiredStatus": "RUNNING",
           "createdAt": "2026-07-28T17:36:57Z",
           "lastStartedAt": "2026-07-28T17:36:57Z"}
    monkeypatch.setattr(rs.urllib.request, "urlopen", _fake_urlopen(calls, pod))
    out = rs._rest_get("/pods/abc123", "k")
    assert out == pod
    assert any("rest.runpod.io" in u for u in calls)      # REST stays primary
    assert any("api.runpod.io/graphql" in u for u in calls)


def test_rest_get_403_graphql_null_means_gone(monkeypatch):
    # GraphQL returns pod: null for a terminated pod - same semantics as REST 404.
    monkeypatch.setattr(rs.urllib.request, "urlopen", _fake_urlopen([], None))
    assert rs._rest_get("/pods/abc123", "k") is None


def test_rest_get_404_returns_none_without_graphql(monkeypatch):
    # 404 is a SEMANTIC answer (pod gone) - it must never trigger the fallback.
    import urllib.error
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
    monkeypatch.setattr(rs.urllib.request, "urlopen", urlopen)
    assert rs._rest_get("/pods/abc123", "k") is None
    assert all("graphql" not in u for u in calls)


def test_standalone_import_contract():
    # The in-pod contract: stdlib + huggingface_hub ONLY. An `import assay` would
    # crash the reader at 2 AM on python:3.12-slim where assay does not exist.
    allowed = {"__future__", "io", "json", "os", "sys", "time", "urllib",
               "datetime", "huggingface_hub"}
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        for m in mods:
            assert m in allowed, f"forbidden import in reader_snapshot.py: {m}"
