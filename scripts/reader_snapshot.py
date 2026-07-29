"""One F-040 reader-pod snapshot cycle. STANDALONE BY CONTRACT: runs in-pod on
python:3.12-slim where the assay package does NOT exist - stdlib + huggingface_hub
imports only (pinned by an AST contract test). Invoked once per cycle by
reader_loop.sh; all loop/terminate logic lives there.

Exit codes (reader_loop.sh keys off these):
  0  - cycle done, keep looping
  10 - main pod gone/terminal, final snapshot committed (or retries exhausted): DONE
  11 - reader TTL expired, final snapshot committed: give up loudly

The reader NEVER writes to /runpod-volume (spec discipline rule, contract-tested)."""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

REST_BASE = "https://rest.runpod.io/v1"
EXIT_CONTINUE = 0
EXIT_DONE = 10
EXIT_TTL = 11
FINAL_COMMIT_ATTEMPTS = 5
FINAL_COMMIT_BACKOFF_S = 30.0
# F-046 (drill 2, on metal): a just-created pod can be invisible to a status
# lookup for a short window (REST read-replica 404 lag / GraphQL null lag) - the
# reader's first cycle saw one "gone" for a RUNNING, heartbeating writer and
# published "burn over". Never finalize on a single observation.
GONE_CONFIRM_CYCLES = 2
# Container-local state (NEVER the volume - spec discipline). Deliberate R-11
# carve-out, safe in BOTH restart outcomes: if a restart clears /tmp the count
# restarts and finalize is merely delayed a cycle; if an in-place restart
# preserves it, every counted observation was a genuine completed gone cycle
# and any alive sighting still resets it. Neither direction can false-finalize.
STATE_DIR = "/tmp/reader"


def _consecutive_gone(state_dir: str, gone: bool) -> int:
    """Count consecutive gone observations across cycle processes via a local
    state file. Any alive sighting resets. Unreadable/corrupt state counts as 0
    (safe direction: delays finalize); state I/O must never crash a cycle."""
    path = os.path.join(state_dir, "gone_count")
    if not gone:
        try:
            os.remove(path)
        except OSError:
            pass
        return 0
    prior = 0
    try:
        with open(path) as fh:
            prior = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        prior = 0
    count = prior + 1
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(str(count))
    except OSError:
        pass
    return count


# A "stale" heartbeat within this many seconds of the 24h wrap is really a FRESH
# one that landed between the cycle's `now` stamp and the file read (F-041: `now`
# is stamped ~2 REST round-trips earlier; 3/12 live drill cycles published 86399.0
# for a seconds-old tick). A genuinely 23.95h-stale heartbeat is indistinguishable
# mod 24h anyway, so clamping to 0 is the honest reading for the operator.
_WRAP_SLACK_SECONDS = 300


def heartbeat_age_seconds(line: str, now: datetime):
    """Age of a heartbeat line from its own stamp. heartbeat.py stamps time-only UTC
    ("HH:MM:SS [NNN] stage | msg"), so the age is computed mod 24h to survive
    midnight rollover; file mtime is never used (R-8)."""
    try:
        ts = datetime.strptime(line[:8], "%H:%M:%S")
    except ValueError:
        return None
    now_s = now.hour * 3600 + now.minute * 60 + now.second
    line_s = ts.hour * 3600 + ts.minute * 60 + ts.second
    age = float((now_s - line_s) % 86400)
    if age > 86400 - _WRAP_SLACK_SECONDS:
        return 0.0
    return age


def _parse_pod_timestamp(value):
    """Pod timestamp -> aware datetime, or None. RunPod's REST surface returns
    Go-format strings ("2026-07-28 17:36:57.73 +0000 UTC", captured live 07-28)
    while GraphQL returns ISO-like ones - fromisoformat alone fail-opened the TTL
    backstop and boot escalation on every real cycle (F-044)."""
    if not value:
        return None
    s = value.strip()
    if s.endswith(" UTC"):
        s = s[: -len(" UTC")].strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    # Go format keeps a SPACE before the offset ("17:36:57.73 +0000"), which
    # fromisoformat rejects; %z accepts "+0000" and %f accepts 1-6 digits.
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def read_tail(path: str, max_bytes: int = 200_000):
    """Last max_bytes of a file, stateless (R-11: no offsets - a same-id main-pod
    restart truncates stdout.log and a fresh full read absorbs that silently).
    Returns (text | None, total_size)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("ascii", errors="replace"), size
    except OSError:
        return None, 0


def _running_minutes(main_pod: dict | None, now: datetime):
    if not main_pod:
        return None
    started = main_pod.get("lastStartedAt") or main_pod.get("createdAt")
    t = _parse_pod_timestamp(started)
    if t is None:
        return None
    return (now - t).total_seconds() / 60.0


def build_status(*, main_pod, main_status: str, dir_exists: bool,
                 heartbeat_lines: list, stdout_bytes: int, now: datetime,
                 escalate_after_min: float) -> dict:
    """The status.json a half-asleep operator reads on a phone. States must be
    honest: boot grace names the wait, escalation names BOTH candidate causes
    (60-90 min image pulls are real; so is the /workspace mount split-brain that
    would otherwise look like an infinite boot - R-10b)."""
    last_hb = heartbeat_lines[-1].rstrip("\n") if heartbeat_lines else None
    status = {
        "cycle_utc": now.isoformat(),
        "main_pod_status": main_status,
        "state": "running",
        "note": "",
        "phase": last_hb,
        "heartbeat_age_seconds": (
            heartbeat_age_seconds(last_hb, now) if last_hb else None),
        "stdout_bytes": stdout_bytes,
    }
    if not dir_exists:
        mins = _running_minutes(main_pod, now)
        if main_status == "RUNNING" and mins is not None and mins > escalate_after_min:
            status["state"] = "no-artifacts-dir-escalated"
            status["note"] = (
                f"main pod RUNNING ~{mins:.0f} min but the artifacts dir never "
                "appeared. Either a very slow image pull (60-90 min observed) or "
                "the /workspace mount split-brain (see pod_entry.sh warning) - if "
                "this persists past ~90 min, treat as split-brain.")
        else:
            status["state"] = "waiting-for-main-pod-boot"
            status["note"] = ("artifacts dir not created yet; image pulls have "
                              "taken 60-90 min. Main pod status is shown above.")
    return status


GRAPHQL_URL = "https://api.runpod.io/graphql"


def _graphql_get_pod(pod_id: str, key: str):
    """F-042 fallback transport: the GraphQL pod query (the burn-proven surface -
    self_terminate has ridden it 5-for-5 on metal). Returns a REST-shaped dict, or
    None for a gone pod (GraphQL returns pod: null where REST 404s - verified live
    07-28 against a terminated pod id). Bearer auth, never key-in-URL."""
    body = json.dumps({"query": (
        'query { pod(input: {podId: "%s"}) '
        "{ desiredStatus createdAt lastStartedAt } }" % pod_id)}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL, data=body, method="POST",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json", "User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    if payload.get("errors"):
        raise RuntimeError("graphql pod query failed: %s" % payload["errors"])
    return payload["data"]["pod"]


def _rest_get(path: str, key: str):
    """GET REST_BASE+path. None on 404 (terminated pods are not queryable - lore).
    Real UA because RunPod sits behind Cloudflare, which 403s Python-urllib (lore
    gotcha). Any OTHER failure of a /pods/ lookup falls back to GraphQL (F-042:
    the 07-28 drill hit an 18-min origin-dependent window where in-pod REST 403'd
    a fully valid key while GraphQL answered - a single-transport reader could
    neither see the main pod die nor self-delete). Fallback failures propagate;
    callers already degrade (poll -> 'unknown', TTL -> skip cycle)."""
    req = urllib.request.Request(
        REST_BASE + path,
        headers={"Authorization": "Bearer " + key, "User-Agent": "curl/8.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        if path.startswith("/pods/"):
            return _graphql_get_pod(path.rsplit("/", 1)[-1], key)
        raise
    except urllib.error.URLError:
        if path.startswith("/pods/"):
            return _graphql_get_pod(path.rsplit("/", 1)[-1], key)
        raise


def _reader_age_seconds(rest_get, env, now: datetime):
    """TTL anchor (R-4): the reader pod's own createdAt from REST - survives
    container restarts, unlike any in-shell timer. Fail-open on lookup failure
    (skip the TTL check this cycle; next cycle retries)."""
    try:
        me = rest_get("/pods/" + env["RUNPOD_POD_ID"], env["RUNPOD_API_KEY"])
        created = me.get("createdAt") if me else None
        t = _parse_pod_timestamp(created)
        if t is None:
            return None
        return (now - t).total_seconds()
    except Exception:  # noqa: BLE001 - TTL check must never crash a cycle
        return None


def _commit(hf_api, dataset: str, ops, message: str) -> None:
    hf_api.create_commit(repo_id=dataset, repo_type="dataset",
                         operations=ops, commit_message=message)


def run_cycle(env, *, hf_api, rest_get, now: datetime, sleep=time.sleep,
              state_dir: str = STATE_DIR) -> int:
    """One stateless snapshot cycle. See module docstring for exit codes."""
    from huggingface_hub import CommitOperationAdd

    main_id = env["ASSAY_READER_MAIN_POD_ID"]
    dataset = env["ASSAY_ARTIFACTS_DATASET"]
    run_dir = os.path.join(env["ASSAY_ARTIFACTS_DIR"], main_id)
    live = "runs/" + main_id + "/live/"

    # Main pod status: only a CONFIRMED 404/terminal ends the reader (F-046: two
    # consecutive gone observations - one can be the boot race); a network blip
    # degrades to "unknown" and the cycle continues (test-pinned).
    main_pod, main_status, gone_observed = None, "unknown", False
    try:
        main_pod = rest_get("/pods/" + main_id, env["RUNPOD_API_KEY"])
        if main_pod is None:
            main_status, gone_observed = "terminated", True
        else:
            main_status = main_pod.get("desiredStatus", "unknown")
            gone_observed = main_status in ("EXITED", "TERMINATED")
    except Exception as exc:  # noqa: BLE001
        print("reader: main-pod poll failed (continuing):", exc, file=sys.stderr)
    gone_count = _consecutive_gone(state_dir, gone_observed)
    main_gone = gone_count >= GONE_CONFIRM_CYCLES

    ttl_expired = False
    age = _reader_age_seconds(rest_get, env, now)
    if age is not None and age > float(env["ASSAY_READER_TTL"]):
        ttl_expired = True

    dir_exists = os.path.isdir(run_dir)
    hb_text, hb_lines = None, []
    stdout_text, stdout_size = None, 0
    tb_text = None
    if dir_exists:
        hb_text, _ = read_tail(os.path.join(run_dir, "heartbeat.log"))
        hb_lines = hb_text.splitlines() if hb_text else []
        stdout_text, stdout_size = read_tail(os.path.join(run_dir, "stdout.log"))
        tb_text, _ = read_tail(os.path.join(run_dir, "traceback.txt"))

    status = build_status(
        main_pod=main_pod, main_status=main_status, dir_exists=dir_exists,
        heartbeat_lines=hb_lines, stdout_bytes=stdout_size, now=now,
        escalate_after_min=float(env["ASSAY_READER_BOOT_ESCALATE_MIN"]))
    if main_gone:
        status["state"] = "final"
        status["note"] = "main pod gone - final snapshot; reader self-terminating"
    elif ttl_expired:
        status["state"] = "reader-ttl-expired"
        status["note"] = ("reader TTL expired; burn MAY still be running - raise "
                          "ASSAY_READER_TTL for multi-day burns. Reader is "
                          "self-terminating; console tab is the fallback.")
    elif gone_observed:
        # First gone observation: could be the boot/propagation race (F-046 -
        # seen live: a 26s-old RUNNING pod answered "gone" once). Say so
        # honestly and confirm next cycle before declaring the burn over.
        status["state"] = "main-pod-not-visible"
        status["note"] = ("main pod lookup says gone, but a single observation "
                          "can be boot/propagation lag - confirming next cycle "
                          "before declaring the burn over.")

    def _op(name: str, data: str):
        return CommitOperationAdd(
            path_in_repo=live + name,
            path_or_fileobj=io.BytesIO(data.encode("ascii", errors="replace")))

    ops = [_op("status.json", json.dumps(status, indent=2))]
    if stdout_text is not None:
        ops.append(_op("stdout.log", stdout_text))
    if hb_text is not None:
        ops.append(_op("heartbeat.log", hb_text))
    if tb_text is not None:
        ops.append(_op("traceback.txt", tb_text))
    if main_gone:
        ops.append(_op("READER_DONE", "burn over at " + now.isoformat() + "\n"))
    elif ttl_expired:
        ops.append(_op("READER_TTL_EXPIRED",
                       "reader gave up at " + now.isoformat() +
                       " - burn may still be running\n"))

    message = "reader " + status["state"] + " " + now.isoformat()
    if main_gone or ttl_expired:
        # Final commit: bounded retry (R-9b - it can race the main pod's own
        # artifacts upload), then give up so self-delete still happens (R-4).
        for attempt in range(FINAL_COMMIT_ATTEMPTS):
            try:
                _commit(hf_api, dataset, ops, message)
                break
            except Exception as exc:  # noqa: BLE001
                print("reader: final commit attempt", attempt + 1, "failed:",
                      exc, file=sys.stderr)
                if attempt < FINAL_COMMIT_ATTEMPTS - 1:
                    sleep(FINAL_COMMIT_BACKOFF_S)
        return EXIT_TTL if ttl_expired and not main_gone else EXIT_DONE
    try:
        _commit(hf_api, dataset, ops, message)
    except Exception as exc:  # noqa: BLE001 - mid-run cycles self-heal next cycle
        print("reader: commit failed (retrying next cycle):", exc, file=sys.stderr)
    return EXIT_CONTINUE


def main() -> int:
    from huggingface_hub import HfApi
    return run_cycle(os.environ, hf_api=HfApi(token=os.environ["HF_TOKEN"]),
                     rest_get=_rest_get, now=datetime.now(timezone.utc))


if __name__ == "__main__":
    sys.exit(main())
