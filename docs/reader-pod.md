# Reader pod (F-040): watching a burn from a browser

The reader pod is a small CPU sidecar that `launch.sh` starts next to every paid burn.
It polls the main pod's status over the RunPod REST API (falling back to the GraphQL
API per lookup when REST raises an HTTP error other than 404 or a connection-level
failure - the 2026-07-28 drill hit an 18-minute window where in-pod REST 403'd a
valid key while GraphQL answered)
and tails its logs from the shared network volume, then pushes a snapshot to a private
Hugging Face dataset every few minutes so you can check on a run from a phone or any
browser instead of staying tethered to a terminal. Its self-delete call carries the
same REST-then-GraphQL fallback. It is default-on, costs roughly $0.02 to $0.04 an hour to run
(a 2-vCPU CPU pod, no GPU involved), and has zero effect on the burn or the cert path:
it never writes to the shared network volume, and if it fails to launch or crashes
mid-run the burn continues exactly as if it were not there.

## The bookmark

Once the main pod is up, `launch.sh` prints its RunPod pod id (`created pod: <id>`).
The reader mirrors everything under that id to one stable URL:

```
https://huggingface.co/datasets/<your-org>/<your-dataset>/tree/main/runs/<pod_id>/live
```

`<your-org>/<your-dataset>` is whatever you set `ASSAY_ARTIFACTS_DATASET` to (see
Config below). Bookmark this URL before launch. The reader overwrites the same paths
every cycle instead of appending new ones, so there is never a "latest" file to go
hunting for.

Files under `live/`:

- `status.json` - the one file worth checking first.
  - `state` - one of `waiting-for-main-pod-boot`, `no-artifacts-dir-escalated`,
    `running`, `main-pod-not-visible`, `final`, `reader-ttl-expired`. See Markers and
    Failure modes below for what to do about the ones that are not `running`.
    `main-pod-not-visible` means one status lookup said the main pod is gone and the
    reader is confirming on the next cycle before believing it: a just-created pod
    can be invisible to lookups for a short window (seen live on the 07-28 drill -
    a 26-second-old RUNNING pod answered "gone" once), so a single observation is
    never treated as the burn ending.
  - `cycle_utc` - when this snapshot was taken.
  - `main_pod_status` - the main pod's own RunPod status (`RUNNING`, `EXITED`, and so
    on), or `unknown` if the last REST poll failed.
  - `phase` - the last heartbeat line, or `null` if none has been written yet.
  - `heartbeat_age_seconds` - how old that line's own timestamp is, computed from the
    stamp inside the line, never from file mtime.
  - `note` - a plain-English explanation, populated for every non-`running` state.
  - `stdout_bytes` - size of the main pod's stdout log on disk. A number that keeps
    growing is itself a liveness signal even before you read the tail.
- `stdout.log` - the last ~200 KB of the main pod's redacted stdout tee.
- `heartbeat.log` - the last ~200 KB of the main pod's heartbeat log (lines look like
  `HH:MM:SS [NNN] stage | message`).
- `traceback.txt` - present only after a crash. This is the safety net for a hard
  credit stop: the main pod's own end-of-run forensics upload is not guaranteed to
  run in that case, but the reader has already been copying this file out.

Every cycle re-reads all of these files from scratch; the reader keeps no saved
offsets. If the main pod restarts under the same pod id (which truncates its
stdout.log), the next reader cycle just absorbs that silently instead of showing a
stale or corrupted tail.

## Markers

Two zero-content marker files can appear next to `status.json`, and both mean the
reader is done watching:

- `READER_DONE` - the burn is over. The main pod hit a terminal RunPod status (or
  vanished outright, a 404 on lookup) on TWO consecutive cycles - one observation is
  never trusted (boot race, see `main-pod-not-visible` above) - the reader took one
  last full snapshot, and everything under `live/` is final. Nothing more will be
  written here. A container restart between the two observations either resets the
  count (delaying this by a cycle) or preserves it (every counted observation was a
  real completed gone cycle) - neither direction can fake it.
- `READER_TTL_EXPIRED` - the reader gave up. This does **not** mean the burn is over,
  only that the reader has been running longer than `ASSAY_READER_TTL` and stopped
  watching. Check the RunPod console directly to see whether the main pod is still
  running. If you are about to launch something that will run longer than the
  default 24-hour TTL, raise `ASSAY_READER_TTL` before launch (see Config below).

## Config

All of these are read once, at reader-launch time, from the same shell that runs
`launch.sh`. Defaults below are copied verbatim from `build_reader_payload` in
`src/assay/runpod_ctl.py`.

| Var | Default | What it does |
|---|---|---|
| `ASSAY_READER` | `1` | Set to `0` to disable the reader for this burn. Any value other than `1` also disables it. |
| `ASSAY_READER_INTERVAL` | `600` (seconds, 10 min) | How often the reader snapshots. Read the Quota section below before lowering this. |
| `ASSAY_READER_TTL` | `86400` (seconds, 24h) | How long the reader keeps watching before giving up (`reader-ttl-expired`). Measured from the reader pod's own `createdAt` via REST, not an in-process timer, so a container restart cannot reset or extend it. |
| `ASSAY_READER_BOOT_ESCALATE_MIN` | `30` (minutes) | How long the main pod can report `RUNNING` with no artifacts directory yet before the state escalates to `no-artifacts-dir-escalated`. |
| `ASSAY_ARTIFACTS_DATASET` | none - required | The private HF dataset the reader (and the main pod's own end-of-run upload) writes to. There is no org default: leaving it unset disables the reader entirely for that burn. Set it to `<your-org>/<your-dataset>`. |
| `ASSAY_READER_IMAGE` | `docker.io/library/python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de` | Override the reader's CPU image. Must stay digest-pinned (contain `@sha256:...`) - a bare tag is rejected before launch, the same rule the main pod's own image follows. |

`ASSAY_ARTIFACTS_DIR` is not reader-specific, but matters here too: the reader
derives the run's artifacts path from it the same way the main pod does (default
`/runpod-volume/assay-out/artifacts`). If you override it for a burn, set it before
both pods launch, not just one, or the reader will be watching the wrong directory.

## Freshness

Two channels, two different lag profiles by design:

- `heartbeat.log` is the fresh channel. The main pod opens the file, appends one
  line, and closes it again on every heartbeat emit, so whatever is on disk when the
  reader's cycle runs is current to within the snapshot interval - there is no
  separate flush delay to account for.
- `stdout.log` is the slower channel by design (a held-open tee that flushes to the
  volume rather than open/append/close per line) - but see the measured numbers
  below: on metal the difference was not observable.

`heartbeat_age_seconds` is computed modulo 24h (see Files above). Values within 5
minutes of the 24h wrap are clamped to `0.0`: a "nearly a day stale" reading is in
practice a FRESH tick that landed between the cycle's timestamp and the file read
(it happened on 3 of the drill's 12 cycles). On a multi-day burn (a raised
`ASSAY_READER_TTL`) the age still cannot by itself tell a fresh heartbeat from one
exactly 24h or 48h stale - cross-check `cycle_utc` and whether `stdout_bytes` is
still growing before trusting the age alone on a burn that long.

**Measured on metal 2026-07-28** (drill writer `omto24hgdh72mx`, reader
`2u1n530tewa65a`, 12 cycles at a 60-second interval): `stdout.log` lag -4 to +1
seconds, `heartbeat.log` lag -3 to +3 seconds, relative to each cycle's
`cycle_utc`. Both channels are current to within the snapshot interval; volume
writeback added nothing measurable, and the small negative values are the
cycle-timestamp skew described above, not time travel. The lifecycle half of that
drill (writer self-delete observed by the reader, final snapshot, reader
self-delete) was invalidated by an in-pod REST outage window (F-042 in the
ledger) and is re-run after the transport-fallback fix; these freshness numbers
stand on their own.

## Quota and commit history

Each reader cycle is exactly one HF `create_commit`, with every file in that cycle's
snapshot bundled into it atomically - there is never a moment where `live/` shows a
half-written snapshot. At the default 600 second interval that is about 6 commits an
hour. HF's repo-commit rate limit is real but undocumented and can change without
notice, so treat a 429 as a real risk, not a hypothetical: do not push
`ASSAY_READER_INTERVAL` below roughly 120 seconds. Going lower does not only risk the
reader's own commits - the main pod's end-of-run artifact upload lands on the same
dataset, and a quota trip there would degrade the one upload that actually matters.

A multi-hour burn accretes on the order of 40 to 50 reader commits under
`runs/<pod_id>/live/` in the dataset's history. That is harmless to any single run but
adds up across many burns over time. Squash it occasionally rather than letting it
grow forever:

```python
from huggingface_hub import HfApi
HfApi().super_squash_history(repo_id="<your-org>/<your-dataset>", repo_type="dataset")
```

## Failure modes

**Reader create fails at launch.** The reader hook in `launch.sh` runs after the main
pod already exists, wrapped so no failure inside it can abort the launcher. If the
create call fails, stderr gets a warning that the burn is proceeding without mid-run
observability, and the burn continues untouched. Fallback: the RunPod console's own
log tab for the main pod.

**Reader crashes mid-burn.** `reader_loop.sh` deliberately avoids `set -e`, so a
single bad cycle falls through to the next sleep instead of exiting the container. If
the container does exit anyway, RunPod restarts it. That is safe: every cycle is
stateless (no saved offsets, so a restart just re-reads from scratch), and the TTL is
anchored to the reader pod's own `createdAt` via a REST lookup rather than anything
held in the container's memory, so a restart cannot quietly reset or extend it.

**HF is down or rate-limiting.** A mid-run commit failure is swallowed and logged, and
the next cycle just tries again. The one commit that matters most, the final snapshot
on burn-end or TTL-expiry, gets 5 attempts with a 30 second backoff between them
before the reader gives up on it - but the reader always self-deletes afterward
regardless of whether that final commit landed, because stopping billing outranks
getting the last snapshot through.

Revoke the per-session RunPod key only after your zero-pods check (RunPod console or
REST) shows the reader gone, not before - the reader needs that key to make its own
self-delete call, and revoking it early strands the reader billing with no way to
stop itself.

**`no-artifacts-dir-escalated`.** This fires when the main pod's REST status is
`RUNNING` but its artifacts directory has never appeared, and more than
`ASSAY_READER_BOOT_ESCALATE_MIN` minutes (default 30) have passed. Two real causes
look identical from here: a slow image pull (60 to 90 minutes has been observed on
metal) or a `/workspace` mount split-brain on the main pod, the same condition
`pod_entry.sh` warns about on its own. If the state clears on its own within roughly
90 minutes, it was the slow pull. If it is still showing past that, treat it as the
split-brain case and check the main pod's own console log for the mount warning.

## Re-running the freshness drill

The numbers that belong in the Freshness section above come from a live drill: a
throwaway CPU "writer" pod that reproduces both log-writing patterns without spending
any GPU time, with the reader pointed at it as if it were a real burn.

1. Launch a writer pod that mimics the main pod's two channels on a scratch path (not
   the real artifacts dir): a held-open, flush-no-close stdout tee through the real
   `log_tee.py` module, and an open-append-close heartbeat line every few seconds,
   both stamped `HH:MM:SS`.
2. Point `scripts/reader_pod.sh` at the writer pod's id, exactly as you would for a
   real main pod.
3. Watch the dataset's `runs/<writer_id>/live/` tree in a browser. Each cycle,
   compute lag per channel as `status.json`'s `cycle_utc` minus the last line's own
   timestamp in that channel's log. Record at least 5 cycles for both `stdout.log`
   and `heartbeat.log`.
4. Let the writer run to completion and self-delete (the same DELETE-self mechanism
   the reader itself uses). Within TWO more reader cycles (gone must be observed on
   two consecutive cycles) you should see `READER_DONE`
   committed and the reader pod itself gone (a REST `GET` on its id returns 404).
5. Clean up: delete the drill's scratch directory from the volume from a throwaway
   pod, never from the reader itself, and confirm zero pods remain afterward.

The full writer script and environment for this drill live in the internal
implementation plan for this feature (not shipped in this snapshot).
