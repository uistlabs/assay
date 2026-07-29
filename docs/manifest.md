# The stack + hardware manifest (F-009)

Every cert run captures a manifest describing the environment that produced the
measurement: the exact image, the exact installed package versions, the GPU, and
whether that GPU's memory reported any errors during the run. This exists because a
certification asserts a measurement, and a measurement is only meaningful against the
environment that produced it - a deterministic metric (wikitext perplexity) has been
observed to move 0.46% across two assay releases purely because the stack moved
underneath it. Code: `src/assay/manifest.py`.

## What the manifest claims

One versioned object, `ManifestV1` (`schema_version: 1`, future fields append and never
mutate), rendered on three surfaces from a single source (a card section, `manifest.json`
in the checkpoint repo, and a copy in the run's artifacts trail) so wording cannot drift
between them:

| group | fields | what it proves |
|---|---|---|
| `image` | digest-pinned ref, `build_sha` | which exact image and commit ran - `ASSAY_IMAGE` is already digest-pinned by construction (`launch.sh` refuses a non-digest ref), so this field cannot silently mean "whatever `latest` was that day" |
| `stack` | per-pin `{name, pinned, observed}` for every `deploy/constraints.txt` entry, plus `python`, `cuda_runtime` | both the required and the actually-installed version, side by side - self-evidencing, not a "trust me" |
| `hardware` | `gpu_name`, `vram_total`, `driver_version`, `cuda_driver`, `ecc_supported`, `ecc_enabled`, `gpu_mem_util` | which card ran the measurement, and whether that card's memory has error protection at all |
| `ecc_window` | `counters_begin`, `counters_end`, `uncorrected_delta`, `corrected_delta`, `verdict` | whether the GPU reported any memory errors during the measurement window |
| `capture` | begin/end UTC timestamps, the exact `nvidia-smi` query strings used | when the capture happened and how, so the tool provenance itself is inspectable |

Capture happens at three points and never initializes CUDA (D7 - nvidia-smi is
driver-level, `importlib.metadata` and `torch.version.cuda` are metadata reads, not
device init): once pre-GPU (stack cross-assert + hardware/ECC baseline, before any paid
GPU-second is spent), once per watchdog tick during the run (ECC counter only, D6), and
once post-eval pre-publish (final ECC read, deltas computed, manifest finalized). A stack
mismatch or a hardware-introspection failure on ECC-capable hardware dies at the first
point, before GPU spend - see `ManifestEnvError`, `PinMismatchError`,
`HardwareCaptureError` in `src/assay/manifest.py`.

## The ECC policy (D3 - Ken's ruling; do not re-litigate)

Any uncorrected memory error during the measurement window voids the run. Corrected
errors are disclosed but never fail the gate - ECC correcting an error is the hardware
doing its job, and no corrected-error threshold is invented without data; if one is ever
designed, it comes from accumulated manifest history, not a guess made today.

| verdict | when | gate outcome | card / disclosure wording (verbatim, `publish.py:_ECC_VERDICT_SENTENCES` - do not reword) |
|---|---|---|---|
| `clean` | ECC-capable hardware, window closed, zero uncorrected errors | PASS (corrected errors, if any, are disclosed only) | "ECC was enabled for this measurement window; no uncorrected or corrected memory errors were observed (verdict: clean)." |
| `void` | ECC-capable hardware, `uncorrected_delta > 0` at any point in the window | HARD FAIL - no cert mints, re-run required | "ECC recorded an uncorrected memory error during this measurement window (verdict: void); the certification for this run is void." |
| `not-captured` | ECC-capable hardware, but the counters could not be read cleanly at begin or end (includes a counter-reset window: `end < begin` is NEVER read as a negative, clean delta) | HARD FAIL, same as `void` - absence of promised evidence is never clean | "The ECC error counters for this measurement window could not be read cleanly (verdict: not-captured); no ECC claim is made for this run." |
| `not-applicable` | hardware has no ECC support at all (e.g. the 5090's consumer GDDR7) | PASS - non-ECC hardware never voids on this check | "This measurement ran on hardware without memory-error protection (no ECC)." |

Existing 5090 certs stay valid under this policy: the gate's sampling floor already
dominates single-flip effects and the paired statistical design resists asymmetric bias.
Their re-certed cards simply state the hardware honestly (`not-applicable`) rather than
implying a guarantee that was never there.

## The ECC_VOID path at 2 AM

If a run voids mid-flight, here is what you will actually see and what to do about it.

**During the run:** the watchdog's existing poll loop carries the uncorrected counter
(`src/assay/watchdog.py:_ecc_fatal_check`). The instant a positive delta is read, it
fires the same abort path as a progress stall, but with the reason string
`ECC_VOID: uncorrected errors during measurement (N)`. That reason is written to the
heartbeat log before the kill, so on the reader pod's `status.json` you will see it as
the last `phase` line (a `stall`-stage heartbeat entry carrying the `ECC_VOID` text),
followed shortly by the main pod going terminal - reader `state` moves to `final`, and a
`traceback.txt` lands next to `status.json` from the run's own forensics upload. This is
not a bug and not a stall you can nurse along: an uncorrected error is a terminal fact
under the D3 policy, not a warning (this does not conflict with the standing
"warnings-are-signals, never a mid-run kill switch" rule - every GPU-second after an
uncorrected error is spend on a measurement that is already void, so killing it early is
strictly better than killing it later).

**If the run instead reaches the end of the window still voided** (verdict `void`, or
`not-captured` on ECC-capable hardware), `apply_ecc_policy` (`src/assay/job.py`) rewrites
the gate result to failed with the ECC detail in `reasons`, before publish ever sees it.
`publish_if_passed` (`src/assay/publish.py`) short-circuits on any failed gate before
building a card or touching HF at all - so a void run never uploads anything, never
touches the live checkpoint repo, and there is no card to clean up afterward. The pod's
existing `finally` teardown still runs: artifacts (including `manifest.json` and
`traceback.txt` if one exists) upload to the artifacts dataset, then the pod
self-terminates exactly like any other finished run.

**What to do, standing at 2 AM watching this happen: nothing, then re-launch.** There is
no forensics step that fixes a void - it is a hardware fault, not a code bug or a data
problem. Let the pod finish self-terminating (it will; no manual kill needed unless it
hangs past the normal teardown window, in which case treat it like any other stuck pod).
Confirm 0 assay pods remain (or, in fleet mode, that the survivors match
`ASSAY_FLEET_EXPECTED`), then re-launch that one recipe per whatever launch sheet is
currently in force for this cert cycle. Nothing about the void changes the launch
config - same image digest, same env, same recipe.

## Fixture-capture instructions for new hardware (the H100 first-run capture)

Unit tests parse manifests against captured `nvidia-smi` output from real hardware -
never invented fixtures (`feedback_captured_fixtures`; spec D9). Two fixtures exist
today: `tests/fixtures/nvidia_smi/gtx1070_query.csv` (ECC-absent, this box) and the 5090
fixture from its own re-cert burn. The H100 fixture is deliberately **pending capture** -
`tests/test_manifest.py::test_h100_fixture_pending_capture` is skipped rather than
faked, and it stays skipped until this step actually runs on rented H100/H200 hardware.

When the first rented-HBM run lands, capture the fixture like this, in-pod, before
tearing the pod down:

1. Run the exact query the manifest module uses (`NVSMI_QUERY` in
   `src/assay/manifest.py`) - do not hand-type a different set of fields, or the parser's
   column-count assert (`parse_nvsmi_line`, expects exactly 6 comma-separated fields)
   will reject it:

   ```
   nvidia-smi --query-gpu=name,memory.total,driver_version,ecc.mode.current,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total --format=csv,noheader,nounits
   ```

2. Save the single output line verbatim (no editing, no re-formatting) to
   `tests/fixtures/nvidia_smi/h100_query.csv` in the repo, matching the naming and
   one-line-CSV shape of the existing `gtx1070_query.csv`.
3. Pull that file back to the workstation (it rides the pod's artifacts trail, or copy it
   out directly - it contains no secrets, just hardware telemetry).
4. In `tests/test_manifest.py`, replace the body of `test_h100_fixture_pending_capture`
   with a real assertion against the new fixture, following the shape of
   `test_parse_captured_1070_line` immediately above it (parse the line, assert
   `ecc_supported is True` this time, and assert the ECC counter fields parse to
   integers rather than `None`, since H100/H200 carry always-on HBM ECC).
5. Run the full suite; the new fixture-backed test must go green and no other test may
   regress.

## The D4 customer-communication rider

On a customer-lane run that voids on ECC: the customer is informed of the void and the
reason, and the run's cost still passes through to the customer. This is a hardware
fault (a memory error on the rented card), not a methodology fault (nothing about the
gate, the recipe, or the measurement protocol was wrong) - the distinction is what
justifies billing for a run that produced no cert. This is a process rule for the
operator, not something the code enforces or automates.

Notifying the provider (RunPod would plausibly want the evidence, to pull a host with a
real hardware fault out of rotation) is explicitly **future work**, deferred until the
first real void happens - see spec D4 and the "Out of scope / future work" section of
the design doc. Revisit it then, with a real void's actual evidence in hand, rather than
designing a notification path against a hypothetical one now.
