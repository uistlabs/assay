"""What environment produced this measurement (F-009, spec 2026-07-28).

Capture NEVER initializes CUDA (spec D7): nvidia-smi + importlib.metadata +
torch module attributes only, so capture cannot perturb the measured process
and rehearses on GPU-less / pre-cu129 hosts (the 1070)."""
from __future__ import annotations
import json
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from importlib.metadata import version as _dist_version, PackageNotFoundError

SCHEMA_VERSION = 1
VERDICTS = ("clean", "void", "not-applicable", "not-captured")

NVSMI_QUERY = ("name,memory.total,driver_version,ecc.mode.current,"
               "ecc.errors.uncorrected.volatile.total,"
               "ecc.errors.corrected.volatile.total")
_NA = {"[N/A]", "[Not Supported]", "N/A"}


class PinMismatchError(RuntimeError):
    """The measured environment does not match deploy/constraints.txt. Raised
    BEFORE GPU spend: a run that cannot prove its stack must not burn."""


class HardwareCaptureError(RuntimeError):
    """Host introspection failed. Fatal at begin-capture on every run: a
    cert-tier run that cannot read its hardware cannot honor the ECC policy."""


class ManifestEnvError(RuntimeError):
    """ASSAY_IMAGE or ASSAY_BUILD_SHA missing from env. PinMismatchError-style
    hard fail (spec D10): an unidentifiable image cannot be certified, so
    begin_capture refuses BEFORE GPU spend rather than stamping a manifest
    that cannot be traced back to a build."""


def _parse_constraints(path: str) -> dict[str, str]:
    """Parse deploy/constraints.txt, keeping ALL pins (including image-only)."""
    pins: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or "==" not in line:
                continue
            name, ver = line.split("==", 1)
            pins[name.strip()] = ver.strip()
    return pins


def image_only_names(constraints_path: str) -> frozenset[str]:
    """Extract names of pins marked `# image-only` (F-030: cu129 GPU stack).

    These packages cannot be installed in dev environments (sm_61 lacks cu129
    support) but are asserted in-pod where they ARE installed.
    """
    names: set[str] = set()
    with open(constraints_path, encoding="utf-8") as fh:
        for raw in fh:
            if "# image-only" in raw:
                line = raw.split("#")[0].strip()
                if line and "==" in line:
                    name = line.split("==")[0].strip()
                    names.add(name)
    return frozenset(names)


def _nvsmi(run):
    out = run(["nvidia-smi", f"--query-gpu={NVSMI_QUERY}",
               "--format=csv,noheader,nounits"],
              capture_output=True, text=True, timeout=10, check=False)
    if getattr(out, "returncode", 1) != 0 or not out.stdout.strip():
        raise HardwareCaptureError(f"nvidia-smi query failed: rc="
                                   f"{getattr(out, 'returncode', 'n/a')}")
    return out.stdout.strip().splitlines()[0]


def parse_nvsmi_line(line: str) -> dict:
    parts = [p.strip() for p in line.split(", ")]
    if len(parts) != 6:
        raise HardwareCaptureError(f"unexpected nvidia-smi shape: {line!r}")
    name, mem, driver, ecc_mode, unc, corr = parts
    supported = ecc_mode not in _NA
    counters = None
    if supported and unc not in _NA and corr not in _NA:
        counters = (int(unc), int(corr))
    return {"gpu_name": name, "vram_total_mib": int(mem),
            "driver_version": driver, "ecc_supported": supported,
            "ecc_enabled": (ecc_mode == "Enabled") if supported else None,
            "ecc_counters": counters}


def _cuda_driver_version() -> str:
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        return f"{v // 1000}.{(v % 1000) // 10}"
    except Exception:
        return "not-captured"


def capture_hardware(gpu_mem_util: float, run=subprocess.run) -> Hardware:
    try:
        d = parse_nvsmi_line(_nvsmi(run))
    except HardwareCaptureError:
        raise
    except Exception as exc:
        raise HardwareCaptureError(str(exc)) from exc
    return Hardware(gpu_name=d["gpu_name"], vram_total_mib=d["vram_total_mib"],
                    driver_version=d["driver_version"],
                    cuda_driver=_cuda_driver_version(),
                    ecc_supported=d["ecc_supported"],
                    ecc_enabled=d["ecc_enabled"], gpu_mem_util=gpu_mem_util)


def read_ecc_counters(run=subprocess.run) -> tuple[int, int] | None:
    try:
        return parse_nvsmi_line(_nvsmi(run))["ecc_counters"]
    except Exception:
        return None


def build_ecc_window(ecc_supported: bool,
                     begin: tuple[int, int] | None,
                     end: tuple[int, int] | None) -> EccWindow:
    """Build an ECC window verdict from counter reads.

    A counter reset window (end < begin on either counter) must yield
    "not-captured" - never a negative delta read as clean, because a reset
    window hides errors (spec D9).

    Args:
        ecc_supported: Whether the hardware supports ECC.
        begin: (uncorrected, corrected) counter tuple at window start, or None.
        end: (uncorrected, corrected) counter tuple at window end, or None.

    Returns:
        EccWindow with verdict in VERDICTS.
    """
    if not ecc_supported:
        return EccWindow(None, None, None, None, "not-applicable")
    if begin is None or end is None or end[0] < begin[0] or end[1] < begin[1]:
        # A reset window hides errors; never read it as clean (spec D9).
        return EccWindow(begin, end, None, None, "not-captured")
    unc, corr = end[0] - begin[0], end[1] - begin[1]
    verdict = "void" if unc > 0 else "clean"
    assert verdict in VERDICTS, f"verdict {verdict!r} not in VERDICTS"
    return EccWindow(begin, end, unc, corr, verdict)


def capture_stack(constraints_path: str, *, exclude: frozenset[str] = frozenset()) -> tuple[StackPin, ...]:
    """Capture installed stack against constraints.txt and cross-assert.

    Raises PinMismatchError naming every offender BEFORE GPU spend.

    Args:
        constraints_path: Path to deploy/constraints.txt
        exclude: Frozenset of package names to skip (e.g., image-only pins
                 in dev environments that lack cu129 support). In-pod callers
                 pass exclude=frozenset() to assert EVERY pin including GPU stack.
    """
    offenders: list[str] = []
    out: list[StackPin] = []
    for name, pinned in sorted(_parse_constraints(constraints_path).items()):
        if name in exclude:
            continue
        try:
            observed = _dist_version(name)
        except PackageNotFoundError:
            offenders.append(f"{name}: pinned {pinned}, NOT INSTALLED")
            continue
        if observed != pinned:
            offenders.append(f"{name}: pinned {pinned}, observed {observed}")
        out.append(StackPin(name=name, pinned=pinned, observed=observed))
    if offenders:
        raise PinMismatchError(
            "manifest stack assert failed (fix the image or the pins, never "
            "override): " + "; ".join(offenders))
    return tuple(out)


def begin_capture(env, gpu_mem_util: float, constraints_path: str = "deploy/constraints.txt",
                  run=subprocess.run) -> "BeginCapture":
    """Pre-GPU capture (spec capture point 1): stack cross-assert, then hardware +
    ECC-counter baseline. No exclude param on purpose (spec D5/D7 resolution) - every
    real run of this (in-pod, or in-image rehearsal) holds every pin including the
    image-only cu129 stack, so the default assert-everything capture_stack() path is
    always correct here. Missing image identity (ASSAY_IMAGE/ASSAY_BUILD_SHA) is
    checked FIRST, before any I/O, because an unidentifiable image cannot be
    certified regardless of what the stack/hardware capture would show (D10)."""
    missing = [k for k in ("ASSAY_IMAGE", "ASSAY_BUILD_SHA") if not env.get(k)]
    if missing:
        raise ManifestEnvError(
            "manifest capture requires " + " and ".join(missing) +
            " in env - an unidentifiable image cannot be certified")
    begin_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stack = capture_stack(constraints_path)
    hardware = capture_hardware(gpu_mem_util, run=run)
    counters_begin = read_ecc_counters(run=run)
    if hardware.ecc_supported and counters_begin is None:
        # whole-branch review FIX 1: a flaked/unsupported counter read on
        # ECC-capable hardware must die HERE, before GPU spend - not silently
        # disarm the mid-run fail-fast (watchdog.build_eval_watchdog only arms
        # ecc_begin when it is not None) and guarantee an end-of-run void after
        # hours of paid GPU instead (spec capture-point 1's whole promise).
        raise HardwareCaptureError(
            "ECC-capable GPU but volatile counters unreadable - a cert run that "
            "cannot read its counters cannot honor the ECC policy; check "
            "nvidia-smi ecc.errors.* support on this SKU")
    import torch  # noqa: PLC0415 - lazy: module attribute only, never initializes CUDA (D7)
    cuda_runtime = torch.version.cuda or "not-captured"
    return BeginCapture(
        image=env["ASSAY_IMAGE"], build_sha=env["ASSAY_BUILD_SHA"], stack=stack,
        python=platform.python_version(), cuda_runtime=cuda_runtime, hardware=hardware,
        counters_begin=counters_begin, begin_utc=begin_utc,
        queries=(f"nvidia-smi --query-gpu={NVSMI_QUERY}",),
    )


def finalize(begin: "BeginCapture", run=subprocess.run) -> ManifestV1:
    """Post-eval, pre-publish capture (spec capture point 3): re-reads the ECC
    counters, computes the window verdict against the begin-capture baseline, and
    stamps end_utc - the manifest is not final (and not gate-inspectable) until
    this runs."""
    counters_end = read_ecc_counters(run=run)
    ecc_window = build_ecc_window(begin.hardware.ecc_supported, begin.counters_begin,
                                  counters_end)
    end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ManifestV1(
        schema_version=SCHEMA_VERSION, image=begin.image, build_sha=begin.build_sha,
        stack=begin.stack, python=begin.python, cuda_runtime=begin.cuda_runtime,
        hardware=begin.hardware, ecc_window=ecc_window,
        capture=Capture(begin_utc=begin.begin_utc, end_utc=end_utc,
                        tool_queries=begin.queries),
    )


@dataclass(frozen=True)
class StackPin:
    name: str
    pinned: str
    observed: str


@dataclass(frozen=True)
class Hardware:
    gpu_name: str
    vram_total_mib: int
    driver_version: str
    cuda_driver: str          # "not-captured" when unreadable (non-fatal field)
    ecc_supported: bool
    ecc_enabled: bool | None  # None where unsupported
    gpu_mem_util: float


@dataclass(frozen=True)
class BeginCapture:
    """Everything captured pre-GPU (spec capture point 1), carried forward so
    finalize() needs no I/O beyond the end-of-window ECC re-read - image/build_sha/
    python/cuda_runtime/stack/hardware are frozen at begin time and never re-measured."""
    image: str
    build_sha: str
    stack: tuple[StackPin, ...]
    python: str
    cuda_runtime: str
    hardware: Hardware
    counters_begin: tuple[int, int] | None
    begin_utc: str
    queries: tuple[str, ...]


@dataclass(frozen=True)
class EccWindow:
    counters_begin: tuple[int, int] | None  # (uncorrected, corrected)
    counters_end: tuple[int, int] | None
    uncorrected_delta: int | None
    corrected_delta: int | None
    verdict: str  # one of VERDICTS


@dataclass(frozen=True)
class Capture:
    begin_utc: str
    end_utc: str | None
    tool_queries: tuple[str, ...]


@dataclass(frozen=True)
class ManifestV1:
    schema_version: int
    image: str
    build_sha: str
    stack: tuple[StackPin, ...]
    python: str
    cuda_runtime: str
    hardware: Hardware
    ecc_window: EccWindow
    capture: Capture

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ManifestV1":
        d = json.loads(s)
        d["stack"] = tuple(StackPin(**p) for p in d["stack"])
        d["hardware"] = Hardware(**d["hardware"])
        ew = d["ecc_window"]
        for k in ("counters_begin", "counters_end"):
            if ew[k] is not None:
                ew[k] = tuple(ew[k])
        d["ecc_window"] = EccWindow(**ew)
        cap = d["capture"]
        cap["tool_queries"] = tuple(cap["tool_queries"])
        d["capture"] = Capture(**cap)
        return cls(**d)
