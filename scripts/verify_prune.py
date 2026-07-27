"""Assert the image prune actually removed what it claims to have removed.

Run inside the builder stage, AFTER the `pip uninstall` loop:

    python3.12 scripts/verify_prune.py

Why this exists
---------------
The prune step used to end every `pip uninstall` in `|| true`, so a package name that
matched nothing exited silently. Three packages the list believed it had removed were
still shipping in the 0.6.0 image:

    nvidia-cutlass-dsl-libs-base   (the list named `nvidia-cutlass-dsl`)
    tokenspeed-mla                 (the list named `tokenspeed-triton`)
    pynvvideocodec                 (never listed at all - a video codec in a
                                    text-only eval image, the exact sibling of the
                                    torchcodec problem the prune exists to solve)

The lesson is that a hand-maintained list of exact distribution names cannot hold: it
drifts every time a dependency is bumped, and the drift is invisible. So this check does
NOT verify the list. It scans every installed distribution and fails if any name matches a
banned FAMILY token, which catches a renamed sibling or a newly-introduced one without
anyone editing anything.

Exit code 0 and a one-line confirmation on success; non-zero with an actionable message
naming the offending distributions on failure.
"""
from __future__ import annotations

import sys
from importlib.metadata import distributions

# Matched as substrings against the normalized distribution name. Substring rather than
# exact match is the entire point - it is the sibling packages that got us, not the ones
# we remembered to name.
#
# Each token, and why it is safe to remove from a TEXT-ONLY NVFP4 eval image:
#   torchcodec   - dlopens FFmpeg at import; ABSENT is required, not merely tidy (a
#                  present-but-broken torchcodec raises RuntimeError, which vLLM's
#                  ImportError guard does not catch, crashing the whole eval import)
#   flashinfer   - JIT paths need nvcc, absent here; VLLM_USE_FLASHINFER_SAMPLER=0
#                  short-circuits before import
#   cutlass      - mamba / DeepSeek-MoE backend, not reached by Qwen dense
#   tilelang     - same
#   tokenspeed   - same
#   z3-solver    - off-path constraint solver
#   opencv       - multimodal try/except path only
#   videocodec   - video decode, never used by a text-only battery
#
# DEFERRED FUSE: cutlass / tilelang / tokenspeed are safe because our recipes are DENSE
# models. If a future quant candidate is an MoE architecture, one of these backends may
# become load-bearing, and it will fail AFTER a paid quantize at eval-engine start - a
# model-dependent lazy import that no build-time gate can catch. Re-check this list
# against the architecture before the first burn of any non-dense model.
BANNED_FAMILIES = (
    "torchcodec",
    "flashinfer",
    "cutlass",
    "tilelang",
    "tokenspeed",
    "z3-solver",
    "opencv",
    "videocodec",
)


def normalize(name: str) -> str:
    """Fold a distribution name the way packaging metadata comparison does.

    importlib.metadata already folds case and separators for lookup, but we are matching
    substrings against raw metadata names here, so do it explicitly.
    """
    return name.replace("_", "-").replace(".", "-").lower()


def installed_names(dists=None) -> list[str]:
    """Every installed distribution name. `dists` is a test seam."""
    if dists is None:
        dists = distributions()
    names = []
    for dist in dists:
        # A malformed dist can have no Name; skip rather than crash the build on it.
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            names.append(name)
    return names


def survivors(names, families=BANNED_FAMILIES) -> list[str]:
    """Installed distributions whose name matches a banned family token.

    Deduplicated and sorted so the build log is stable and diffable.
    """
    return sorted({n for n in names
                   if any(f in normalize(n) for f in families)})


def format_failure(hits) -> str:
    """The 2 AM message: say what is wrong, what it costs, and what to do next."""
    listed = "".join(f"  - {h}\n" for h in hits)
    return (
        "PRUNE FAILED: these installed distributions match a banned family:\n"
        f"{listed}"
        "\n"
        "They are not used by the text-only NVFP4 eval path, so they would ship as dead\n"
        "weight in every image pull.\n"
        "\n"
        "Most likely a dependency bump renamed a package or introduced a new sibling\n"
        "(this check exists because nvidia-cutlass-dsl-libs-base, tokenspeed-mla and\n"
        "pynvvideocodec all slipped through a list of exact names).\n"
        "\n"
        "To fix: add each name above to PRUNE in deploy/Dockerfile, rebuild, re-check.\n"
        "If one is genuinely load-bearing now - an MoE recipe needing a backend we used\n"
        "to prune, say - remove its token from BANNED_FAMILIES here and write down why.\n"
        "Do not silence this by widening the list without a reason."
    )


def main(argv=None) -> int:
    hits = survivors(installed_names())
    if hits:
        print(format_failure(hits), file=sys.stderr)
        return 1
    print(f"prune verified: no distribution matches {len(BANNED_FAMILIES)} banned families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
