"""Partition a directory tree into N balanced staging buckets, one per image layer.

    python3.12 scripts/plan_blobs.py --root /usr/local --buckets 9 --stage /stage [--dry-run]

Why this exists
---------------
An image layer is a blob, and a blob is the unit of pull RETRY. The 2026-07-26 cert run
pulled for ~50 minutes at ~80 Mbps and died on `unexpected EOF`, when a 7.13 GB pull at
that rate should take ~12 minutes. The difference was retry amplification: the torch layer
was one 4.81 GB compressed blob, and a drop anywhere inside it re-fetches ALL of it.
Splitting the tree across several layers turns a restart-everything failure into a
retry-that-piece failure.

Why it is COMPUTED rather than a hand-written list
--------------------------------------------------
The obvious implementation is a list of directories transcribed from one measurement:

    stage 01 nvidia/cudnn
    stage 02 nvidia/cublas
    ...

That works exactly once. Every torch or vLLM bump renames, adds, splits or resizes those
directories, and then someone has to re-measure the tree by hand and retune the list
before the image will build - with no guidance on how. It is a maintenance trap dressed
up as simplicity, and dependency bumps are the single most common maintenance event this
image has.

So the split is derived from the tree instead. Bump a dependency and the packing simply
re-balances; nobody edits anything. The only fixed quantity is the NUMBER of buckets,
which must match the number of COPY instructions in deploy/Dockerfile - and a mismatch
there fails the build loudly rather than dropping content.

Determinism
-----------
Same tree in, same partition out. Units are sorted by (size descending, path) so ties
break on path rather than on filesystem iteration order, and the packer is plain greedy.
This matters for layer caching: an unchanged tree must produce byte-identical layers, or
every rebuild would invalidate the cache for no reason.

Sizing
------
Sizes are APPARENT bytes (`st_size`), never block counts. `du` without `--apparent-size`
reports compressed blocks when the container store is on lz4 ZFS, which understated this
very image's live tree as 8.0 GB when it is 12.0 GB. Blocks would also make the packing
depend on which host built the image.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def tree_size(path: str) -> int:
    """Total apparent bytes of the FILES and SYMLINKS under `path`.

    Directory inodes are deliberately NOT counted, and that is load-bearing rather than a
    rounding choice: this function has to be ADDITIVE OVER A PARTITION. The packer splits
    a tree into units and the manifest check later asserts that the reassembled tree has
    the same total, so `tree_size(parent)` must equal the sum of `tree_size(child)` for
    its children. Counting directory inodes breaks that - the parent counts each child
    directory's own inode, but `tree_size(child)` does not count the child's own inode, so
    every level of nesting loses bytes and the totals silently disagree. Counting only
    file content makes the measure exactly additive.

    Symlinks count at their own (tiny) size rather than their target's, so a link farm
    cannot inflate a bucket or double-count a real file. Note os.walk lists symlinks that
    point at directories in `dirnames`, not `filenames`, and with followlinks=False it
    never visits them - so they are picked up explicitly below or they would vanish from
    the total.
    """
    if os.path.islink(path) or os.path.isfile(path):
        return os.lstat(path).st_size
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass  # vanished mid-walk; not worth failing a build over
        for name in dirnames:
            full = os.path.join(dirpath, name)
            try:
                if os.path.islink(full):
                    total += os.lstat(full).st_size
            except OSError:
                pass
    return total


def collect_units(root: str, max_unit_bytes: int, _depth: int = 0,
                  max_depth: int = 12) -> list[tuple[str, int]]:
    """Split `root` into (path, size) units no larger than max_unit_bytes where possible.

    Descends into any directory bigger than the threshold, so a single huge package like
    site-packages/nvidia (~5.0 GiB) is broken into its children rather than pinning an
    entire bucket. Stops when a directory is under the threshold, or has no children to
    split - a single 2 GB file is indivisible and is returned as an oversized unit, which
    the packer then places on its own.

    max_depth is ONLY a runaway guard against a pathological tree, NOT the operative
    control - size is. It is set well clear of any real layout for a reason: at the
    default of 4 it silently defeated the entire mechanism. /usr/local/lib/python3.12/
    site-packages/nvidia sits at exactly depth 4, so the limit fired before the size
    threshold could, and nvidia came back as one 5.0 GiB unit - a bigger blob than the one
    whose failure motivated this whole file. Raise this before lowering it.

    Returned units are always DISJOINT and together cover `root` exactly. That is what
    makes completeness structural: nothing can be omitted by a bad list, because there is
    no list.
    """
    size = tree_size(root)
    if size <= max_unit_bytes or _depth >= max_depth:
        return [(root, size)]
    try:
        children = sorted(os.listdir(root))
    except OSError:
        return [(root, size)]
    if not children:
        return [(root, size)]
    units: list[tuple[str, int]] = []
    for child in children:
        units.extend(collect_units(os.path.join(root, child), max_unit_bytes,
                                   _depth + 1, max_depth))
    return units


def pack(units: list[tuple[str, int]], buckets: int) -> list[list[tuple[str, int]]]:
    """Greedy longest-processing-time-first bin packing into exactly `buckets` bins.

    Sort units largest-first, then repeatedly place the next unit into whichever bin is
    currently smallest. Not optimal - optimal bin packing is NP-hard and we do not need
    optimal, we need "no bin is dramatically larger than the others". LPT is the standard
    cheap approximation and gets within a few percent in practice.

    Ties break on path so the result does not depend on directory iteration order.
    """
    if buckets < 1:
        raise ValueError(f"buckets must be >= 1, got {buckets}")
    ordered = sorted(units, key=lambda u: (-u[1], u[0]))
    bins: list[list[tuple[str, int]]] = [[] for _ in range(buckets)]
    totals = [0] * buckets
    for path, size in ordered:
        target = totals.index(min(totals))
        bins[target].append((path, size))
        totals[target] += size
    return bins


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GiB"


def stage_path(stage_root: str, index: int, path: str) -> str:
    """Destination for `path` inside bucket `index`, preserving its absolute location.

    /usr/local/bin -> /stage/03/usr/local/bin, so that `COPY --from=builder /stage/03/ /`
    puts every file back exactly where it was.
    """
    return os.path.join(stage_root, f"{index:02d}", path.lstrip("/"))


def apply_plan(bins, stage_root: str, dry_run: bool) -> None:
    for index, units in enumerate(bins, start=1):
        total = sum(size for _, size in units)
        print(f"  stage {index:02d}: {human(total):>9}  ({len(units)} units)")
        for path, size in sorted(units):
            print(f"      {human(size):>9}  {path}")
            if dry_run:
                continue
            dest = stage_path(stage_root, index, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(path, dest)


MANIFEST_NAME = "manifest.txt"


def write_manifest(stage_root: str, total: int, buckets: int) -> str:
    """Record what the staged tree is supposed to add up to.

    This is what makes a wrong bucket count LOUD. The runtime stage has one COPY per
    bucket; if those two numbers ever disagree - someone adds a bucket without adding a
    COPY, or vice versa - the missing bucket's files are simply absent from the image, and
    nothing else in the build necessarily notices. `--verify` compares the reassembled
    tree against this total and fails the build instead.
    """
    os.makedirs(stage_root, exist_ok=True)
    path = os.path.join(stage_root, MANIFEST_NAME)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"total_bytes {total}\nbuckets {buckets}\n")
    return path


def read_manifest(path: str) -> dict:
    values = {}
    with open(path, encoding="ascii") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = int(parts[1])
    return values


def verify(root: str, manifest_path: str) -> tuple[bool, str]:
    """True iff `root` holds exactly the bytes the manifest recorded."""
    expected = read_manifest(manifest_path)["total_bytes"]
    actual = tree_size(root)
    if actual == expected:
        return True, f"blob reassembly verified: {root} = {human(actual)}, matches manifest"
    delta = expected - actual
    return False, (
        f"BLOB REASSEMBLY FAILED: {root} holds {human(actual)} but the builder staged "
        f"{human(expected)} ({human(abs(delta))} {'missing' if delta > 0 else 'extra'}).\n"
        "\n"
        "Almost certainly the number of `COPY --from=builder /stage/NN/ /` instructions in\n"
        "deploy/Dockerfile does not match --buckets passed to plan_blobs.py. Every bucket\n"
        "needs exactly one COPY; a bucket without one is silently absent from the image.\n"
        "\n"
        "Count the COPY lines, make --buckets equal that number, and rebuild."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="tree to partition, or to check when --verify is given")
    ap.add_argument("--buckets", type=int,
                    help="number of buckets; MUST equal the number of COPY --from=builder "
                         "instructions in deploy/Dockerfile")
    ap.add_argument("--stage", default="/stage",
                    help="staging root to create bucket dirs under (default /stage)")
    ap.add_argument("--max-unit-mib", type=int, default=None,
                    help="descend into any directory larger than this; default is derived "
                         "as half the ideal bucket size, so it scales with the tree")
    ap.add_argument("--max-bucket-mib", type=int, default=None,
                    help="FAIL if any bucket exceeds this. This is the whole point of the "
                         "exercise - a single oversized blob restarts from zero on a "
                         "dropped pull - so it is enforced, not merely reported")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without moving anything")
    ap.add_argument("--verify", metavar="MANIFEST",
                    help="check --root against a manifest written by an earlier run "
                         "instead of partitioning anything")
    args = ap.parse_args(argv)

    if args.verify:
        ok, message = verify(args.root, args.verify)
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    if args.buckets is None:
        print("plan_blobs: --buckets is required unless --verify is given", file=sys.stderr)
        return 1
    if not os.path.isdir(args.root):
        print(f"plan_blobs: --root {args.root} is not a directory", file=sys.stderr)
        return 1

    # Derive the split threshold from the tree rather than hardcoding it. Half the ideal
    # bucket size means a unit can never occupy more than ~50% of a bucket on its own,
    # which gives the packer room to balance. Deriving it also means the threshold keeps
    # working as the image grows - a fixed constant silently stops splitting enough the
    # moment a dependency doubles in size.
    total = tree_size(args.root)
    if args.max_unit_mib is not None:
        max_unit = args.max_unit_mib * 1024 * 1024
    else:
        max_unit = max(total // args.buckets // 2, 1024 * 1024)

    units = collect_units(args.root, max_unit)
    print(f"plan_blobs: {len(units)} units, {human(total)} total, "
          f"into {args.buckets} buckets (split threshold {human(max_unit)})")
    bins = pack(units, args.buckets)
    apply_plan(bins, args.stage, args.dry_run)
    if not args.dry_run:
        write_manifest(args.stage, total, args.buckets)

    totals = [sum(size for _, size in b) for b in bins]
    largest = max(totals) if totals else 0
    print(f"plan_blobs: largest bucket {human(largest)}, "
          f"ideal {human(total // args.buckets)}")

    empty = [i for i, b in enumerate(bins, start=1) if not b]
    if empty:
        # Not fatal - an empty bucket still COPYs cleanly - but it means --buckets is
        # larger than the tree can usefully split, which is worth saying out loud.
        print(f"plan_blobs: NOTE - buckets {empty} are empty; consider fewer buckets "
              f"or a smaller --max-unit-mib", file=sys.stderr)

    # The enforced acceptance criterion. Everything else in this script is machinery in
    # service of this one number: no blob so large that a dropped pull throws away a
    # ruinous amount of transfer. Reporting it is not enough - a build that quietly
    # produces a 5 GiB blob looks exactly like a build that worked.
    if args.max_bucket_mib is not None and largest > args.max_bucket_mib * 1024 * 1024:
        print(
            f"\nBLOB TOO LARGE: biggest bucket is {human(largest)}, limit is "
            f"{human(args.max_bucket_mib * 1024 * 1024)}.\n"
            "\n"
            "A dropped pull re-fetches an ENTIRE blob, so one oversized layer reproduces\n"
            "the failure this splitting exists to prevent.\n"
            "\n"
            "Usually this means one directory is too big to split at the current settings.\n"
            "The plan printed above names it. Options, in order of preference:\n"
            "  1. lower --max-unit-mib so the packer descends further into it\n"
            "  2. raise --buckets (and add a matching COPY --from=builder in the Dockerfile)\n"
            "  3. if it is a single indivisible FILE, it cannot be split - raise\n"
            "     --max-bucket-mib above its size and note why.",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
