"""The blob packer decides how the image is split across layers, and a layer is the unit
of pull retry. Two properties carry all the weight:

  COMPLETENESS - the units must partition the tree exactly. Anything the packer drops is
  silently missing from the shipped image, and the failure surfaces at runtime on a rented
  GPU rather than at build time.

  DETERMINISM - the same tree must produce the same partition, or every rebuild
  invalidates layer caches for no reason.

Balance is the goal but only loosely testable; it is asserted as "no bin is wildly larger
than the mean" rather than against a fixed number.
"""
import importlib.util
import shutil
import os
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "plan_blobs.py"


def _load():
    """Import the script by path - it lives in scripts/, not in the assay package."""
    spec = importlib.util.spec_from_file_location("plan_blobs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plan_blobs = _load()


def _mktree(root: pathlib.Path, layout: dict) -> None:
    """layout maps relative path -> byte count."""
    for rel, size in layout.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * size)


@pytest.fixture
def tree(tmp_path):
    _mktree(tmp_path, {
        "big/a.bin": 4000,
        "big/b.bin": 4000,
        "mid/c.bin": 1500,
        "small/d.bin": 200,
        "small/nested/e.bin": 100,
        "loose.bin": 50,
    })
    return tmp_path


def test_units_cover_the_tree_exactly(tree):
    """Every byte lands in exactly one unit. This is the property that makes completeness
    structural rather than dependent on someone maintaining a correct list."""
    units = plan_blobs.collect_units(str(tree), max_unit_bytes=2000)
    total_in_units = sum(size for _, size in units)
    assert total_in_units == plan_blobs.tree_size(str(tree))

    # Disjoint: no unit may be a path prefix of another, or its bytes are counted twice
    # and the file would be COPYd into two layers.
    paths = sorted(p.rstrip("/") for p, _ in units)
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            assert not b.startswith(a + os.sep), f"{b} nested inside {a}"


def test_descends_only_into_oversized_directories(tree):
    """A directory under the threshold stays whole; an oversized one is split. Without
    this, one huge package pins an entire bucket and the split buys nothing."""
    units = dict(plan_blobs.collect_units(str(tree), max_unit_bytes=2000))
    # big/ is 8000 bytes -> must be broken into its two files
    assert str(tree / "big" / "a.bin") in units
    assert str(tree / "big") not in units
    # small/ is 300 bytes -> stays intact
    assert str(tree / "small") in units


def test_indivisible_oversized_file_is_returned_whole(tmp_path):
    """A single file larger than the threshold cannot be split. It must come back as one
    oversized unit rather than being dropped or causing infinite recursion."""
    _mktree(tmp_path, {"huge.bin": 9999})
    units = plan_blobs.collect_units(str(tmp_path), max_unit_bytes=100)
    assert units == [(str(tmp_path / "huge.bin"), 9999)]


def test_max_depth_stops_recursion(tmp_path):
    """Depth is bounded so a pathologically deep tree cannot blow the stack mid-build."""
    _mktree(tmp_path, {"a/b/c/d/e/f/g.bin": 5000})
    units = plan_blobs.collect_units(str(tmp_path), max_unit_bytes=1, max_depth=2)
    assert len(units) == 1
    # Stopped at depth 2 rather than walking to the leaf file.
    assert units[0][0].count(os.sep) < str(tmp_path / "a/b/c/d/e/f/g.bin").count(os.sep)


def test_descends_past_the_real_site_packages_depth(tmp_path):
    """REGRESSION. The first build of the multi-stage image looked completely green and
    still produced a 5.0 GiB blob - bigger than the one whose failed pull motivated this
    whole mechanism.

    Cause: the depth guard defaulted to 4, and the heaviest directory in a real image,
    lib/python3.12/site-packages/nvidia, sits at exactly depth 4 under /usr/local. The
    guard fired before the size threshold could, so the packer returned 5 GiB as one
    indivisible unit. Depth must never be the binding constraint at a realistic layout
    depth - size is the control, depth is only a runaway guard.
    """
    deep = tmp_path / "lib" / "python3.12" / "site-packages" / "nvidia"
    _mktree(tmp_path, {
        f"lib/python3.12/site-packages/nvidia/{lib}/payload.bin": 3000
        for lib in ("cudnn", "cublas", "cusolver", "cusparse")
    })
    units = plan_blobs.collect_units(str(tmp_path), max_unit_bytes=4000)
    paths = [p for p, _ in units]
    assert str(deep) not in paths, "nvidia/ came back whole - the depth guard fired again"
    assert len(units) == 4, f"expected one unit per child library, got {paths}"


def test_packing_is_deterministic(tree):
    """Same tree in, same partition out - otherwise every rebuild busts layer caches."""
    units = plan_blobs.collect_units(str(tree), max_unit_bytes=2000)
    first = plan_blobs.pack(units, 3)
    second = plan_blobs.pack(list(reversed(units)), 3)
    assert first == second, "packing must not depend on input ordering"


def test_packing_preserves_every_unit(tree):
    """The packer may rearrange but never lose. A dropped unit is a missing file in the
    shipped image."""
    units = plan_blobs.collect_units(str(tree), max_unit_bytes=2000)
    bins = plan_blobs.pack(units, 4)
    flattened = sorted(u for b in bins for u in b)
    assert flattened == sorted(units)


def test_packing_balances(tree):
    """Loose balance check - the point is that no single bin dominates, which is what
    would reproduce the one-huge-blob failure this whole mechanism exists to prevent."""
    units = plan_blobs.collect_units(str(tree), max_unit_bytes=2000)
    bins = plan_blobs.pack(units, 3)
    totals = [sum(s for _, s in b) for b in bins]
    assert max(totals) <= 2 * (sum(totals) / len(totals))


def test_packing_rejects_a_nonsense_bucket_count(tree):
    units = plan_blobs.collect_units(str(tree), max_unit_bytes=2000)
    with pytest.raises(ValueError, match="buckets must be >= 1"):
        plan_blobs.pack(units, 0)


def test_stage_path_preserves_the_absolute_location():
    """COPY --from=builder /stage/03/ / must put files back exactly where they were, so
    the staged path has to mirror the original absolute path."""
    assert plan_blobs.stage_path("/stage", 3, "/usr/local/bin") == "/stage/03/usr/local/bin"


def test_dry_run_moves_nothing(tree, capsys):
    units = plan_blobs.collect_units(str(tree), max_unit_bytes=2000)
    bins = plan_blobs.pack(units, 2)
    plan_blobs.apply_plan(bins, str(tree / "stage"), dry_run=True)
    assert (tree / "big" / "a.bin").exists()
    assert not (tree / "stage").exists()


def test_apply_plan_moves_every_unit_and_leaves_nothing_behind(tmp_path):
    """End to end: after staging, the source paths are gone and every byte is reachable
    under exactly one stage dir."""
    src = tmp_path / "src"
    _mktree(src, {"big/a.bin": 4000, "big/b.bin": 4000, "small/c.bin": 100})
    units = plan_blobs.collect_units(str(src), max_unit_bytes=2000)
    before = plan_blobs.tree_size(str(src))

    stage = tmp_path / "stage"
    plan_blobs.apply_plan(plan_blobs.pack(units, 2), str(stage), dry_run=False)

    staged_total = sum(plan_blobs.tree_size(str(p)) for p in stage.iterdir())
    assert staged_total == before
    assert not (src / "big" / "a.bin").exists()


def test_symlink_is_not_followed(tmp_path):
    """A symlink counts as itself, not its target - otherwise a link farm double-counts
    real bytes and skews the packing."""
    _mktree(tmp_path, {"real.bin": 5000})
    os.symlink(tmp_path / "real.bin", tmp_path / "link.bin")
    assert plan_blobs.tree_size(str(tmp_path)) < 6000


def test_main_dry_run_reports_and_exits_clean(tree, capsys):
    rc = plan_blobs.main(["--root", str(tree), "--buckets", "3", "--dry-run"])
    assert rc == 0
    assert "into 3 buckets" in capsys.readouterr().out


def test_main_rejects_a_missing_root(tmp_path, capsys):
    rc = plan_blobs.main(["--root", str(tmp_path / "nope"), "--buckets", "2"])
    assert rc == 1
    assert "not a directory" in capsys.readouterr().err


def test_main_requires_buckets_unless_verifying(tmp_path, capsys):
    rc = plan_blobs.main(["--root", str(tmp_path)])
    assert rc == 1
    assert "--buckets is required" in capsys.readouterr().err


# --- the reassembly contract -------------------------------------------------
# The Dockerfile has one COPY per bucket. Nothing structural forces those two numbers to
# agree, and if they diverge the extra bucket's files are simply missing from the shipped
# image - a silent failure that would surface as a mystery ImportError on a rented GPU.
# These tests pin the check that makes it loud instead.

def test_manifest_roundtrips(tmp_path):
    plan_blobs.write_manifest(str(tmp_path), total=12345, buckets=9)
    got = plan_blobs.read_manifest(str(tmp_path / plan_blobs.MANIFEST_NAME))
    assert got == {"total_bytes": 12345, "buckets": 9}


def test_verify_passes_when_the_tree_matches(tmp_path):
    src = tmp_path / "src"
    _mktree(src, {"a.bin": 500, "sub/b.bin": 300})
    plan_blobs.write_manifest(str(tmp_path), plan_blobs.tree_size(str(src)), 3)
    ok, msg = plan_blobs.verify(str(src), str(tmp_path / plan_blobs.MANIFEST_NAME))
    assert ok, msg
    assert "verified" in msg


def test_verify_fails_and_names_the_cause_when_bytes_are_missing(tmp_path):
    """Simulates a bucket that never got a COPY."""
    src = tmp_path / "src"
    _mktree(src, {"a.bin": 500})
    plan_blobs.write_manifest(str(tmp_path), 900, 3)  # builder staged more than arrived
    ok, msg = plan_blobs.verify(str(src), str(tmp_path / plan_blobs.MANIFEST_NAME))
    assert not ok
    assert "BLOB REASSEMBLY FAILED" in msg
    assert "missing" in msg
    # Must point at the actual remedy, per the 2 AM rule.
    assert "--buckets" in msg and "COPY" in msg


def test_verify_reports_extra_bytes_distinctly(tmp_path):
    src = tmp_path / "src"
    _mktree(src, {"a.bin": 900})
    plan_blobs.write_manifest(str(tmp_path), 500, 3)
    ok, msg = plan_blobs.verify(str(src), str(tmp_path / plan_blobs.MANIFEST_NAME))
    assert not ok
    assert "extra" in msg


def test_staging_then_verifying_round_trips(tmp_path):
    """End to end, the way the Dockerfile uses it: partition a tree, reassemble every
    bucket, and confirm the manifest check accepts the result."""
    src = tmp_path / "usr-local"
    _mktree(src, {"big/a.bin": 4000, "big/b.bin": 4000, "bin/tool": 120, "lib/c.bin": 700})
    stage = tmp_path / "stage"
    assert plan_blobs.main(["--root", str(src), "--buckets", "3",
                            "--stage", str(stage), "--max-unit-mib", "0"]) == 0

    # Reassemble exactly as the runtime stage's COPYs do: each bucket is copied onto /.
    dest = tmp_path / "reassembled"
    for bucket in sorted(p for p in stage.iterdir() if p.is_dir()):
        shutil.copytree(bucket, dest, dirs_exist_ok=True)

    # stage_path strips the leading slash, so the tree reappears at dest/<abs path minus />
    rebuilt = dest / str(src).lstrip("/")
    ok, msg = plan_blobs.verify(str(rebuilt), str(stage / plan_blobs.MANIFEST_NAME))
    assert ok, msg


# --- the enforced acceptance criterion --------------------------------------
# Reporting the largest blob is not enough. A build that quietly emits a 5 GiB layer looks
# identical to one that worked, which is exactly how the first build of this image passed
# every gate while defeating its own purpose.

def test_oversized_bucket_fails_the_build(tmp_path, capsys):
    src = tmp_path / "src"
    _mktree(src, {"huge.bin": 8 * 1024 * 1024})
    rc = plan_blobs.main(["--root", str(src), "--buckets", "2",
                          "--stage", str(tmp_path / "stage"), "--dry-run",
                          "--max-bucket-mib", "1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOB TOO LARGE" in err
    # Must name the remedies, not just complain.
    assert "--max-unit-mib" in err and "--buckets" in err


def test_bucket_within_the_limit_passes(tmp_path, capsys):
    src = tmp_path / "src"
    _mktree(src, {f"f{i}.bin": 200 * 1024 for i in range(8)})
    rc = plan_blobs.main(["--root", str(src), "--buckets", "4",
                          "--stage", str(tmp_path / "stage"), "--dry-run",
                          "--max-bucket-mib", "1"])
    assert rc == 0
    assert "largest bucket" in capsys.readouterr().out


def test_split_threshold_is_derived_from_the_tree_when_not_given(tmp_path, capsys):
    """A hardcoded threshold silently stops splitting enough as soon as a dependency
    doubles in size. Deriving it from total/buckets keeps it proportionate."""
    src = tmp_path / "src"
    _mktree(src, {f"f{i}.bin": 1024 * 1024 for i in range(16)})  # 16 MiB total
    plan_blobs.main(["--root", str(src), "--buckets", "8", "--dry-run"])
    out = capsys.readouterr().out
    # 16 MiB / 8 buckets / 2 = 1 MiB
    assert "split threshold 1.0MiB" in out


def test_verify_mode_returns_nonzero_from_main(tmp_path, capsys):
    src = tmp_path / "src"
    _mktree(src, {"a.bin": 100})
    plan_blobs.write_manifest(str(tmp_path), 999, 2)
    rc = plan_blobs.main(["--root", str(src), "--verify",
                          str(tmp_path / plan_blobs.MANIFEST_NAME)])
    assert rc == 1
    assert "BLOB REASSEMBLY FAILED" in capsys.readouterr().err
