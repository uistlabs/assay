#!/usr/bin/env python3
"""Emit (or drift-check) a recipe's base-model identity pins - F-015 amendment 4.

The pins (base_revision + base_files sha256 map) anchor weights verification to the
reviewed GIT recipe. That moves the authoring trust boundary rather than closing it,
and moving it is only a win if the INDEPENDENT source is the easy path - so this tool
fetches the truth from the Hub at a pinned commit and prints a paste-ready snippet:

    uv run python scripts/pin_base_files.py r1_distill_qwen_7b          # by slug
    uv run python scripts/pin_base_files.py org/Some-Model              # by repo id
    uv run python scripts/pin_base_files.py qwen2_5_7b_instruct --check # warn-only drift

Runs on the DEV BOX where network is fine; the in-pod verifier (assay.verify) is
offline by design. Shard sha256s come free from the Hub's LFS metadata (no shard
download); only the small identity files are fetched and hashed locally. --check is
WARN-ONLY (exit 0 either way): it is the launch-side affordance, same posture as the
cost pre-flight - the hard gate is the in-pod verifier, not this convenience.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

# The identity files of the reviewed F-015 design, extended by F-031: config.json
# (architecture), the safetensors index (which shards the loader reads), the FULL
# tokenizer surface - tokenizer.json / vocab.json / merges.txt hold the actual
# vocab and merges the loader reads; tokenizer_config.json alone left a swapped
# tokenizer invisible to both verify passes - and generation_config.json (default
# sampling). pick_pin_files intersects with the repo's real file list, so members
# a repo lacks simply pin nothing. Shards are matched by suffix. Top-level only:
# nested safetensors (e.g. original/consolidated.*) are never read by the
# transformers loader, and pinning them would buy in-pod hashing minutes for files
# the run does not touch.
_SMALL_PIN_FILES = (
    "config.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json",
)


def pick_pin_files(filenames: list[str]) -> list[str]:
    """The pinned subset of a repo's file list, sorted for a stable rendered diff."""
    return sorted(
        n for n in filenames
        if n in _SMALL_PIN_FILES or (n.endswith(".safetensors") and "/" not in n)
    )


def render_pins(revision: str, files: dict[str, str]) -> str:
    """Paste-ready recipe fields: revision first, files sorted."""
    lines = [f'base_revision="{revision}",', "base_files={"]
    for name in sorted(files):
        lines.append(f'    "{name}": "{files[name]}",')
    lines.append("},")
    return "\n".join(lines)


def diff_pins(recipe_revision: str, recipe_files: dict[str, str],
              hub_revision: str, hub_files: dict[str, str]) -> list[str]:
    """Human-readable drift between a recipe's pins and the hub's current state.
    Empty list == no drift."""
    lines = []
    if recipe_revision != hub_revision:
        lines.append(f"revision drift: recipe pins {recipe_revision}, "
                     f"hub HEAD is now {hub_revision}")
    for name in sorted(set(recipe_files) | set(hub_files)):
        pinned, live = recipe_files.get(name), hub_files.get(name)
        if pinned == live:
            continue
        if live is None:
            lines.append(f"{name}: pinned in the recipe but no longer on the hub")
        elif pinned is None:
            lines.append(f"{name}: new on the hub, not pinned in the recipe")
        else:
            lines.append(f"{name}: sha256 drift (recipe {pinned[:12]}.., "
                         f"hub {live[:12]}..)")
    return lines


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_pins(repo_id: str, revision: str | None = None):  # pragma: no cover - network
    """(commit_sha, {file: sha256}) for a repo at a ref (default: HEAD of main).

    Shards (LFS) carry sha256 in tree metadata - free, no download. The small
    identity files are plain git blobs whose metadata oid is a git sha1, NOT a
    content sha256, so those are downloaded (KBs) and hashed locally."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    commit = info.sha
    by_name = {s.rfilename: s for s in info.siblings}
    files: dict[str, str] = {}
    for name in pick_pin_files(list(by_name)):
        lfs = by_name[name].lfs
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha is None and isinstance(lfs, dict):
            sha = lfs.get("sha256")
        if sha is None:
            sha = _sha256_file(hf_hub_download(repo_id, name, revision=commit))
        files[name] = sha
    return commit, files


def main(argv=None) -> int:  # pragma: no cover - CLI + network wiring
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="recipe slug (assay.recipes) or a raw HF repo id")
    parser.add_argument("--revision", default=None,
                        help="pin at this ref (default: the repo's current HEAD)")
    parser.add_argument("--check", action="store_true",
                        help="warn-only: diff the recipe's existing pins against the hub")
    args = parser.parse_args(argv)

    recipe = None
    if "/" in args.target:
        repo_id = args.target
    else:
        from assay.recipes import get_recipe
        recipe = get_recipe(args.target)
        repo_id = recipe.base_model

    if args.check:
        if recipe is None:
            parser.error("--check needs a recipe slug (it diffs the recipe's pins)")
        if not recipe.base_files:
            print(f"WARN: recipe {recipe.slug!r} has no pins to check - "
                  "generate them first (this tool, without --check)")
            return 0
        drift = diff_pins(recipe.base_revision, dict(recipe.base_files),
                          *fetch_pins(repo_id, args.revision))
        if not drift:
            print(f"pins OK: {recipe.slug} matches {repo_id} ({recipe.base_revision[:12]})")
        for line in drift:
            # Drift is INFORMATION, not an error: the pinned revision stays the
            # certified identity; upstream moving means "re-review before re-pinning".
            print(f"WARN pin drift [{recipe.slug}]: {line}")
        return 0

    commit, files = fetch_pins(repo_id, args.revision)
    print(f"# {repo_id} @ {commit}")
    print("# paste into the recipe in src/assay/recipes.py:")
    print(render_pins(commit, files))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
