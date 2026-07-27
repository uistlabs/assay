"""The modularity invariant, enforced rather than trusted.

assay.cost is a bolt-on observer. It may import from assay core; core must NEVER
import assay.cost. Without this test the boundary is a convention someone remembers,
and the first time it is broken the cost feature stops being deletable.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "assay"


def _package_for(path: pathlib.Path) -> str:
    """Dotted package that contains this file, e.g. src/assay/job.py -> "assay",
    src/assay/lm_eval_tasks/utils.py -> "assay.lm_eval_tasks". This is the base a
    relative import written in that file resolves against.
    """
    rel_dir = path.resolve().parent.relative_to(SRC.parent)
    return ".".join(rel_dir.parts)


def _imported_names(path: pathlib.Path, source: str | None = None) -> set[str]:
    """Every module name this file imports, in fully-qualified dotted form.

    Relative imports (`from . import cost`, `from .cost import rates`, `from
    ..cost import rates`, etc) are resolved against the importing file's own
    package before being recorded, so they show up "assay.cost"-shaped exactly
    like an absolute `import assay.cost` would. Without that resolution a
    relative import loses its "assay" prefix entirely and slips past the
    containment check unnoticed - and once cost is a sibling subpackage, the
    relative form is the natural thing for a contributor to write.

    `source`, when given, is used in place of reading `path` from disk - this
    lets tests drive the resolver with synthetic source text against a real
    (or hypothetical) path without creating any file.
    """
    text = source if source is not None else path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    package = _package_for(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                # Absolute import: node.module is already fully qualified.
                full_module = node.module or ""
            else:
                # Relative import: level 1 ("from . import x" / "from .x import
                # y") resolves against the file's own package; each additional
                # dot strips one more trailing component of that package before
                # joining node.module (which is None for the bare "from . import
                # x" form - the base itself is the imported name in that case).
                parts = package.split(".")
                keep = max(0, len(parts) - (node.level - 1))
                base = ".".join(parts[:keep])
                full_module = f"{base}.{node.module}" if node.module else base
            names.add(full_module)
            names.update(f"{full_module}.{alias.name}" for alias in node.names)
    return names


def _core_files():
    """Every source file in assay core - recursively - excluding the cost
    subtree itself (cost importing cost is fine and expected) and any bytecode
    cache directory.
    """
    for path in sorted(SRC.rglob("*.py")):
        rel_parts = path.relative_to(SRC).parts
        if rel_parts[0] == "cost":
            continue
        if "__pycache__" in rel_parts:
            continue
        yield path


def test_core_modules_never_import_cost():
    offenders = {}
    for path in _core_files():
        bad = sorted(n for n in _imported_names(path)
                     if n == "assay.cost" or n.startswith("assay.cost."))
        if bad:
            offenders[path.name] = bad
    assert offenders == {}, (
        "assay core must never import assay.cost - cost is a bolt-on observer and "
        f"has to stay deletable. Offending imports: {offenders}")


def test_the_boundary_test_is_actually_scanning_files():
    # A glob that silently matches nothing (or only the top level) would make the
    # test above vacuously pass forever. Pin that it sees the real core modules,
    # including ones in a subpackage.
    scanned = {p.relative_to(SRC).as_posix() for p in _core_files()}
    assert "job.py" in scanned
    assert "config.py" in scanned
    assert len(scanned) >= 10
    assert "lm_eval_tasks/utils.py" in scanned


def test_relative_from_import_resolves_to_fully_qualified_name():
    # `from .cost import rates` written inside a top-level core module (package
    # "assay") must resolve to "assay.cost" / "assay.cost.rates", not the bare
    # "cost" / "cost.rates" the unqualified module name alone would suggest.
    names = _imported_names(SRC / "job.py", source="from .cost import rates\n")
    assert "assay.cost" in names
    assert "assay.cost.rates" in names


def test_relative_bare_import_resolves_to_fully_qualified_name():
    # `from . import cost` is the other natural way to reach a sibling subpackage.
    names = _imported_names(SRC / "job.py", source="from . import cost\n")
    assert "assay.cost" in names


def test_relative_import_from_a_subpackage_resolves_with_correct_prefix():
    # From inside assay.lm_eval_tasks, ".." steps back up to "assay" before
    # picking up "cost" - a naive dot-count-only strip would under- or over-shoot.
    names = _imported_names(
        SRC / "lm_eval_tasks" / "utils.py", source="from ..cost import rates\n")
    assert "assay.cost" in names
    assert "assay.cost.rates" in names


def test_absolute_import_of_cost_still_resolves():
    # Level 0 (absolute) imports must keep working exactly as they did before.
    names = _imported_names(SRC / "job.py", source="import assay.cost\n")
    assert "assay.cost" in names
