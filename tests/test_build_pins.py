"""F-006/F-007/F-011 - the pinned coherent set has ONE source of truth.

deploy/constraints.txt pins the exact versions of every dependency the certification
relies on (the NVFP4 writer above all). The Dockerfile resolves against it in the SAME
pip resolve as `pip install .` - post-hoc `pip install pkg==X` force-pins are banned,
because they can silently down-force a transitive of something already installed into
an incoherent pair while every assert still passes (the F-007 trap). The build gate
then asserts the installed set matches the file by READING it, so a new pin cannot be
added without the gate covering it (no hand-maintained parallel list - the F-026
lesson). These are string-contract tests, same style as the pod_entry marker test:
they keep the three surfaces (constraints file, Dockerfile, dev env) from drifting.
"""
import pathlib
import re

from importlib.metadata import version as _installed

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONSTRAINTS = ROOT / "deploy" / "constraints.txt"
DOCKERFILE = ROOT / "deploy" / "Dockerfile"
PYPROJECT = ROOT / "pyproject.toml"


def _pins() -> dict[str, str]:
    pins = {}
    for raw in CONSTRAINTS.read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        name, want = line.split("==")
        pins[name.strip()] = want.strip()
    return pins


def _image_only() -> set[str]:
    """Pins marked `# image-only` (F-030): members of the coherent set the dev box
    structurally cannot install (cu129 GPU builds). The build gate asserts them
    like every other pin; only the dev-env test skips them."""
    names = set()
    for raw in CONSTRAINTS.read_text().splitlines():
        line = raw.split("#")[0].strip()
        if line and "# image-only" in raw:
            names.add(line.split("==")[0].strip())
    return names


def test_constraints_pin_the_certification_stack():
    pins = _pins()
    # The NVFP4 writer (F-006), the on-disk format contract, the model loader, the
    # eval harness, and the gate's statistics library (F-011). Exact ==, no floors.
    for name in ("llmcompressor", "compressed-tensors", "transformers",
                 "lm-eval", "scipy"):
        assert name in pins, f"{name} missing from deploy/constraints.txt"
        assert re.fullmatch(r"[0-9][0-9A-Za-z.]*", pins[name]), (name, pins[name])


def test_dockerfile_resolves_against_constraints_in_one_pass():
    text = DOCKERFILE.read_text()
    install = next(l for l in text.splitlines()
                   if "pip install ." in l and l.strip().startswith("RUN"))
    assert "constraints.txt" in install, (
        "pip install . must pass -c deploy/constraints.txt so the coherent set is "
        "resolved in ONE pass, not force-pinned afterwards")


def test_no_posthoc_force_pins_remain():
    """A `RUN pip install pkg==X` AFTER `pip install .` is the F-007 trap: it can
    down-force a transitive silently. Every exact pin lives in constraints.txt; the
    only allowed exact-version installs are index-selection lines (torch/vllm wheels
    from non-PyPI sources) and lm-eval's extra, which must ALSO be constrained."""
    text = DOCKERFILE.read_text()
    assert 'pip install "compressed-tensors==' not in text
    assert 'pip install "transformers==' not in text


def test_lmeval_extra_install_matches_constraints():
    # lm-eval needs its own install line for the [math] extra; the version it names
    # must be the constrained one (two-sided contract, not a second opinion).
    text = DOCKERFILE.read_text()
    m = re.search(r'lm-eval\[math\]==([0-9A-Za-z.]+)', text)
    assert m, "Dockerfile must install lm-eval[math] (minerva hard-imports math_verify)"
    assert m.group(1) == _pins()["lm-eval"]
    line = next(l for l in text.splitlines() if "lm-eval[math]" in l and "RUN" in l)
    assert "constraints.txt" in line


def test_build_gate_asserts_pins_by_reading_the_file():
    """F-007: the gate PRINTED llmcompressor's version - a check that cannot fail is
    not a check. It must now iterate constraints.txt and assert every installed
    version matches, so adding a pin automatically extends the gate."""
    text = DOCKERFILE.read_text()
    gate = text[text.index("import torch, llmcompressor"):]
    assert "constraints.txt" in gate[:2000]
    assert "drifted off the pinned coherent set" in gate[:2000]


def test_build_gate_imports_the_gates_statistics_symbol():
    # F-011: BITE 2 puts the certification decision on scipy.stats.sem. Import the
    # exact symbol, same philosophy as the vllm.LLM import above it.
    assert "from scipy.stats import sem" in DOCKERFILE.read_text()


def test_scipy_declared_in_pyproject():
    # F-011: the gate's statistics dependency must be declared, not inherited as a
    # transitive of the eval stack that could vanish on a dep bump.
    deps = PYPROJECT.read_text()
    assert re.search(r'"scipy>=', deps), "scipy must be a declared dependency"


def test_pyproject_llmcompressor_floor_matches_reality():
    # F-006: ">=0.6.0" was a floor satisfied by 0.12 only by luck. The floor must
    # name the tested major.minor so a fresh dev env cannot resolve something older
    # than what certification runs on.
    m = re.search(r'"llmcompressor>=([0-9.]+)"', PYPROJECT.read_text())
    assert m, "llmcompressor must stay a declared dependency"
    pinned = _pins()["llmcompressor"]
    assert pinned.startswith(m.group(1).rstrip(".")[:4]) or m.group(1) == pinned, (
        f"pyproject floor {m.group(1)} does not track the pinned {pinned}")


def test_dev_environment_matches_the_pinned_set():
    """The dev box runs the same certification code paths in tests; a version drift
    between the local env and the image invalidates 'green locally' as evidence.
    This is the one check here that exercises reality, not strings. Image-only pins
    (the cu129 GPU stack, F-030) are excluded: the dev box cannot install them, and
    no local test exercises them - the build gate is their reality check."""
    image_only = _image_only()
    for name, want in _pins().items():
        if name in image_only:
            continue
        assert _installed(name) == want, (
            f"local {name} is {_installed(name)}, constraints pin {want} - "
            "bump uv env and constraints together, as a unit")


def test_gpu_stack_pinned_image_only_and_dockerfile_matches():
    """F-030: torch/vLLM install outside the constraints resolve (cu129 index /
    GitHub wheel), so the gate's read-the-file loop is what holds their exact
    versions - they must BE in the file, and the Dockerfile's install lines must
    name the same versions (two-sided contract, same shape as the lm-eval extra)."""
    pins, image_only, text = _pins(), _image_only(), DOCKERFILE.read_text()
    for name in ("torch", "vllm"):
        assert name in pins, f"{name} missing from deploy/constraints.txt (F-030)"
        assert name in image_only, (
            f"{name} must be marked `# image-only`: the dev box cannot install "
            "cu129 builds, so the dev-env test would fail forever")
        assert pins[name].endswith("+cu129"), (name, pins[name])
    m = re.search(r"pip install torch==([0-9][0-9.]*) --index-url \S*/cu129", text)
    assert m, "Dockerfile must install torch from the cu129 index at an exact version"
    assert pins["torch"] == m.group(1) + "+cu129", (
        f"Dockerfile installs torch {m.group(1)} (cu129), constraints pin "
        f"{pins['torch']} - bump both together, as a unit")
    assert f"vllm-{pins['vllm']}-" in text, (
        f"Dockerfile's vLLM wheel URL does not match the pinned {pins['vllm']} - "
        "bump the wheel asset and the constraint together, as a unit")


def test_dockerignore_ships_the_constraints_file():
    """.dockerignore is an ALLOWLIST (`*` then `!` re-includes): anything not named
    is silently absent from `COPY . /app`. Both pip resolves and the build gate read
    deploy/constraints.txt inside the image, so it must be re-included - without
    this line the build dies at pip install, loudly but pointlessly."""
    rules = (ROOT / ".dockerignore").read_text().splitlines()
    assert "!deploy/constraints.txt" in rules
