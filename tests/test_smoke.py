import assay
from assay import smoke


def test_package_imports():
    assert assay.__version__ == "0.5.1"


def test_tier1_structural_passes_on_real_recipes():
    # Resolves every real recipe's assay-owned tasks, runs process_results on a
    # synthetic sample, and crosses a REAL spawn Pipe embedding the path-named
    # filter_fn - the boundary that hid the v0.4.1 crash. Raises on any failure.
    smoke.tier1_structural()
