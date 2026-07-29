"""Shared pytest fixtures across the test suite.

F-009 T6: ECC-policy manifest variants for apply_ecc_policy's four spec cases,
built via dataclasses.replace from the Task 1 canonical shape
(tests/test_manifest.py::_sample_manifest) - one shared shape, four verdicts,
never four separate ManifestV1(...) constructions.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from assay.manifest import EccWindow
from tests.test_manifest import _sample_manifest


def _ecc_manifest(verdict, *, ecc_supported=True, ecc_enabled=True,
                  counters_begin=(0, 0), counters_end=(0, 0),
                  uncorrected_delta=0, corrected_delta=0):
    """_sample_manifest() with ECC-capable hardware and a given ecc_window
    verdict - the one construction every ECC-policy fixture replaces onto."""
    base = _sample_manifest()
    m = replace(base, hardware=replace(base.hardware,
                                       ecc_supported=ecc_supported, ecc_enabled=ecc_enabled))
    return replace(m, ecc_window=EccWindow(
        counters_begin=counters_begin, counters_end=counters_end,
        uncorrected_delta=uncorrected_delta, corrected_delta=corrected_delta,
        verdict=verdict))


@pytest.fixture
def manifest_void():
    """ECC-capable hardware, verdict=void: an uncorrected error landed inside
    the measurement window."""
    return _ecc_manifest("void", counters_begin=(0, 0), counters_end=(1, 0),
                         uncorrected_delta=1, corrected_delta=0)


@pytest.fixture
def manifest_ecc_not_captured():
    """ECC-capable hardware whose end-of-window counter read failed or reset
    (spec D9/D3): silence on ECC-capable hardware is treated as void, never
    clean."""
    return _ecc_manifest("not-captured", counters_begin=(0, 0), counters_end=None,
                         uncorrected_delta=None, corrected_delta=None)


@pytest.fixture
def manifest_corrected_only():
    """ECC-capable hardware, verdict=clean but corrected errors occurred in
    the window - disclosed (card/manifest), never a gate failure."""
    return _ecc_manifest("clean", counters_begin=(0, 0), counters_end=(0, 3),
                         uncorrected_delta=0, corrected_delta=3)


@pytest.fixture
def manifest_no_ecc():
    """Hardware with no ECC support at all - verdict=not-applicable, the
    Task 1 canonical shape unchanged."""
    return _sample_manifest()
