from assay.cost.rates import (
    HOURS_PER_MONTH,
    NETWORK_VOLUME_GB_MONTH_OVER_1TB,
    NETWORK_VOLUME_GB_MONTH_UNDER_1TB,
    NETWORK_VOLUME_TIER_BOUNDARY_GB,
    RATE_TABLE_VERSION,
    network_volume_rate,
)


def test_rate_table_is_versioned():
    # The record embeds this so a recomputation years later is deterministic even
    # after RunPod changes prices. A rate change MUST bump it.
    assert RATE_TABLE_VERSION == "2026-07-25"


def test_network_volume_rate_below_and_at_the_tier_boundary():
    # RunPod's cheaper tier applies ABOVE 1TB, so exactly 1024 GB is still the
    # under-1TB rate. Our own volumes (100 GB, 250 GB) both land here.
    assert network_volume_rate(100) == NETWORK_VOLUME_GB_MONTH_UNDER_1TB
    assert network_volume_rate(250) == NETWORK_VOLUME_GB_MONTH_UNDER_1TB
    assert network_volume_rate(NETWORK_VOLUME_TIER_BOUNDARY_GB) == (
        NETWORK_VOLUME_GB_MONTH_UNDER_1TB)


def test_network_volume_rate_above_the_tier_boundary():
    assert network_volume_rate(NETWORK_VOLUME_TIER_BOUNDARY_GB + 1) == (
        NETWORK_VOLUME_GB_MONTH_OVER_1TB)
    assert network_volume_rate(4096) == NETWORK_VOLUME_GB_MONTH_OVER_1TB


def test_rate_math_matches_the_spec_worked_examples():
    # Worked examples from spec F5, pinned so a rate edit cannot silently restate
    # the economics every quote is built on. Pure rate math, not an inventory claim:
    # 100 GB at the under-1TB rate = $7.00/month; 250 GB at the same rate = $17.50/month.
    assert round(100 * network_volume_rate(100), 2) == 7.00
    assert round(250 * network_volume_rate(250), 2) == 17.50


def test_hours_per_month_is_the_365_day_average():
    assert HOURS_PER_MONTH == 730.0
