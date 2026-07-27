"""RunPod storage rate table - the ONLY hand-maintained cost input.

RunPod exposes GPU prices through the gpuTypes GraphQL query but NOT storage rates,
so these live in git as code-as-config (the documented escape hatch to the
no-runtime-config rule). Source: docs.runpod.io/pods/pricing, read 2026-07-25.
The reconciler is what proves they have not drifted - if it ever does, bump
RATE_TABLE_VERSION so old records stay recomputable against the rates they were
written with. ASCII only.
"""
from __future__ import annotations

# Bump on ANY rate edit below. Embedded in every cost record.
RATE_TABLE_VERSION = "2026-07-25"

# Charged only while the pod is RUNNING; a stopped pod pays nothing for container disk.
CONTAINER_DISK_GB_MONTH_RUNNING = 0.10

# Pod volume disk. assay uses a NETWORK volume rather than pod volume disk, so
# volume_disk is normally 0 on our runs - carried for correctness, not for our path.
VOLUME_DISK_GB_MONTH_RUNNING = 0.10
VOLUME_DISK_GB_MONTH_STOPPED = 0.20

# Network volume: billed continuously, whether or not any pod exists. This is the
# STANDING cost, and it is allocated host-side at roll-up, never inside the pod.
NETWORK_VOLUME_GB_MONTH_UNDER_1TB = 0.07
NETWORK_VOLUME_GB_MONTH_OVER_1TB = 0.05

# RunPod's network-volume tier boundary, in GB. The cheaper rate applies ABOVE this.
NETWORK_VOLUME_TIER_BOUNDARY_GB = 1024

# 365*24/12. Reaches ONLY the container-disk line (about $0.016 on a 3h run), so the
# 730-vs-720 ambiguity is sub-cent exposure. Standing cost prorates by calendar month
# and never touches this constant. See spec section 8.
HOURS_PER_MONTH = 730.0


def network_volume_rate(size_gb: float) -> float:
    """$/GB/month for a network volume of `size_gb`.

    RunPod drops the rate ABOVE 1TB, so exactly 1024 GB still pays the under-1TB
    rate. Both UIST volumes (100 GB and 250 GB) are under the boundary today.
    """
    if float(size_gb) > NETWORK_VOLUME_TIER_BOUNDARY_GB:
        return NETWORK_VOLUME_GB_MONTH_OVER_1TB
    return NETWORK_VOLUME_GB_MONTH_UNDER_1TB
