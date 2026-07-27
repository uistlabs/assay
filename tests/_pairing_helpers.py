"""Shared builder for per-item paired data with an EXACT chosen paired SE.

Used by the gate and publish tests: base items sit at a constant value and the
quantized side adds a per-item delta pattern alternating +/-d around the mean
delta. For even n, mean(quant items) = quant_val exactly, and the sample-sem of
the deltas is d/sqrt(n-1), so d = paired_se*sqrt(n-1) hits the requested SE
exactly (up to FP ulps, which the gate's round(...,12) absorbs)."""
import math


def paired_items(base_val: float, quant_val: float, paired_se: float,
                 n: int = 4, prefix: str = "doc"):
    d = paired_se * math.sqrt(n - 1) if n > 1 else 0.0
    delta = quant_val - base_val
    base = {f"{prefix}{i}": {"v": base_val, "h": f"h-{prefix}{i}"} for i in range(n)}
    quant = {f"{prefix}{i}": {"v": base_val + delta + (d if i % 2 else -d),
                              "h": f"h-{prefix}{i}"} for i in range(n)}
    return base, quant
