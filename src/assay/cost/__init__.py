"""Cost telemetry for assay - a bolt-on observer, never a pipeline stage.

Invoked from scripts/pod_entry.sh at the same layer as log_tee and
publish_artifacts, so the whole feature can be removed by deleting this directory
and two shell lines.

IMPORTANT - THE DEPENDENCY ARROW POINTS ONE WAY. This package may import from assay
core; core must NEVER import from assay.cost. tests/test_cost_boundary.py enforces
it. Deliberately no re-exports here: `python -m assay.cost` must stay light and must
not pull the runpod SDK at import time.
"""
