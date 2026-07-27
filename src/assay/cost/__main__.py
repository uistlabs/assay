"""`python -m assay.cost begin|finalize` - the entry point pod_entry.sh calls."""
from __future__ import annotations

import sys

from assay.cost.collect import main

raise SystemExit(main(sys.argv[1:]))
