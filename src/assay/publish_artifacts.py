"""Self-upload the run's durable artifacts (redacted stdout log, heartbeat, traceback,
delta table, eval JSONs) to a PRIVATE HF dataset - the 'what actually ran' audit trail,
on PASS and FAIL. Retires the reader pod. Called best-effort + time-bounded from
pod_entry.sh: a failed/hung HF push must NEVER block teardown (the volume copy is the
source of truth). ASCII only."""
from __future__ import annotations

import os
import sys


def upload_artifacts(artifacts_dir: str, dataset_repo: str, token: str,
                     run_id: str, api=None) -> str:
    """Upload artifacts_dir to dataset_repo under path-in-repo=run_id. Returns run_id."""
    if api is None:  # pragma: no cover - exercised only against live HF
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True,
                    private=True, token=token)
    api.upload_folder(folder_path=artifacts_dir, repo_id=dataset_repo,
                      repo_type="dataset", path_in_repo=run_id, token=token)
    return run_id


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python -m assay.publish_artifacts <artifacts_dir> <dataset_repo>",
              file=sys.stderr)
        return 2
    artifacts_dir, dataset_repo = argv[1], argv[2]
    if not dataset_repo:
        # No ASSAY_ARTIFACTS_DATASET set: self-upload is opt-in (there is no org
        # default - see pod_entry.sh). Skip cleanly, same as the no-token path: a
        # missing destination is a config choice, never a teardown-blocking failure.
        print("publish_artifacts: no ASSAY_ARTIFACTS_DATASET; skipping upload",
              file=sys.stderr)
        return 0
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("publish_artifacts: no HF_TOKEN; skipping upload", file=sys.stderr)
        return 0  # best-effort: absence of a token is not a teardown-blocking failure
    run_id = os.environ.get("RUNPOD_POD_ID", "run")
    dest = upload_artifacts(artifacts_dir, dataset_repo, token, run_id)
    print(f"publish_artifacts: uploaded {artifacts_dir} -> {dataset_repo}/{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
