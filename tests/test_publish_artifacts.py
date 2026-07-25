from assay import publish_artifacts


def test_uploads_dir_to_run_scoped_path(tmp_path):
    (tmp_path / "heartbeat.log").write_text("hi\n")
    (tmp_path / "stdout.log").write_text("ran\n")
    calls = {}
    class _Api:
        def create_repo(self, repo_id, repo_type, exist_ok, private, token):
            calls["repo"] = (repo_id, repo_type, private)
        def upload_folder(self, folder_path, repo_id, repo_type, path_in_repo, token):
            calls["upload"] = (folder_path, repo_id, repo_type, path_in_repo)
    path = publish_artifacts.upload_artifacts(
        str(tmp_path), "uist-labs/assay-run-artifacts", "tok", run_id="pod123", api=_Api())
    assert calls["repo"] == ("uist-labs/assay-run-artifacts", "dataset", True)
    assert calls["upload"][3] == "pod123"
    assert path == "pod123"


def test_main_skips_when_no_token(tmp_path, monkeypatch, capsys):
    """Test that main() returns 0 when HF_TOKEN is absent (never blocks teardown)."""
    artifacts_dir = str(tmp_path)
    dataset_repo = "org/ds"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = publish_artifacts.main(["publish_artifacts", artifacts_dir, dataset_repo])
    assert result == 0
    captured = capsys.readouterr()
    assert "no HF_TOKEN" in captured.err


def test_main_usage_on_missing_args(capsys):
    """Test that main() returns non-zero with actionable usage on missing args."""
    result = publish_artifacts.main(["publish_artifacts"])
    assert result != 0
    captured = capsys.readouterr()
    assert "usage:" in captured.err.lower()


def test_main_skips_when_dataset_empty(tmp_path, monkeypatch, capsys):
    """An empty dataset_repo (no ASSAY_ARTIFACTS_DATASET, no org default) skips the
    upload cleanly and returns 0 - an external user of the public image must not 403
    against someone else's namespace or block teardown. Checked BEFORE the token so a
    tokened external user with no dataset still skips instead of erroring."""
    monkeypatch.setenv("HF_TOKEN", "tok")  # tokened, but no destination
    result = publish_artifacts.main(["publish_artifacts", str(tmp_path), ""])
    assert result == 0
    assert "no ASSAY_ARTIFACTS_DATASET" in capsys.readouterr().err
