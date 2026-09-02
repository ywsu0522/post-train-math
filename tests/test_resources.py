import json
from pathlib import Path

from postrain_math_lab import resources


def test_download_model_records_resolved_commit(tmp_path, monkeypatch) -> None:
    class FakeInfo:
        sha = "abc123"

    class FakeApi:
        def model_info(self, repo_id: str, revision: str):
            assert repo_id == "org/model"
            assert revision == "main"
            return FakeInfo()

    def fake_snapshot_download(*, repo_id, revision, local_dir):
        assert repo_id == "org/model"
        assert revision == "abc123"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(resources, "HfApi", FakeApi)
    monkeypatch.setattr(resources, "snapshot_download", fake_snapshot_download)

    target = tmp_path / "models" / "model"
    resources.download_model(
        repo_id="org/model",
        output_dir=target,
    )

    manifest = json.loads((target / "source.json").read_text(encoding="utf-8"))
    assert manifest["resolved_commit"] == "abc123"
    assert manifest["local_dir"] == str(target)


def test_download_model_reuses_complete_local_copy(tmp_path, monkeypatch) -> None:
    target = tmp_path / "models" / "model"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "source.json").write_text("{}", encoding="utf-8")

    class FailApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("network lookup should not run")

    monkeypatch.setattr(resources, "HfApi", FailApi)

    result = resources.download_model(
        repo_id="org/model",
        output_dir=target,
    )

    assert result == target
