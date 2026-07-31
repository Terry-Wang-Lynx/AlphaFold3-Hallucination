import json
from pathlib import Path

from af3_hallucination import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_release_candidate_is_github_ready_but_not_published():
    manifest = json.loads((ROOT / "release/release_candidate_manifest.json").read_text())
    assert manifest["schema_version"] == "af3h_release_candidate_v2"
    assert manifest["review_status"] == "github_ready_unpublished"
    assert manifest["github_submission_ready"] is True
    assert manifest["github_submitted"] is False
    assert manifest["published"] is False
    assert manifest["version"] == __version__ == "0.1.0rc2"
    assert f'version = "{__version__}"' in (ROOT / "pyproject.toml").read_text()
    assert all(value == "pass" for value in manifest["platforms"].values())
    assert not any(manifest["artifact_policy"].values())
