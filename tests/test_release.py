import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_candidate_is_accepted_but_not_published():
    manifest = json.loads((ROOT / "release/release_candidate_manifest.json").read_text())
    assert manifest["review_status"] == "accepted_release_candidate"
    assert manifest["published"] is False
    assert all(value == "pass" for value in manifest["platforms"].values())
    assert not any(manifest["artifact_policy"].values())
