import json
from pathlib import Path

from af3_hallucination.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_and_config_cli(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["capabilities"]["core"]
    assert main(["config", "validate", str(ROOT / "configs/hallucination/fast20.yaml")]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "valid"


def test_hallucination_dry_run(tmp_path, capsys):
    assert (
        main(
            [
                "hallucinate",
                "run",
                str(ROOT / "configs/hallucination/fast20.yaml"),
                "--output-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "dry_run"
    assert value["planned_steps"] == 20
    assert (tmp_path / "schedule.json").is_file()


def test_pocket_is_explicit_placeholder(capsys):
    assert main(["pocket", "status"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "not_implemented"
