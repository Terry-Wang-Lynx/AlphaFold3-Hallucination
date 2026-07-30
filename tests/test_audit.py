import hashlib
import json
from pathlib import Path

from af3_hallucination.audit import audit_run
from af3_hallucination.config import load_config
from af3_hallucination.schedule import SCHEDULE_PROVENANCE, expand_schedule
from af3_hallucination.workflow import AntibodyWorkflow


def test_workflow_audit_detects_tampering(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """schema_version: 1
kind: antibody
hallucination:
  backend: mock
  stages: [{type: logits, steps: 1}]
  losses: []
antibody:
  cdr_only: true
  framework_fixed: true
  antigen_fixed: true
  hallucination: {plugin: mock, config: {artifact_name: h}}
  diffusion: {plugin: mock, config: {artifact_name: d}}
  inverse_folding: {plugin: mock, config: {artifact_name: i}}
  consistency: {plugin: mock, config: {artifact_name: c}}
  final_evaluation: {plugin: mock, config: {artifact_name: f}}
"""
    )
    output = tmp_path / "run"
    AntibodyWorkflow(load_config(config_path)).run(output)
    assert audit_run(output)["status"] == "pass"
    artifact = next(output.glob("00_hallucination/*.json"))
    artifact.write_text("tampered\n")
    result = audit_run(output)
    assert result["status"] == "fail"
    assert any("SHA-256 changed" in error for error in result["errors"])


def test_hallucination_audit_detects_config_and_sequence_tampering(tmp_path):
    root = tmp_path / "hallucination"
    config = Path(__file__).resolve().parents[1] / "configs/hallucination/fast20.yaml"
    resolved = load_config(config)
    (root / "checkpoints").mkdir(parents=True)
    (root / "resolved_config.json").write_text(json.dumps(resolved.to_dict()))
    (root / "schedule.json").write_text(
        json.dumps(
            {
                "provenance": SCHEDULE_PROVENANCE,
                "steps": expand_schedule(resolved.hallucination),
            }
        )
    )
    (root / "checkpoints.json").write_text('{"checkpoints": []}')
    (root / "trajectory.jsonl").write_text("")
    (root / "final_logits.npy").write_bytes(b"not-npy")
    (root / "final_sequence.txt").write_text("AAAA\n")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "config_sha256": resolved.sha256,
                "completed_steps": 0,
                "checkpoint_count": 0,
                "final_logits_sha256": "wrong",
                "final_sequence_sha256": hashlib.sha256(b"BBBB").hexdigest(),
            }
        )
    )

    modified = resolved.to_dict()
    modified["hallucination"]["seed"] = 99
    (root / "resolved_config.json").write_text(json.dumps(modified))
    result = audit_run(root)
    assert result["status"] == "fail"
    assert any("config hash" in error for error in result["errors"])
    assert any("final sequence hash" in error for error in result["errors"])
