import dataclasses
import hashlib
import json
from pathlib import Path

from af3_hallucination.audit import audit_run
from af3_hallucination.config import load_config
from af3_hallucination.hashing import canonical_sha256, sha256_file
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
    hallucination_sha256 = canonical_sha256(dataclasses.asdict(resolved.hallucination))
    (root / "checkpoints.json").write_text(
        json.dumps(
            {
                "schema_version": "af3h_checkpoints_v1",
                "hallucination_sha256": hallucination_sha256,
                "input_sha256": "input-hash",
                "checkpoints": [],
            }
        )
    )
    (root / "trajectory.jsonl").write_text("")
    (root / "final_logits.npy").write_bytes(b"not-npy")
    (root / "final_sequence.txt").write_text("AAAA\n")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "config_sha256": resolved.sha256,
                "hallucination_sha256": hallucination_sha256,
                "input_sha256": "input-hash",
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


def test_hallucination_audit_rejects_checkpoint_path_escape(tmp_path):
    root = tmp_path / "hallucination"
    root.mkdir()
    config_path = Path(__file__).resolve().parents[1] / "configs/hallucination/fast20.yaml"
    config = load_config(config_path)
    hallucination_sha256 = canonical_sha256(dataclasses.asdict(config.hallucination))
    (root / "resolved_config.json").write_text(json.dumps(config.to_dict()))
    (root / "schedule.json").write_text(
        json.dumps(
            {
                "provenance": SCHEDULE_PROVENANCE,
                "steps": expand_schedule(config.hallucination),
            }
        )
    )
    escaped = tmp_path / "outside.npy"
    escaped.write_bytes(b"outside")
    (root / "checkpoints.json").write_text(
        json.dumps(
            {
                "schema_version": "af3h_checkpoints_v1",
                "hallucination_sha256": hallucination_sha256,
                "input_sha256": "input-hash",
                "checkpoints": [
                    {
                        "evaluation_forward": 1,
                        "path": "../outside.npy",
                        "sha256": sha256_file(escaped),
                    }
                ],
            }
        )
    )
    (root / "trajectory.jsonl").write_text("{}\n")
    (root / "final_logits.npy").write_bytes(b"logits")
    (root / "final_sequence.txt").write_text("AAAA\n")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "af3h_hallucination_summary_v1",
                "status": "completed",
                "config_sha256": config.sha256,
                "hallucination_sha256": hallucination_sha256,
                "input_sha256": "input-hash",
                "completed_steps": 1,
                "checkpoint_count": 1,
                "final_logits_sha256": sha256_file(root / "final_logits.npy"),
                "final_sequence_sha256": hashlib.sha256(b"AAAA").hexdigest(),
            }
        )
    )

    result = audit_run(root)
    assert result["status"] == "fail"
    assert any("escapes the run directory" in error for error in result["errors"])


def test_workflow_audit_rejects_nonterminal_state(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/antibody/mock.yaml")
    output = tmp_path / "run"
    AntibodyWorkflow(config).run(output)
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "running"
    state_path.write_text(json.dumps(state))

    result = audit_run(output)
    assert result["status"] == "fail"
    assert "workflow run is not in a terminal state" in result["errors"]
