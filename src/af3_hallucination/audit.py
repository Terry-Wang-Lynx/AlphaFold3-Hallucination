"""Independent, import-light integrity audits for completed runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import parse_config
from .hashing import canonical_sha256, sha256_file
from .schedule import SCHEDULE_PROVENANCE, expand_schedule


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _workflow_audit(root: Path) -> dict[str, Any]:
    state = _json(root / "run_state.json")
    errors: list[str] = []
    config_path = root / "resolved_config.json"
    if not config_path.is_file():
        errors.append("resolved_config.json is missing")
    else:
        try:
            config = parse_config(_json(config_path), source_path=config_path)
            if config.sha256 != state.get("config_sha256"):
                errors.append("resolved config hash differs from run_state.json")
        except Exception as exc:  # noqa: BLE001 - audit must report all corruption
            errors.append(f"resolved config is invalid: {exc}")
    checked = 0
    for name, record in state.get("artifacts", {}).items():
        path = Path(record.get("path", ""))
        if not path.is_file():
            errors.append(f"artifact {name} is missing: {path}")
            continue
        checked += 1
        if path.stat().st_size != int(record.get("bytes", -1)):
            errors.append(f"artifact {name} byte count changed")
        if sha256_file(path) != record.get("sha256"):
            errors.append(f"artifact {name} SHA-256 changed")
    return {
        "schema_version": "af3h_audit_v1",
        "kind": "antibody_workflow",
        "status": "pass" if not errors else "fail",
        "run_status": state.get("status"),
        "checked_artifacts": checked,
        "errors": errors,
    }


def _hallucination_audit(root: Path) -> dict[str, Any]:
    summary = _json(root / "summary.json")
    checkpoints = _json(root / "checkpoints.json")
    errors: list[str] = []
    config_path = root / "resolved_config.json"
    config = None
    if not config_path.is_file():
        errors.append("resolved_config.json is missing")
    else:
        try:
            config = parse_config(_json(config_path), source_path=config_path)
            if config.sha256 != summary.get("config_sha256"):
                errors.append("resolved config hash differs from summary.json")
        except Exception as exc:  # noqa: BLE001 - audit must report all corruption
            errors.append(f"resolved config is invalid: {exc}")
    schedule_path = root / "schedule.json"
    if not schedule_path.is_file():
        errors.append("schedule.json is missing")
    elif config is not None:
        expected = {
            "provenance": SCHEDULE_PROVENANCE,
            "steps": expand_schedule(config.hallucination),
        }
        try:
            if canonical_sha256(_json(schedule_path)) != canonical_sha256(expected):
                errors.append("schedule differs from resolved configuration")
        except Exception as exc:  # noqa: BLE001 - audit must report all corruption
            errors.append(f"schedule is invalid: {exc}")
    records = checkpoints.get("checkpoints", [])
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"checkpoint is missing: {record['path']}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"checkpoint hash changed: {record['path']}")
    final_logits = root / "final_logits.npy"
    if not final_logits.is_file():
        errors.append("final_logits.npy is missing")
    elif sha256_file(final_logits) != summary.get("final_logits_sha256"):
        errors.append("final logits hash changed")
    final_sequence = root / "final_sequence.txt"
    if not final_sequence.is_file():
        errors.append("final_sequence.txt is missing")
    else:
        import hashlib

        sequence = final_sequence.read_text().strip()
        if hashlib.sha256(sequence.encode()).hexdigest() != summary.get(
            "final_sequence_sha256"
        ):
            errors.append("final sequence hash changed")
    trajectory = root / "trajectory.jsonl"
    if not trajectory.is_file():
        errors.append("trajectory.jsonl is missing")
        trajectory_count = 0
    else:
        trajectory_count = sum(bool(line.strip()) for line in trajectory.read_text().splitlines())
        if trajectory_count != int(summary.get("completed_steps", -1)):
            errors.append("trajectory row count differs from completed_steps")
    if len(records) != int(summary.get("checkpoint_count", -1)):
        errors.append("checkpoint record count differs from checkpoint_count")
    return {
        "schema_version": "af3h_audit_v1",
        "kind": "hallucination",
        "status": "pass" if not errors else "fail",
        "run_status": summary.get("status"),
        "checked_checkpoints": len(records),
        "trajectory_rows": trajectory_count,
        "errors": errors,
    }


def audit_run(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if (root / "run_state.json").is_file():
        return _workflow_audit(root)
    if (root / "summary.json").is_file() and (root / "checkpoints.json").is_file():
        return _hallucination_audit(root)
    raise FileNotFoundError(f"no recognized af3h run found in {root}")
