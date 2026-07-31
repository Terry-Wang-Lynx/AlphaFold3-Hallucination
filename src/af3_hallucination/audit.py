"""Independent, import-light integrity audits for completed runs."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import WORKFLOW_STEPS, parse_config
from .hashing import canonical_sha256, sha256_file
from .schedule import SCHEDULE_PROVENANCE, expand_schedule


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _failed(kind: str, error: str) -> dict[str, Any]:
    return {
        "schema_version": "af3h_audit_v1",
        "kind": kind,
        "status": "fail",
        "run_status": None,
        "errors": [error],
    }


def _contained_path(
    root: Path,
    raw_path: Any,
    *,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"{label} has an invalid path")
        return None
    relative = Path(raw_path)
    if relative.is_absolute():
        errors.append(f"{label} path must be relative to the run directory: {raw_path}")
        return None
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        errors.append(f"{label} path escapes the run directory: {raw_path}")
        return None
    return resolved


def _workflow_audit(root: Path) -> dict[str, Any]:
    try:
        state = _json(root / "run_state.json")
    except Exception as exc:  # noqa: BLE001 - return a structured corruption report
        return _failed("antibody_workflow", f"run_state.json is invalid: {exc}")
    errors: list[str] = []
    if state.get("schema_version") != "af3h_workflow_state_v1":
        errors.append("run_state.json has an unsupported schema_version")
    run_status = state.get("status")
    if run_status not in {"completed", "rejected", "skipped"}:
        errors.append("workflow run is not in a terminal state")
    raw_steps = state.get("steps", {})
    if not isinstance(raw_steps, Mapping):
        errors.append("run_state.json steps must be a mapping")
        raw_steps = {}
    unknown_steps = set(raw_steps) - set(WORKFLOW_STEPS)
    if unknown_steps:
        errors.append(f"run_state.json contains unknown steps: {sorted(unknown_steps)}")
    missing_steps = set(WORKFLOW_STEPS) - set(raw_steps)
    if missing_steps:
        errors.append(f"run_state.json is missing steps: {sorted(missing_steps)}")
    step_statuses = []
    for name in WORKFLOW_STEPS:
        record = raw_steps.get(name)
        if not isinstance(record, Mapping):
            if name in raw_steps:
                errors.append(f"step {name} record must be a mapping")
            step_statuses.append(None)
        else:
            step_statuses.append(record.get("status"))
    if run_status == "completed" and any(status != "completed" for status in step_statuses):
        errors.append("completed workflow contains a non-completed step")
    elif run_status in {"rejected", "skipped"} and all(
        status is not None for status in step_statuses
    ):
        first_terminal = next(
            (index for index, status in enumerate(step_statuses) if status != "completed"),
            None,
        )
        if (
            first_terminal is None
            or step_statuses[first_terminal] != run_status
            or any(status != "skipped" for status in step_statuses[first_terminal + 1 :])
        ):
            errors.append("terminal workflow step statuses are inconsistent")
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
    raw_artifacts = state.get("artifacts", {})
    if not isinstance(raw_artifacts, Mapping):
        errors.append("run_state.json artifacts must be a mapping")
        raw_artifacts = {}
    for name, record in raw_artifacts.items():
        if not isinstance(record, Mapping):
            errors.append(f"artifact {name} record must be a mapping")
            continue
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            errors.append(f"artifact {name} has an invalid path")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            errors.append(f"artifact {name} is missing: {path}")
            continue
        checked += 1
        try:
            expected_bytes = int(record.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
            errors.append(f"artifact {name} has an invalid byte count")
        if path.stat().st_size != expected_bytes:
            errors.append(f"artifact {name} byte count changed")
        if sha256_file(path) != record.get("sha256"):
            errors.append(f"artifact {name} SHA-256 changed")
        producer = record.get("producer")
        if producer not in WORKFLOW_STEPS:
            errors.append(f"artifact {name} has an invalid producer")
        else:
            step_record = raw_steps.get(producer)
            declared = step_record.get("artifacts", []) if isinstance(step_record, Mapping) else []
            if not isinstance(declared, list) or name not in declared:
                errors.append(f"artifact {name} is not linked from its producer step")
    return {
        "schema_version": "af3h_audit_v1",
        "kind": "antibody_workflow",
        "status": "pass" if not errors else "fail",
        "run_status": run_status,
        "checked_artifacts": checked,
        "errors": errors,
    }


def _hallucination_audit(root: Path) -> dict[str, Any]:
    try:
        summary = _json(root / "summary.json")
        checkpoints = _json(root / "checkpoints.json")
    except Exception as exc:  # noqa: BLE001 - return a structured corruption report
        return _failed("hallucination", f"run manifest is invalid: {exc}")
    errors: list[str] = []
    if summary.get("schema_version") != "af3h_hallucination_summary_v1":
        errors.append("summary.json has an unsupported schema_version")
    if summary.get("status") != "completed":
        errors.append("Hallucination run is not completed")
    if checkpoints.get("schema_version") != "af3h_checkpoints_v1":
        errors.append("checkpoints.json has an unsupported schema_version")
    config_path = root / "resolved_config.json"
    config = None
    if not config_path.is_file():
        errors.append("resolved_config.json is missing")
    else:
        try:
            config = parse_config(_json(config_path), source_path=config_path)
            if config.sha256 != summary.get("config_sha256"):
                errors.append("resolved config hash differs from summary.json")
            hallucination_sha256 = canonical_sha256(dataclasses.asdict(config.hallucination))
            if summary.get("hallucination_sha256") != hallucination_sha256:
                errors.append("Hallucination config hash differs from summary.json")
            if checkpoints.get("hallucination_sha256") != hallucination_sha256:
                errors.append("Hallucination config hash differs from checkpoints.json")
            if checkpoints.get("input_sha256") != summary.get("input_sha256"):
                errors.append("input hash differs between summary and checkpoints")
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
    if not isinstance(records, list):
        errors.append("checkpoints.json checkpoints must be a list")
        records = []
    checked = 0
    seen_forwards: set[int] = set()
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"checkpoint {index} record must be a mapping")
            continue
        forward = record.get("evaluation_forward")
        if isinstance(forward, bool) or not isinstance(forward, int) or forward < 1:
            errors.append(f"checkpoint {index} has an invalid evaluation_forward")
        else:
            if forward in seen_forwards:
                errors.append(f"checkpoint evaluation_forward is duplicated: {forward}")
            seen_forwards.add(forward)
        if not isinstance(record.get("sha256"), str):
            errors.append(f"checkpoint {index} has an invalid SHA-256")
        if not isinstance(record.get("sequence_options"), Mapping):
            errors.append(f"checkpoint {index} has invalid sequence_options")
        path = _contained_path(
            root,
            record.get("path"),
            label=f"checkpoint {index}",
            errors=errors,
        )
        if path is None:
            continue
        if path in seen_paths:
            errors.append(f"checkpoint path is duplicated: {record.get('path')}")
        seen_paths.add(path)
        if not path.is_file():
            errors.append(f"checkpoint is missing: {record.get('path')}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"checkpoint hash changed: {record.get('path')}")
        else:
            checked += 1
    final_logits = (root / "final_logits.npy").resolve()
    final_logits_contained = final_logits.is_relative_to(root)
    if not final_logits_contained:
        errors.append("final_logits.npy resolves outside the run directory")
    if final_logits_contained and not final_logits.is_file():
        errors.append("final_logits.npy is missing")
    elif final_logits_contained and sha256_file(final_logits) != summary.get(
        "final_logits_sha256"
    ):
        errors.append("final logits hash changed")
    final_sequence = (root / "final_sequence.txt").resolve()
    final_sequence_contained = final_sequence.is_relative_to(root)
    if not final_sequence_contained:
        errors.append("final_sequence.txt resolves outside the run directory")
    if final_sequence_contained and not final_sequence.is_file():
        errors.append("final_sequence.txt is missing")
    elif final_sequence_contained:
        import hashlib

        sequence = final_sequence.read_text().strip()
        if hashlib.sha256(sequence.encode()).hexdigest() != summary.get(
            "final_sequence_sha256"
        ):
            errors.append("final sequence hash changed")
    trajectory = (root / "trajectory.jsonl").resolve()
    trajectory_contained = trajectory.is_relative_to(root)
    if not trajectory_contained:
        errors.append("trajectory.jsonl resolves outside the run directory")
    if trajectory_contained and not trajectory.is_file():
        errors.append("trajectory.jsonl is missing")
        trajectory_count = 0
    elif trajectory_contained:
        trajectory_count = 0
        for line_number, line in enumerate(trajectory.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            trajectory_count += 1
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("row is not an object")
            except Exception as exc:  # noqa: BLE001 - include every corrupt row
                errors.append(f"trajectory row {line_number} is invalid: {exc}")
        try:
            completed_steps = int(summary.get("completed_steps", -1))
        except (TypeError, ValueError):
            completed_steps = -1
            errors.append("summary completed_steps is invalid")
        if trajectory_count != completed_steps:
            errors.append("trajectory row count differs from completed_steps")
    else:
        trajectory_count = 0
    try:
        checkpoint_count = int(summary.get("checkpoint_count", -1))
    except (TypeError, ValueError):
        checkpoint_count = -1
        errors.append("summary checkpoint_count is invalid")
    if len(records) != checkpoint_count:
        errors.append("checkpoint record count differs from checkpoint_count")
    return {
        "schema_version": "af3h_audit_v1",
        "kind": "hallucination",
        "status": "pass" if not errors else "fail",
        "run_status": summary.get("status"),
        "checked_checkpoints": checked,
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
