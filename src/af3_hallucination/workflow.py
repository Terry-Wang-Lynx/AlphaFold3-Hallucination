"""Resumable, fail-closed workflow engine."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .config import WORKFLOW_STEPS, ProjectConfig
from .errors import ResumeError
from .hashing import canonical_sha256, sha256_file
from .plugins import PluginRegistry, default_registry
from .version import __version__


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclasses.dataclass
class WorkflowContext:
    output_dir: Path
    run_id: str
    seed: int
    artifacts: dict[str, Path]
    project_config: ProjectConfig | None = None
    config_dir: Path = Path(".")
    step_name: str = ""
    step_dir: Path = Path(".")


class AntibodyWorkflow:
    """Execute the five public antibody steps with exact resume provenance."""

    def __init__(self, config: ProjectConfig, *, registry: PluginRegistry | None = None):
        if config.kind != "antibody" or config.antibody is None:
            raise ValueError("AntibodyWorkflow requires kind='antibody'")
        self.config = config
        self.registry = registry or default_registry()

    def plan(self) -> list[dict[str, Any]]:
        assert self.config.antibody is not None
        return [
            {
                "index": index,
                "step": name,
                "plugin": self.config.antibody.steps[name].plugin,
                "config": self.config.antibody.steps[name].config,
            }
            for index, name in enumerate(WORKFLOW_STEPS)
        ]

    def _new_state(self, output_dir: Path) -> dict[str, Any]:
        run_id = str(self.config.run.get("id") or uuid.uuid4())
        return {
            "schema_version": "af3h_workflow_state_v1",
            "package_version": __version__,
            "run_id": run_id,
            "kind": self.config.kind,
            "config_sha256": self.config.sha256,
            "status": "running",
            "created_at_unix": time.time(),
            "updated_at_unix": time.time(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "hostname": socket.gethostname(),
            },
            "output_dir": str(output_dir),
            "steps": {},
            "artifacts": {},
        }

    def _load_or_create(self, output_dir: Path, resume: bool) -> dict[str, Any]:
        state_path = output_dir / "run_state.json"
        if not state_path.exists():
            return self._new_state(output_dir)
        if not resume:
            raise ResumeError(f"run already exists: {output_dir}; use --resume")
        state = json.loads(state_path.read_text())
        if state.get("config_sha256") != self.config.sha256:
            raise ResumeError("existing run config hash differs from the requested config")
        for name, record in state.get("artifacts", {}).items():
            path = Path(record.get("path", ""))
            if not path.is_file():
                raise ResumeError(f"recorded artifact is missing: {name}")
            if path.stat().st_size != int(record.get("bytes", -1)):
                raise ResumeError(f"recorded artifact byte count changed: {name}")
            if sha256_file(path) != record.get("sha256"):
                raise ResumeError(f"recorded artifact hash changed: {name}")
        return state

    def run(self, output_dir: str | Path, *, resume: bool = False) -> dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        state = self._load_or_create(root, resume)
        state_path = root / "run_state.json"
        if resume and state.get("status") in {"completed", "rejected"}:
            return state
        _atomic_json(root / "resolved_config.json", self.config.to_dict())
        _atomic_json(state_path, state)
        artifacts = {
            name: Path(record["path"])
            for name, record in state.get("artifacts", {}).items()
        }
        context = WorkflowContext(
            output_dir=root,
            run_id=state["run_id"],
            seed=self.config.hallucination.seed,
            artifacts=artifacts,
            project_config=self.config,
            config_dir=(
                self.config.source_path.parent
                if self.config.source_path is not None
                else Path.cwd()
            ),
        )
        assert self.config.antibody is not None
        try:
            for index, step_name in enumerate(WORKFLOW_STEPS):
                existing = state["steps"].get(step_name)
                if existing and existing.get("status") == "completed":
                    continue
                spec = self.config.antibody.steps[step_name]
                context.step_name = step_name
                context.step_dir = root / f"{index:02d}_{step_name}"
                context.step_dir.mkdir(parents=True, exist_ok=True)
                context.artifacts = artifacts
                plugin = self.registry.create(step_name, spec.plugin)
                started = time.time()
                state["steps"][step_name] = {
                    "status": "running",
                    "plugin": spec.plugin,
                    "config_sha256": canonical_sha256(spec.config),
                    "started_at_unix": started,
                }
                state["updated_at_unix"] = time.time()
                _atomic_json(state_path, state)
                result = plugin.run(context=context, config=spec.config)
                artifact_records = {}
                for name, path in result.artifacts.items():
                    if name in state["artifacts"]:
                        raise RuntimeError(f"plugin attempted to replace artifact {name!r}")
                    resolved = Path(path).resolve()
                    artifact_records[name] = {
                        "path": str(resolved),
                        "sha256": sha256_file(resolved),
                        "bytes": resolved.stat().st_size,
                        "producer": step_name,
                    }
                    artifacts[name] = resolved
                state["artifacts"].update(artifact_records)
                state["steps"][step_name] = {
                    **state["steps"][step_name],
                    "status": result.status,
                    "completed_at_unix": time.time(),
                    "runtime_seconds": round(time.time() - started, 6),
                    "metrics": result.metrics,
                    "artifacts": sorted(artifact_records),
                }
                state["updated_at_unix"] = time.time()
                _atomic_json(state_path, state)
                if result.status in {"rejected", "skipped"}:
                    state["status"] = result.status
                    for later_name in WORKFLOW_STEPS[index + 1 :]:
                        state["steps"].setdefault(
                            later_name,
                            {
                                "status": "skipped",
                                "reason": f"upstream step {step_name} returned {result.status}",
                            },
                        )
                    state["completed_at_unix"] = time.time()
                    state["updated_at_unix"] = time.time()
                    _atomic_json(state_path, state)
                    return state
        except Exception as exc:
            state["status"] = "failed"
            state["failure_type"] = type(exc).__name__
            state["failure_message"] = str(exc)
            state["updated_at_unix"] = time.time()
            _atomic_json(state_path, state)
            raise
        state["status"] = "completed"
        state["completed_at_unix"] = time.time()
        state["updated_at_unix"] = time.time()
        _atomic_json(state_path, state)
        return state
