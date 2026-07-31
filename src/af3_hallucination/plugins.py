"""Plugin registry and built-in orchestration plugins."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .errors import PluginError
from .hashing import sha256_file

PluginFactory = Callable[[], Any]


class PluginRegistry:
    """Explicit plugin registry with no import-time entry-point side effects."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, PluginFactory]] = {}

    def register(
        self, kind: str, name: str, factory: PluginFactory, *, replace: bool = False
    ) -> None:
        if not kind or not name:
            raise PluginError("plugin kind and name must be non-empty")
        family = self._factories.setdefault(kind, {})
        if name in family and not replace:
            raise PluginError(f"plugin already registered: {kind}/{name}")
        family[name] = factory

    def create(self, kind: str, name: str) -> Any:
        try:
            factory = self._factories[kind][name]
        except KeyError as exc:
            available = sorted(self._factories.get(kind, {}))
            raise PluginError(
                f"unknown plugin {kind}/{name}; available: {available or 'none'}"
            ) from exc
        return factory()

    def inventory(self) -> dict[str, list[str]]:
        return {kind: sorted(family) for kind, family in sorted(self._factories.items())}


@dataclasses.dataclass(frozen=True)
class PluginResult:
    status: str
    artifacts: dict[str, Path]
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"completed", "rejected", "skipped"}:
            raise PluginError(f"invalid plugin result status: {self.status}")


def _render_token(token: str, values: Mapping[str, str]) -> str:
    try:
        return token.format_map(values)
    except KeyError as exc:
        raise PluginError(f"unknown command template field: {exc.args[0]}") from exc


class CommandPlugin:
    """Run an external adapter without invoking a shell.

    The command is a YAML list. Tokens may reference `{output_dir}`, `{step_dir}`,
    `{run_id}`, `{seed}`, and artifact names from earlier workflow steps.
    """

    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        unknown = set(config) - {
            "command",
            "env",
            "timeout_seconds",
            "result_json",
            "artifacts",
        }
        if unknown:
            raise PluginError(f"unknown command plugin keys: {sorted(unknown)}")
        command = config.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) for x in command)
        ):
            raise PluginError(
                "command plugin requires config.command as a non-empty string list"
            )
        step_dir = Path(context.step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "output_dir": str(context.output_dir),
            "step_dir": str(step_dir),
            "run_id": context.run_id,
            "seed": str(context.seed),
            **{name: str(path) for name, path in context.artifacts.items()},
        }
        rendered = [_render_token(token, values) for token in command]
        env = os.environ.copy()
        raw_env = config.get("env", {})
        if not isinstance(raw_env, Mapping):
            raise PluginError("command plugin config.env must be a mapping")
        env.update(
            {str(key): _render_token(str(value), values) for key, value in raw_env.items()}
        )
        raw_timeout = config.get("timeout_seconds")
        try:
            timeout = None if raw_timeout is None else float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise PluginError(
                "command plugin timeout_seconds must be finite and positive"
            ) from exc
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise PluginError("command plugin timeout_seconds must be finite and positive")
        started = time.time()
        try:
            completed = subprocess.run(
                rendered,
                cwd=str(step_dir),
                env=env,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
            (step_dir / "stdout.log").write_text(stdout or "")
            (step_dir / "stderr.log").write_text(stderr or "")
            raise PluginError(f"external command timed out after {timeout} seconds") from exc
        (step_dir / "stdout.log").write_text(completed.stdout)
        (step_dir / "stderr.log").write_text(completed.stderr)
        if completed.returncode != 0:
            raise PluginError(
                f"external command failed with exit {completed.returncode}: {shlex.join(rendered)}"
            )
        result_path = step_dir / str(config.get("result_json", "result.json"))
        payload: dict[str, Any] = {}
        if result_path.exists():
            value = json.loads(result_path.read_text())
            if not isinstance(value, dict):
                raise PluginError("command result JSON must contain an object")
            payload = value
        artifact_config = config.get("artifacts", {})
        if not isinstance(artifact_config, Mapping):
            raise PluginError("command plugin config.artifacts must be a mapping")
        artifacts = {}
        for name, raw_path in artifact_config.items():
            path = Path(_render_token(str(raw_path), values))
            if not path.is_absolute():
                path = step_dir / path
            if not path.is_file():
                raise PluginError(f"declared artifact is not a regular file: {path}")
            artifacts[str(name)] = path.resolve()
        raw_metrics = payload.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise PluginError("command result metrics must be a mapping")
        metrics = dict(raw_metrics)
        metrics.update(
            {
                "runtime_seconds": round(time.time() - started, 6),
                "return_code": completed.returncode,
                "command": rendered,
            }
        )
        return PluginResult(
            status=str(payload.get("status", "completed")),
            artifacts=artifacts,
            metrics=metrics,
        )


class MockPlugin:
    """Deterministic cross-platform workflow plugin used by examples and CI."""

    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        unknown = set(config) - {"artifact_name", "file_name", "value"}
        if unknown:
            raise PluginError(f"unknown mock plugin keys: {sorted(unknown)}")
        step_dir = Path(context.step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = str(config.get("artifact_name", f"{context.step_name}_artifact"))
        file_name = str(config.get("file_name", f"{context.step_name}.json"))
        output = step_dir / file_name
        payload = {
            "schema_version": "af3h_mock_plugin_v1",
            "run_id": context.run_id,
            "step": context.step_name,
            "seed": context.seed,
            "input_artifacts": {
                name: sha256_file(path) for name, path in context.artifacts.items()
            },
            "value": config.get("value", context.step_name),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return PluginResult(
            status="completed",
            artifacts={artifact_name: output.resolve()},
            metrics={"mock": True, "input_artifact_count": len(context.artifacts)},
        )


class PocketPlaceholderPlugin:
    """Intentional non-implementation for the future pocket workflow."""

    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        del context, config
        raise NotImplementedError(
            "small-molecule pocket redesign is a documented placeholder and is not implemented"
        )


def default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    for kind in (
        "hallucination",
        "diffusion",
        "inverse_folding",
        "consistency",
        "final_evaluation",
    ):
        registry.register(kind, "mock", MockPlugin)
        if kind != "inverse_folding":
            registry.register(kind, "command", CommandPlugin)
    registry.register("pocket_redesign", "not_implemented", PocketPlaceholderPlugin)
    from .builtin_plugins import register_builtin_plugins

    register_builtin_plugins(registry)
    return registry
