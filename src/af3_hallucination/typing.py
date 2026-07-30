"""Shared lightweight types with no JAX or AlphaFold dependency."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

JSONValue: TypeAlias = (
    type(None) | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


@runtime_checkable
class WorkflowPlugin(Protocol):
    """One resumable workflow operation."""

    def run(self, *, context: RunContext, config: Mapping[str, Any]) -> StepResult: ...


class RunContext(Protocol):
    output_dir: Path
    run_id: str
    seed: int
    artifacts: Mapping[str, Path]


class StepResult(Protocol):
    status: str
    artifacts: Mapping[str, Path]
    metrics: Mapping[str, JSONValue]
