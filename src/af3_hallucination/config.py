"""Strict, import-light YAML configuration for public workflows."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

SCHEMA_VERSION = 1
STAGE_TYPES = ("logits", "soft", "hard", "semigreedy")
WORKFLOW_STEPS = (
    "hallucination",
    "diffusion",
    "inverse_folding",
    "consistency",
    "final_evaluation",
)


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ConfigurationError(f"{name} must be {qualifier}")
    return result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if result != value or result < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return result


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return dict(value)


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One freely ordered Hallucination stage."""

    type: str
    steps: int
    learning_rate: float | None = None
    soft_start: float | None = None
    soft_end: float | None = None
    temperature_start: float | None = None
    temperature_end: float | None = None
    hard_start: float | None = None
    hard_end: float | None = None
    step_start: float = 1.0
    step_end: float | None = None
    dropout: bool = False
    tries: int | None = None
    name: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> StageSpec:
        data = dict(value)
        stage_type = str(data.pop("type", "")).strip()
        if stage_type not in STAGE_TYPES:
            raise ConfigurationError(
                f"hallucination.stages[{index}].type must be one of {STAGE_TYPES}"
            )
        steps = _integer(f"hallucination.stages[{index}].steps", data.pop("steps", None))
        defaults = {
            "logits": (0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            "soft": (1.0, 1.0, 1.0, 0.01, 0.0, 0.0),
            "hard": (1.0, 1.0, 0.01, 0.01, 1.0, 1.0),
            "semigreedy": (1.0, 1.0, 0.01, 0.01, 1.0, 1.0),
        }[stage_type]
        numeric = {}
        keys = (
            "soft_start",
            "soft_end",
            "temperature_start",
            "temperature_end",
            "hard_start",
            "hard_end",
        )
        for key, default in zip(keys, defaults, strict=True):
            numeric[key] = _finite(
                f"hallucination.stages[{index}].{key}", data.pop(key, default)
            )
        for key in ("soft_start", "soft_end", "hard_start", "hard_end"):
            if not 0.0 <= numeric[key] <= 1.0:
                raise ConfigurationError(f"hallucination.stages[{index}].{key} must be in [0,1]")
        for key in ("temperature_start", "temperature_end"):
            if numeric[key] <= 0:
                raise ConfigurationError(f"hallucination.stages[{index}].{key} must be positive")
        lr_raw = data.pop("learning_rate", None)
        lr = None if lr_raw is None else _finite(
            f"hallucination.stages[{index}].learning_rate", lr_raw, positive=True
        )
        step_start = _finite(
            f"hallucination.stages[{index}].step_start", data.pop("step_start", 1.0),
            positive=True,
        )
        step_end_raw = data.pop("step_end", None)
        step_end = None if step_end_raw is None else _finite(
            f"hallucination.stages[{index}].step_end", step_end_raw, positive=True
        )
        tries_raw = data.pop("tries", None)
        tries = None if tries_raw is None else _integer(
            f"hallucination.stages[{index}].tries", tries_raw, minimum=1
        )
        dropout = data.pop("dropout", False)
        if not isinstance(dropout, bool):
            raise ConfigurationError(f"hallucination.stages[{index}].dropout must be boolean")
        if dropout:
            raise ConfigurationError(
                f"hallucination.stages[{index}].dropout=true is not supported by "
                "the current official AF3 trunk adapter"
            )
        name_raw = data.pop("name", None)
        name = None if name_raw is None else str(name_raw).strip()
        if name == "":
            raise ConfigurationError(f"hallucination.stages[{index}].name cannot be empty")
        if data:
            raise ConfigurationError(
                f"unknown keys in hallucination.stages[{index}]: {sorted(data)}"
            )
        return cls(
            type=stage_type,
            steps=steps,
            learning_rate=lr,
            step_start=step_start,
            step_end=step_end,
            dropout=dropout,
            tries=tries,
            name=name,
            **numeric,
        )


@dataclasses.dataclass(frozen=True)
class LossSpec:
    type: str
    weight: float
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> LossSpec:
        data = dict(value)
        loss_type = str(data.pop("type", "")).strip()
        if not loss_type:
            raise ConfigurationError(f"hallucination.losses[{index}].type is required")
        weight = _finite(f"hallucination.losses[{index}].weight", data.pop("weight", 1.0))
        parameters = _mapping(
            f"hallucination.losses[{index}].parameters", data.pop("parameters", {})
        )
        if data:
            raise ConfigurationError(
                f"unknown keys in hallucination.losses[{index}]: {sorted(data)}"
            )
        return cls(type=loss_type, weight=weight, parameters=parameters)


@dataclasses.dataclass(frozen=True)
class HallucinationSpec:
    backend: str
    stages: tuple[StageSpec, ...]
    losses: tuple[LossSpec, ...]
    learning_rate: float = 0.1
    optimizer: str = "sgd"
    seed: int = 0
    bucket: int = 256
    design_recycles: int = 0
    alpha: float = 2.0
    init_scale: float = 0.01
    omit_amino_acids: str = "C"
    checkpoints: tuple[int, ...] = ()
    stopper: dict[str, Any] = dataclasses.field(default_factory=dict)
    backend_config: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HallucinationSpec:
        data = dict(value)
        backend = str(data.pop("backend", "af3_jax")).strip()
        if not backend:
            raise ConfigurationError("hallucination.backend cannot be empty")
        raw_stages = data.pop("stages", None)
        if not isinstance(raw_stages, (list, tuple)) or not raw_stages:
            raise ConfigurationError("hallucination.stages must be a non-empty list")
        stages = tuple(StageSpec.from_mapping(_mapping("stage", row), i) for i, row in enumerate(raw_stages))
        raw_losses = data.pop("losses", [])
        if not isinstance(raw_losses, (list, tuple)):
            raise ConfigurationError("hallucination.losses must be a list")
        losses = tuple(LossSpec.from_mapping(_mapping("loss", row), i) for i, row in enumerate(raw_losses))
        optimizer = str(data.pop("optimizer", "sgd"))
        if optimizer not in {"sgd", "adam", "adabelief", "rmsprop"}:
            raise ConfigurationError("hallucination.optimizer is unsupported")
        seed = _integer("hallucination.seed", data.pop("seed", 0))
        bucket = _integer("hallucination.bucket", data.pop("bucket", 256), minimum=1)
        recycles = _integer("hallucination.design_recycles", data.pop("design_recycles", 0))
        learning_rate = _finite(
            "hallucination.learning_rate", data.pop("learning_rate", 0.1), positive=True
        )
        alpha = _finite("hallucination.alpha", data.pop("alpha", 2.0), positive=True)
        init_scale = _finite("hallucination.init_scale", data.pop("init_scale", 0.01))
        if init_scale < 0:
            raise ConfigurationError("hallucination.init_scale must be nonnegative")
        omit = str(data.pop("omit_amino_acids", "C")).upper()
        if any(aa not in "ARNDCQEGHILKMFPSTWYV" for aa in omit):
            raise ConfigurationError("hallucination.omit_amino_acids contains a non-protein code")
        raw_checkpoints = data.pop("checkpoints", [])
        if not isinstance(raw_checkpoints, (list, tuple)):
            raise ConfigurationError("hallucination.checkpoints must be a list")
        checkpoints = tuple(sorted({_integer("hallucination.checkpoints[]", x, minimum=1) for x in raw_checkpoints}))
        total_gradient_steps = sum(stage.steps for stage in stages if stage.type != "semigreedy")
        if checkpoints and checkpoints[-1] > total_gradient_steps:
            raise ConfigurationError("a hallucination checkpoint exceeds the number of gradient steps")
        stopper = _mapping("hallucination.stopper", data.pop("stopper", {"type": "none"}))
        backend_config = _mapping("hallucination.backend_config", data.pop("backend_config", {}))
        if data:
            raise ConfigurationError(f"unknown hallucination keys: {sorted(data)}")
        return cls(
            backend=backend,
            stages=stages,
            losses=losses,
            learning_rate=learning_rate,
            optimizer=optimizer,
            seed=seed,
            bucket=bucket,
            design_recycles=recycles,
            alpha=alpha,
            init_scale=init_scale,
            omit_amino_acids=omit,
            checkpoints=checkpoints,
            stopper=stopper,
            backend_config=backend_config,
        )


@dataclasses.dataclass(frozen=True)
class PluginSpec:
    plugin: str
    config: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> PluginSpec:
        data = dict(value)
        plugin = str(data.pop("plugin", "")).strip()
        if not plugin:
            raise ConfigurationError(f"antibody.{name}.plugin is required")
        config = _mapping(f"antibody.{name}.config", data.pop("config", {}))
        if data:
            raise ConfigurationError(f"unknown antibody.{name} keys: {sorted(data)}")
        return cls(plugin=plugin, config=config)


@dataclasses.dataclass(frozen=True)
class AntibodySpec:
    steps: dict[str, PluginSpec]
    cdr_only: bool = True
    framework_fixed: bool = True
    antigen_fixed: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AntibodySpec:
        data = dict(value)
        steps: dict[str, PluginSpec] = {}
        for name in WORKFLOW_STEPS:
            raw = data.pop(name, None)
            if raw is None:
                raise ConfigurationError(f"antibody.{name} is required")
            steps[name] = PluginSpec.from_mapping(name, _mapping(f"antibody.{name}", raw))
        flags = {}
        for name, default in (
            ("cdr_only", True),
            ("framework_fixed", True),
            ("antigen_fixed", True),
        ):
            flags[name] = data.pop(name, default)
            if not isinstance(flags[name], bool):
                raise ConfigurationError(f"antibody.{name} must be boolean")
        if not all(flags.values()):
            raise ConfigurationError(
                "the public antibody workflow currently requires cdr_only, framework_fixed, and antigen_fixed"
            )
        if data:
            raise ConfigurationError(f"unknown antibody keys: {sorted(data)}")
        return cls(steps=steps, **flags)


@dataclasses.dataclass(frozen=True)
class ProjectConfig:
    kind: str
    hallucination: HallucinationSpec
    antibody: AntibodySpec | None
    run: dict[str, Any]
    extensions: dict[str, Any]
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value.pop("source_path", None)
        if value["antibody"] is not None:
            antibody = value["antibody"]
            steps = antibody.pop("steps")
            antibody.update(steps)
        value["schema_version"] = SCHEMA_VERSION
        return value

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def parse_config(value: Mapping[str, Any], *, source_path: Path | None = None) -> ProjectConfig:
    data = dict(value)
    version = _integer("schema_version", data.pop("schema_version", None), minimum=1)
    if version != SCHEMA_VERSION:
        raise ConfigurationError(f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")
    kind = str(data.pop("kind", "")).strip()
    if kind not in {"hallucination", "antibody"}:
        raise ConfigurationError("kind must be 'hallucination' or 'antibody'")
    hallucination = HallucinationSpec.from_mapping(
        _mapping("hallucination", data.pop("hallucination", None))
    )
    antibody_raw = data.pop("antibody", None)
    antibody = None
    if kind == "antibody":
        antibody = AntibodySpec.from_mapping(_mapping("antibody", antibody_raw))
    elif antibody_raw is not None:
        raise ConfigurationError("antibody is only valid when kind='antibody'")
    run = _mapping("run", data.pop("run", {}))
    extensions = _mapping("extensions", data.pop("extensions", {}))
    if data:
        raise ConfigurationError(f"unknown top-level keys: {sorted(data)}")
    return ProjectConfig(
        kind=kind,
        hallucination=hallucination,
        antibody=antibody,
        run=run,
        extensions=extensions,
        source_path=source_path,
    )


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(source.read_text())
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {source}: {exc}") from exc
    return parse_config(_mapping("document", value), source_path=source)
