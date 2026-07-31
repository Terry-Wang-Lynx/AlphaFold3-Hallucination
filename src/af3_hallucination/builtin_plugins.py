"""First-party workflow plugins, with model imports deferred until execution."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import normalize_candidate_pool
from .errors import PluginError
from .hallucination import run_hallucination
from .plugins import CommandPlugin, PluginRegistry, PluginResult

_ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _runtime_path(context, value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise PluginError(f"{name} must be a non-empty path")
    raw = str(value)
    missing = sorted(
        {
            match.group(1) or match.group(2)
            for match in _ENV_REFERENCE.finditer(raw)
            if (match.group(1) or match.group(2)) not in os.environ
        }
    )
    if missing:
        raise PluginError(f"{name} references unset environment variables: {missing}")
    expanded = Path(os.path.expandvars(raw)).expanduser()
    if not expanded.is_absolute():
        expanded = context.config_dir / expanded
    return expanded.resolve()


def _artifact(context, config: Mapping[str, Any], key: str, default: str) -> Path:
    name = str(config.get(key, default))
    try:
        return Path(context.artifacts[name]).resolve()
    except KeyError as exc:
        raise PluginError(f"required upstream artifact is missing: {name}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _require_known_keys(config: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise PluginError(f"unknown {label} plugin keys: {sorted(unknown)}")


class AF3JaxHallucinationPlugin:
    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        _require_known_keys(config, {"input_json", "model_dir"}, "af3_jax")
        if context.project_config is None:
            raise PluginError("AF3 Hallucination plugin requires the resolved project config")
        backend = context.project_config.hallucination.backend_config
        design_local = backend.get("design_local_indices")
        if not isinstance(design_local, (list, tuple)) or not design_local:
            raise PluginError(
                "the CDR-only antibody workflow requires non-empty "
                "hallucination.backend_config.design_local_indices"
            )
        if "design_token_indices" in backend:
            raise PluginError(
                "the CDR-only antibody workflow requires chain-local design indices, "
                "not design_token_indices"
            )
        input_json = _runtime_path(context, config.get("input_json"), "input_json")
        model_dir = _runtime_path(context, config.get("model_dir"), "model_dir")
        run_dir = Path(context.step_dir) / "run"
        summary = run_hallucination(
            context.project_config,
            input_json=input_json,
            model_dir=model_dir,
            output_dir=run_dir,
        )
        return PluginResult(
            status="completed",
            artifacts={
                "populated_input": input_json,
                "hallucination_summary": run_dir / "summary.json",
                "hallucination_checkpoints": run_dir / "checkpoints.json",
                "hallucination_logits": run_dir / "final_logits.npy",
                "hallucination_sequence": run_dir / "final_sequence.txt",
            },
            metrics={
                "completed_steps": summary["completed_steps"],
                "final_loss": summary["final_loss"],
                "checkpoint_count": summary["checkpoint_count"],
            },
        )


class AF3DiffusionPlugin:
    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        _require_known_keys(
            config,
            {
                "input_artifact",
                "checkpoints_artifact",
                "model_dir",
                "checkpoint_forward",
                "trunk_seed",
                "diffusion_seed",
                "diffusion_steps",
                "num_recycles",
            },
            "af3 diffusion",
        )
        if context.project_config is None:
            raise PluginError("AF3 diffusion plugin requires the resolved project config")
        from .af3.runtime import decode_checkpoint

        input_json = _artifact(context, config, "input_artifact", "populated_input")
        checkpoints = _artifact(
            context, config, "checkpoints_artifact", "hallucination_checkpoints"
        )
        model_dir = _runtime_path(context, config.get("model_dir"), "model_dir")
        result = decode_checkpoint(
            public_spec=context.project_config.hallucination,
            input_json=input_json,
            checkpoints_json=checkpoints,
            model_dir=model_dir,
            output_dir=context.step_dir,
            checkpoint_forward=(
                None
                if config.get("checkpoint_forward") is None
                else int(config["checkpoint_forward"])
            ),
            trunk_seed=int(config.get("trunk_seed", context.seed)),
            diffusion_seed=int(config.get("diffusion_seed", context.seed)),
            diffusion_steps=int(config.get("diffusion_steps", 200)),
            num_recycles=int(config.get("num_recycles", 3)),
        )
        return PluginResult(
            status="completed",
            artifacts={
                "anchor_manifest": result["manifest_path"],
                "anchor_structure": result["structure_path"],
                "anchor_geometry": result["geometry_path"],
                "anchor_input": result["candidate_input_path"],
            },
            metrics=result["metrics"],
        )


def _anchor_contract(context) -> tuple[dict[str, Any], Path]:
    anchor_path = _artifact(context, {}, "anchor_artifact", "anchor_manifest")
    anchor = json.loads(anchor_path.read_text())
    return anchor, anchor_path


class FrozenCandidatePlugin:
    """Publish user-supplied candidates for tests and frozen-sequence evaluations."""

    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        _require_known_keys(config, {"candidates"}, "frozen_candidates")
        anchor, _ = _anchor_contract(context)
        candidates = normalize_candidate_pool(
            {"candidates": config.get("candidates")},
            reference_sequence=anchor["hard_sequence"],
            design_local_indices=anchor["design_local_indices"],
        )
        path = Path(context.step_dir) / "candidates.json"
        _write_json(
            path,
            {
                "schema_version": "af3h_candidate_pool_v1",
                "status": "completed",
                "source": "frozen_candidates",
                "candidates": candidates,
            },
        )
        return PluginResult(
            status="completed",
            artifacts={"candidate_pool": path},
            metrics={"candidate_count": len(candidates), "model_calls": 0},
        )


class CandidateCommandPlugin:
    """Run any inverse folder through a strict CDR-only candidate JSON contract."""

    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        command_result = CommandPlugin().run(context=context, config=config)
        if command_result.status != "completed":
            return command_result
        try:
            pool_path = command_result.artifacts["candidate_pool"]
        except KeyError as exc:
            raise PluginError(
                "candidate_command must declare artifacts.candidate_pool"
            ) from exc
        anchor, _ = _anchor_contract(context)
        payload = json.loads(pool_path.read_text())
        candidates = normalize_candidate_pool(
            payload,
            reference_sequence=anchor["hard_sequence"],
            design_local_indices=anchor["design_local_indices"],
        )
        validated = Path(context.step_dir) / "validated_candidates.json"
        _write_json(
            validated,
            {
                "schema_version": "af3h_candidate_pool_v1",
                "status": "completed",
                "source": "candidate_command",
                "candidates": candidates,
            },
        )
        return PluginResult(
            status="completed",
            artifacts={"candidate_pool": validated, "candidate_pool_raw": pool_path},
            metrics={**command_result.metrics, "candidate_count": len(candidates)},
        )


class AF3ConsistencyPlugin:
    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        _require_known_keys(
            config,
            {
                "anchor_artifact",
                "candidate_artifact",
                "model_dir",
                "scorer_seed",
                "design_recycles",
                "cdr3_local_indices",
                "thresholds",
                "top_k",
            },
            "af3_fixed_geometry",
        )
        from .af3.runtime import score_consistency_pool

        anchor = _artifact(context, config, "anchor_artifact", "anchor_manifest")
        pool = _artifact(context, config, "candidate_artifact", "candidate_pool")
        model_dir = _runtime_path(context, config.get("model_dir"), "model_dir")
        result = score_consistency_pool(
            anchor_manifest=anchor,
            candidate_pool=pool,
            model_dir=model_dir,
            output_dir=context.step_dir,
            scorer_seed=int(config.get("scorer_seed", context.seed)),
            design_recycles=int(config.get("design_recycles", 1)),
            cdr3_local_indices=tuple(int(x) for x in config.get("cdr3_local_indices", ())),
            thresholds=dict(config.get("thresholds", {})),
            top_k=int(config.get("top_k", 1)),
        )
        status = "completed" if result["eligible_count"] > 0 else "rejected"
        return PluginResult(
            status=status,
            artifacts={"consistency_results": result["result_path"]},
            metrics={
                "candidate_count": result["candidate_count"],
                "eligible_count": result["eligible_count"],
                "diffusion_calls": 0,
            },
        )


class AF3FinalEvaluationPlugin:
    def run(self, *, context, config: Mapping[str, Any]) -> PluginResult:
        _require_known_keys(
            config,
            {
                "anchor_artifact",
                "selected_artifact",
                "model_dir",
                "seeds",
                "diffusion_steps",
                "num_recycles",
            },
            "af3 final evaluation",
        )
        from .af3.runtime import evaluate_selected_candidates

        anchor = _artifact(context, config, "anchor_artifact", "anchor_manifest")
        selected = _artifact(context, config, "selected_artifact", "consistency_results")
        model_dir = _runtime_path(context, config.get("model_dir"), "model_dir")
        result = evaluate_selected_candidates(
            anchor_manifest=anchor,
            consistency_results=selected,
            model_dir=model_dir,
            output_dir=context.step_dir,
            seeds=tuple(int(x) for x in config.get("seeds", (context.seed,))),
            diffusion_steps=int(config.get("diffusion_steps", 200)),
            num_recycles=int(config.get("num_recycles", 3)),
        )
        artifacts = {"final_evaluation": result["result_path"]}
        for index, path in enumerate(result["structure_paths"]):
            artifacts[f"final_structure_{index:03d}"] = path
        return PluginResult(
            status="completed",
            artifacts=artifacts,
            metrics={
                "candidate_count": result["candidate_count"],
                "full_af3_calls": result["full_af3_calls"],
            },
        )


def register_builtin_plugins(registry: PluginRegistry) -> None:
    registry.register("hallucination", "af3_jax", AF3JaxHallucinationPlugin)
    registry.register("diffusion", "af3", AF3DiffusionPlugin)
    registry.register("inverse_folding", "frozen_candidates", FrozenCandidatePlugin)
    registry.register("inverse_folding", "candidate_command", CandidateCommandPlugin)
    registry.register("consistency", "af3_fixed_geometry", AF3ConsistencyPlugin)
    registry.register("final_evaluation", "af3", AF3FinalEvaluationPlugin)
