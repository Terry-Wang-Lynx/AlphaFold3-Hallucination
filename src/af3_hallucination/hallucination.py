"""High-level public Hallucination run entry point."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .hashing import sha256_file
from .schedule import SCHEDULE_PROVENANCE, expand_schedule
from .version import __version__


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _protein_sequence(path: Path, chain_id: str) -> str:
    document = json.loads(path.read_text())
    matches = [
        entry["protein"]["sequence"]
        for entry in document.get("sequences", [])
        if "protein" in entry and entry["protein"].get("id") == chain_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one protein chain {chain_id!r} in {path}")
    return str(matches[0])


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    if result.returncode:
        return []
    output = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            output.append({"index": fields[0], "name": fields[1], "memory_mib": fields[2]})
    return output


def _resolve_token_indices(input_json: Path, backend: dict[str, Any], bucket: int):
    import numpy as np

    from .af3.hybrid import binder_mask_from_batch, featurise

    binder_asym_id = int(backend.get("binder_asym_id", 2))
    batch = featurise(input_json, bucket)
    valid = np.asarray(batch["seq_mask"]) > 0
    asym = np.asarray(batch["asym_id"], np.int32)
    binder_mask = binder_mask_from_batch(batch, binder_asym_id)
    binder_indices = np.flatnonzero(binder_mask)
    if "design_token_indices" in backend and "design_local_indices" in backend:
        raise ValueError("provide only one of design_token_indices/design_local_indices")
    if "design_token_indices" in backend:
        design = np.asarray(backend["design_token_indices"], np.int32)
    elif "design_local_indices" in backend:
        local = np.asarray(backend["design_local_indices"], np.int32)
        if np.any(local < 0) or np.any(local >= binder_indices.size):
            raise ValueError("a design_local_indices value is outside the binder")
        design = binder_indices[local]
    else:
        design = binder_indices
    if "hotspot_token_indices" in backend and "hotspot_local_indices" in backend:
        raise ValueError("provide only one of hotspot_token_indices/hotspot_local_indices")
    if "hotspot_token_indices" in backend:
        hotspot = np.asarray(backend["hotspot_token_indices"], np.int32)
    elif "hotspot_local_indices" in backend:
        target_asym_id = int(backend.get("target_asym_id", 1))
        target_indices = np.flatnonzero(valid & (asym == target_asym_id))
        local = np.asarray(backend["hotspot_local_indices"], np.int32)
        if np.any(local < 0) or np.any(local >= target_indices.size):
            raise ValueError("a hotspot_local_indices value is outside the target")
        hotspot = target_indices[local]
    else:
        hotspot = np.flatnonzero(valid & ~binder_mask)
    target = None
    if "target_token_indices" in backend:
        target = tuple(int(value) for value in backend["target_token_indices"])
    return (
        binder_asym_id,
        tuple(int(value) for value in design),
        tuple(int(value) for value in hotspot),
        target,
    )


def run_hallucination(
    config: ProjectConfig,
    *,
    input_json: str | Path | None,
    model_dir: str | Path | None,
    output_dir: str | Path,
    dry_run: bool = False,
    semigreedy_scorer: Callable | None = None,
) -> dict[str, Any]:
    """Run or plan one AF3/JAX Hallucination trajectory."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = expand_schedule(config.hallucination)
    _write_json(root / "resolved_config.json", config.to_dict())
    _write_json(root / "schedule.json", {"provenance": SCHEDULE_PROVENANCE, "steps": plan})
    if dry_run:
        summary = {
            "schema_version": "af3h_hallucination_summary_v1",
            "status": "dry_run",
            "config_sha256": config.sha256,
            "planned_steps": len(plan),
            "gradient_steps": sum(row["gradient_enabled"] for row in plan),
            "checkpoints": list(config.hallucination.checkpoints),
        }
        _write_json(root / "summary.json", summary)
        return summary
    if (
        any(stage.type == "semigreedy" for stage in config.hallucination.stages)
        and semigreedy_scorer is None
    ):
        raise ValueError(
            "a semigreedy stage requires semigreedy_scorer through the Python API"
        )
    from .af3.engine import (
        AF3JaxHallucinationEngine,
        PreparedDesign,
        compile_stopper,
    )

    should_stop = compile_stopper(config.hallucination.stopper)
    if input_json is None or model_dir is None:
        raise ValueError("input_json and model_dir are required unless dry_run=True")
    input_path = Path(input_json).expanduser().resolve()
    model_path = Path(model_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not (model_path / "af3.bin").is_file():
        raise FileNotFoundError(model_path / "af3.bin")
    backend = dict(config.hallucination.backend_config)
    import numpy as np

    binder_chain_id = str(backend.get("binder_chain_id", "B"))
    binder_sequence = _protein_sequence(input_path, binder_chain_id)
    binder_asym_id, design_indices, hotspot_indices, target_indices = _resolve_token_indices(
        input_path, backend, config.hallucination.bucket
    )
    engine = AF3JaxHallucinationEngine(
        config.hallucination,
        PreparedDesign(
            input_json=input_path,
            binder_chain_id=binder_chain_id,
            binder_asym_id=binder_asym_id,
            binder_sequence=binder_sequence,
            design_token_indices=design_indices,
            hotspot_token_indices=hotspot_indices,
            target_token_indices=target_indices,
        ),
        model_dir=model_path,
        remat=bool(backend.get("remat", True)),
    )
    checkpoints = set(config.hallucination.checkpoints)
    checkpoint_records = []

    def checkpoint(row, logits):
        forward = int(row["evaluation_forward"])
        if forward not in checkpoints:
            return
        path = root / "checkpoints" / f"forward_{forward:06d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(logits, np.float32), allow_pickle=False)
        checkpoint_records.append(
            {
                "evaluation_forward": forward,
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "loss": row["loss"],
                "stage": row["stage"],
                "sequence_options": {
                    "soft": row["soft"],
                    "temperature": row["temperature"],
                    "hard": row["hard"],
                    "alpha": config.hallucination.alpha,
                },
            }
        )

    started = time.time()
    trajectory = engine.run(
        root / "scratch",
        step_callback=checkpoint,
        should_stop=should_stop,
        semigreedy_scorer=semigreedy_scorer,
    )
    final_logits = root / "final_logits.npy"
    np.save(final_logits, np.asarray(engine.logits, np.float32), allow_pickle=False)
    with (root / "trajectory.jsonl").open("w") as handle:
        for row in trajectory:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    final_binder_ids = engine.fixed_binder_ids.copy()
    final_binder_ids[engine.design_local_indices] = np.argmax(
        np.asarray(engine.logits + engine.bias), axis=-1
    )
    from .af3.engine import AF3_AA_ORDER

    final_sequence = "".join(AF3_AA_ORDER[int(value)] for value in final_binder_ids)
    (root / "final_sequence.txt").write_text(final_sequence + "\n")
    _write_json(root / "checkpoints.json", {"checkpoints": checkpoint_records})
    _write_json(root / "stage_summaries.json", engine.stage_summaries)
    summary = {
        "schema_version": "af3h_hallucination_summary_v1",
        "status": "completed",
        "package_version": __version__,
        "config_sha256": config.sha256,
        "input_sha256": sha256_file(input_path),
        "input_basename": input_path.name,
        "af3_source_commit_audited": "b2f3d45fbfcacc5183bd5345d15df93571b8437f",
        "af3_parameter_identifier": engine.parameter_identifier,
        "af3_runtime": __import__(
            "af3_hallucination.af3.provenance",
            fromlist=["runtime_source_fingerprint"],
        ).runtime_source_fingerprint(),
        "seed": config.hallucination.seed,
        "bucket": config.hallucination.bucket,
        "design_recycles": config.hallucination.design_recycles,
        "planned_steps": len(plan),
        "completed_steps": len(trajectory),
        "checkpoint_count": len(checkpoint_records),
        "final_loss": trajectory[-1]["loss"] if trajectory else None,
        "final_sequence_sha256": __import__("hashlib").sha256(final_sequence.encode()).hexdigest(),
        "final_logits_sha256": sha256_file(final_logits),
        "runtime_seconds": round(time.time() - started, 6),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "physical_gpu": os.environ.get("AF3_PHYSICAL_GPU", "unknown"),
            "gpus": _gpu_inventory(),
        },
    }
    _write_json(root / "summary.json", summary)
    return summary
