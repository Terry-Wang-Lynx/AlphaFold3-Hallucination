"""Runtime fingerprints for reproducible AF3 executions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from typing import Any


def parameter_identifier(parameters) -> str:
    raw = parameters["__meta__"]["__identifier__"].tobytes()
    try:
        return raw.decode().rstrip("\x00")
    except UnicodeDecodeError:
        return hashlib.sha256(raw).hexdigest()


def result_parameter_identifier(result) -> str:
    raw = result["__identifier__"]
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    if isinstance(raw, str):
        return raw
    try:
        return bytes(raw).decode().rstrip("\x00")
    except UnicodeDecodeError:
        return hashlib.sha256(bytes(raw)).hexdigest()


def runtime_source_fingerprint() -> dict[str, Any]:
    """Hash the model modules actually imported in the current AF3 process."""

    import run_alphafold
    from alphafold3.model import model
    from alphafold3.model.network import confidence_head, diffusion_head

    modules = {
        "run_alphafold": run_alphafold,
        "alphafold3.model.model": model,
        "alphafold3.model.network.confidence_head": confidence_head,
        "alphafold3.model.network.diffusion_head": diffusion_head,
    }
    hashes = {}
    for name, module in modules.items():
        path = inspect.getsourcefile(module)
        if path is None:
            hashes[name] = None
        else:
            with open(path, "rb") as handle:
                hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    try:
        version = importlib.metadata.version("alphafold3")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"package_version": version, "module_sha256": hashes}
