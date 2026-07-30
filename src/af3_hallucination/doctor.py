"""Environment diagnostics without importing optional stacks prematurely."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from typing import Any


def environment_report() -> dict[str, Any]:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "yaml",
            "numpy",
            "jax",
            "haiku",
            "optax",
            "alphafold3",
            "run_alphafold",
        )
    }
    distributions = {
        "yaml": "PyYAML",
        "numpy": "numpy",
        "jax": "jax",
        "haiku": "dm-haiku",
        "optax": "optax",
        "alphafold3": "alphafold3",
    }
    versions = {}
    for module, distribution in distributions.items():
        if not modules[module]:
            versions[module] = None
            continue
        try:
            versions[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[module] = "unknown"
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "modules": modules,
        "versions": versions,
        "commands": {
            name: shutil.which(name)
            for name in ("ssh", "nvidia-smi", "apptainer", "singularity")
        },
        "capabilities": {
            "core": modules["yaml"],
            "jax": modules["jax"] and modules["haiku"] and modules["optax"],
            "af3": modules["alphafold3"] and modules["run_alphafold"],
            "local_af3_gpu_expected": platform.system() == "Linux" and bool(shutil.which("nvidia-smi")),
            "ssh_remote": bool(shutil.which("ssh")),
        },
    }
