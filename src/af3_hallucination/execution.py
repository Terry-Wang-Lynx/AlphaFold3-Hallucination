"""Portable local and SSH command executors."""

from __future__ import annotations

import dataclasses
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class CompletedExecution:
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


class LocalExecutor:
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompletedExecution:
        process_env = os.environ.copy()
        process_env.update(dict(env or {}))
        result = subprocess.run(
            list(command),
            cwd=None if cwd is None else str(cwd),
            env=process_env,
            text=True,
            capture_output=True,
            check=False,
        )
        return CompletedExecution(tuple(command), result.returncode, result.stdout, result.stderr)


class SSHExecutor:
    """Minimal generic SSH executor for macOS-to-Linux submission."""

    def __init__(self, host: str, *, ssh_binary: str = "ssh") -> None:
        if not host or host.startswith("-") or any(char.isspace() for char in host):
            raise ValueError("SSH host must be a non-empty token")
        self.host = host
        self.ssh_binary = ssh_binary

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CompletedExecution:
        parts = []
        if cwd is not None:
            parts.extend(["cd", shlex.quote(str(cwd)), "&&"])
        if env:
            parts.append("env")
            for key, value in env.items():
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
                    raise ValueError(f"invalid environment variable name: {key!r}")
                parts.append(f"{key}={shlex.quote(value)}")
        parts.extend(shlex.quote(str(value)) for value in command)
        remote = " ".join(parts)
        full = (self.ssh_binary, self.host, remote)
        result = subprocess.run(
            full,
            text=True,
            capture_output=True,
            check=False,
        )
        return CompletedExecution(full, result.returncode, result.stdout, result.stderr)
