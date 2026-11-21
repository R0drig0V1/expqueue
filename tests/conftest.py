"""Shared helpers of the offline test suite.

A throwaway repository, results directory and queue file whose jobs are small
``python -c`` scripts (the interpreter running the tests is the only allowed program), so
every test runs in well under a few seconds.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from expqueue.state import Paths

PY = sys.executable


def toml_value(value: Any) -> str:
    """A TOML literal for str / number / list / dict values (JSON strings are valid TOML)."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def py_job(job_id: str, code: str, *args: str, **extra: Any) -> dict[str, Any]:
    """A job that runs ``python -c code args...``."""
    return {"id": job_id, "argv": [PY, "-c", code, *args], **extra}


@dataclass
class Sandbox:
    """A fake repository (``root``), a results directory outside it and the queue file."""

    root: Path
    results: Path
    queue: Path

    @property
    def paths(self) -> Paths:
        """The results-directory layout."""
        return Paths(self.results)

    def write(
        self,
        jobs: list[dict[str, Any]],
        *,
        timeout: str = "20s",
        kill_grace: str = "1s",
        env: dict[str, str] | None = None,
        programs: list[str] | None = None,
        provenance: list[str] | None = None,
    ) -> Path:
        """Write ``queue.toml`` with these jobs; return its path."""
        lines = [
            "[queue]",
            'root = "."',
            f"programs = {toml_value(programs or [PY])}",
            f'kill_grace = "{kill_grace}"',
            f"provenance = {toml_value(provenance or [])}",
            "",
            "[defaults]",
            f'timeout = "{timeout}"',
        ]
        if env:
            lines += ["[defaults.env]", *(f"{k} = {toml_value(v)}" for k, v in env.items())]
        for job in jobs:
            lines += ["", "[[job]]", *(f"{k} = {toml_value(v)}" for k, v in job.items())]
        self.queue.write_text("\n".join(lines) + "\n")
        return self.queue

    def args(self, *command: str) -> list[str]:
        """``expq`` arguments pointing at this sandbox (fast polling)."""
        base = ["--queue", str(self.queue), "--results-dir", str(self.results)]
        if command and command[0] == "run":
            return [*base, *command, "--poll", "0.05"]
        return [*base, *command]

    def job_meta(self, job_id: str) -> dict[str, Any]:
        """The ``job.json`` of a job."""
        data: dict[str, Any] = json.loads((self.results / job_id / "job.json").read_text())
        return data


@pytest.fixture(autouse=True)
def no_ambient_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite never reads the developer's own ``EXPQ_*`` settings."""
    monkeypatch.delenv("EXPQ_RESULTS_DIR", raising=False)
    monkeypatch.delenv("EXPQ_DATA_DIR", raising=False)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    root = tmp_path / "repo"
    root.mkdir()
    return Sandbox(root, tmp_path / "results", root / "queue.toml")


@pytest.fixture
def wait_for() -> Callable[..., None]:
    """Poll ``condition`` until it is true or fail after ``timeout`` seconds."""
    import time

    def wait(condition: Callable[[], bool], timeout: float = 10.0, what: str = "") -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                pytest.fail(f"timed out waiting for {what or condition}")
            time.sleep(0.02)

    return wait
