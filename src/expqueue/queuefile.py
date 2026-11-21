"""Reading and validating the queue file (a project's ``queue.toml``).

The queue file is the only source of commands: the runner never takes a command from the
command line, so what runs is what was reviewed and committed. Layout (all keys documented in
``README.md``)::

    [queue]
    root = "."                       # repository root, relative to this file
    programs = ["python", "uv"]      # allowed argv[0] values
    kill_grace = "30s"               # SIGTERM -> SIGKILL delay on timeout or stop
    provenance = ["uv.lock"]         # files whose sha256 every job.json records

    [defaults]                       # inherited by every job, overridable per job
    timeout = "1h"
    [defaults.env]
    OMP_NUM_THREADS = "1"

    [[job]]
    id = "train-a-60"
    cwd = "."                        # relative to root, must stay inside it
    argv = ["uv", "run", "python", "scripts/train.py", "--out", "{job_dir}"]
    env = { }                        # merged over defaults.env
    outputs = ["{job_dir}/summary.csv"]
    timeout = "30m"
    estimate = "8m"                  # optional, for status / list
    memory = "2G"                    # optional peak-memory estimate (also in [defaults])
    after = ["other-job"]            # optional: start only once these earlier jobs are done

``argv`` must be a list of strings: a single string (a shell command line) is rejected, and
the runner executes the list directly (``subprocess.Popen(list)``, never ``shell=True``), so
quotes, ``;``, ``|``, ``$()`` and globs inside arguments reach the program literally.

The placeholders ``{job_dir}``, ``{repo}``, ``{results_dir}`` and ``{data_dir}``
(``$EXPQ_DATA_DIR``, required as soon as a selected job or ``provenance`` uses it; there is
no default) are replaced in argv, env values, outputs and provenance by plain string
substitution; any other ``{name}`` is an error (a typo would otherwise reach the program).
Every validation failure raises :class:`QueueError` with the job id and the offending key.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLACEHOLDERS = ("job_dir", "repo", "results_dir", "data_dir")
# Placeholders whose value is a required environment variable of the runner (no default).
PLACEHOLDER_ENV = {"data_dir": "EXPQ_DATA_DIR"}
# "{{" and "}}" are escapes for literal braces; "{name}" is a placeholder.
_PLACEHOLDER_RE = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([smh])$")
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMG])$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_QUEUE_KEYS = {"root", "programs", "kill_grace", "provenance"}
_DEFAULT_KEYS = {"timeout", "env", "memory"}
_JOB_KEYS = {"id", "cwd", "argv", "env", "outputs", "timeout", "estimate", "memory", "after"}


class QueueError(ValueError):
    """The queue file is missing, not valid TOML, or breaks a rule of the format."""


@dataclass(frozen=True)
class Job:
    """One job of the queue, as written in the file (placeholders not yet expanded).

    ``spec_hash`` identifies the job's definition (cwd, argv, env, outputs, timeout), so the
    status can flag a job whose definition changed after it ran. ``memory_bytes`` (the peak
    memory estimate) and ``after`` (ids of earlier jobs that must be ``done`` first) only
    schedule the job, so they are not part of the hash: adding them to a job that already ran
    does not make it look changed.
    """

    id: str
    cwd: str
    argv: tuple[str, ...]
    env: dict[str, str]
    outputs: tuple[str, ...]
    timeout_s: float
    estimate_s: float | None
    spec_hash: str
    memory_bytes: int | None = None
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Queue:
    """A validated queue file: repository root, allowed programs and ordered jobs."""

    path: Path
    root: Path
    programs: tuple[str, ...]
    kill_grace_s: float
    jobs: tuple[Job, ...] = field(default_factory=tuple)
    provenance: tuple[str, ...] = ()

    def job(self, job_id: str) -> Job:
        """Return the job with this id; raise :class:`QueueError` if there is none."""
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise QueueError(f"no job with id {job_id!r} in {self.path}")


def parse_duration(value: Any, where: str) -> float:
    """Seconds from ``"90s"``, ``"45m"``, ``"2h"`` or a positive number of seconds.

    Raises :class:`QueueError` (naming ``where``) for anything else, including zero.
    """
    if isinstance(value, bool):
        raise QueueError(f"{where}: expected a duration, got {value!r}")
    if isinstance(value, int | float):
        seconds = float(value)
    elif isinstance(value, str) and (m := _DURATION_RE.match(value.strip())):
        seconds = float(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)]
    else:
        raise QueueError(f"{where}: expected a duration like '90s', '45m', '2h', got {value!r}")
    if seconds <= 0:
        raise QueueError(f"{where}: duration must be positive, got {value!r}")
    return seconds


def parse_size(value: Any, where: str) -> int:
    """Bytes from ``"512M"``, ``"6G"`` or ``"800K"`` (binary units: 1G = 2**30 bytes).

    Raises :class:`QueueError` (naming ``where``) for anything else, including zero.
    """
    if not isinstance(value, str) or not (m := _SIZE_RE.match(value.strip())):
        raise QueueError(f"{where}: expected a size like '512M' or '6G', got {value!r}")
    size = int(float(m.group(1)) * {"K": 2**10, "M": 2**20, "G": 2**30}[m.group(2)])
    if size <= 0:
        raise QueueError(f"{where}: size must be positive, got {value!r}")
    return size


def expand(text: str, values: dict[str, str]) -> str:
    """Replace the placeholders of :data:`PLACEHOLDERS` (``{job_dir}``, ...) in ``text``.

    Plain substitution, no shell or format-spec semantics; ``{{`` and ``}}`` become literal
    braces, any other brace is left as is. Raises :class:`QueueError` for a
    placeholder that is not one of :data:`PLACEHOLDERS`, or that has no entry in ``values``
    (``{data_dir}`` when ``$EXPQ_DATA_DIR`` is not set).
    """

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name is None:  # an escaped brace
            return m.group(0)[0]
        if name not in PLACEHOLDERS:
            raise QueueError(f"unknown placeholder {{{name}}} in {text!r}")
        if name not in values:
            raise QueueError(
                f"placeholder {{{name}}} in {text!r} has no value: set ${PLACEHOLDER_ENV[name]}"
            )
        return values[name]

    return _PLACEHOLDER_RE.sub(sub, text)


def _check_placeholders(text: str, where: str) -> None:
    for name in _PLACEHOLDER_RE.findall(text):
        if name and name not in PLACEHOLDERS:
            raise QueueError(f"{where}: unknown placeholder {{{name}}} (allowed: {PLACEHOLDERS})")


def uses_placeholder(texts: Iterable[str], name: str) -> bool:
    """True if any of ``texts`` contains the placeholder ``{name}`` (escaped braces excluded)."""
    return any(name in _PLACEHOLDER_RE.findall(text) for text in texts)


def _str_list(value: Any, where: str, *, non_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str):
        raise QueueError(
            f"{where}: must be a list of strings, not a single string "
            "(commands are argv lists, never shell command lines)"
        )
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise QueueError(f"{where}: must be a list of strings, got {value!r}")
    if non_empty and not value:
        raise QueueError(f"{where}: must not be empty")
    return tuple(value)


def _env_table(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise QueueError(f'{where}: must be a table of NAME = "value" strings')
    env: dict[str, str] = {}
    for name, val in value.items():
        if not _ENV_NAME_RE.match(name):
            raise QueueError(f"{where}: invalid variable name {name!r}")
        if not isinstance(val, str):
            raise QueueError(f"{where}.{name}: value must be a string, got {val!r}")
        _check_placeholders(val, f"{where}.{name}")
        env[name] = val
    return env


def _unknown_keys(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise QueueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _spec_hash(spec: dict[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _scheduling(
    raw: dict[str, Any], defaults: dict[str, Any], where: str
) -> tuple[int | None, tuple[str, ...]]:
    """The job's ``memory`` estimate (own or default, bytes) and its ``after`` ids."""
    if "memory" in raw:
        memory_bytes: int | None = parse_size(raw["memory"], f"{where}.memory")
    else:
        memory_bytes = defaults["memory"]
    return memory_bytes, _str_list(raw.get("after", []), f"{where}.after", non_empty=False)


def _parse_job(
    raw: Any, index: int, root: Path, programs: tuple[str, ...], defaults: dict[str, Any]
) -> Job:
    where = f"job #{index + 1}"
    if not isinstance(raw, dict):
        raise QueueError(f"{where}: must be a table")
    job_id = raw.get("id")
    if not isinstance(job_id, str) or not _ID_RE.match(job_id):
        raise QueueError(
            f"{where}: id must match {_ID_RE.pattern} (it names the results directory), "
            f"got {job_id!r}"
        )
    where = f"job {job_id!r}"
    _unknown_keys(raw, _JOB_KEYS, where)

    cwd = raw.get("cwd", ".")
    if not isinstance(cwd, str) or Path(cwd).is_absolute():
        raise QueueError(f"{where}: cwd must be a path relative to the repository root")
    if not _inside((root / cwd).resolve(), root):
        raise QueueError(f"{where}: cwd {cwd!r} leaves the repository root {root}")

    if "argv" not in raw:
        raise QueueError(f"{where}: argv is required")
    argv = _str_list(raw["argv"], f"{where}.argv", non_empty=True)
    if argv[0] not in programs:
        raise QueueError(
            f"{where}: program {argv[0]!r} is not in [queue].programs {list(programs)}"
        )
    for arg in argv:
        _check_placeholders(arg, f"{where}.argv")

    env = dict(defaults["env"])
    env.update(_env_table(raw.get("env", {}), f"{where}.env"))
    outputs = _str_list(raw.get("outputs", []), f"{where}.outputs", non_empty=False)
    for out in outputs:
        _check_placeholders(out, f"{where}.outputs")

    if "timeout" in raw:
        timeout_s = parse_duration(raw["timeout"], f"{where}.timeout")
    elif defaults["timeout"] is not None:
        timeout_s = defaults["timeout"]
    else:
        raise QueueError(f"{where}: no timeout (set it on the job or in [defaults])")
    estimate_s = parse_duration(raw["estimate"], f"{where}.estimate") if "estimate" in raw else None
    memory_bytes, after = _scheduling(raw, defaults, where)
    spec = {
        "cwd": cwd,
        "argv": list(argv),
        "env": env,
        "outputs": list(outputs),
        "timeout_s": timeout_s,
    }
    return Job(
        job_id,
        cwd,
        argv,
        env,
        outputs,
        timeout_s,
        estimate_s,
        _spec_hash(spec),
        memory_bytes,
        after,
    )


def load_queue(path: Path) -> Queue:
    """Read and validate the queue file at ``path``.

    Checks: TOML syntax; no unknown keys (typos fail loudly); ``root`` is an existing
    directory; job ids unique and usable as directory names; ``cwd`` relative and inside the
    root (after resolving ``..`` and symlinks); ``argv`` a non-empty list of strings whose
    first element is listed in ``[queue].programs``; env names valid and values strings;
    only known placeholders; every job has a positive timeout; ``memory`` sizes valid;
    every ``after`` id names a job earlier in the file (so dependencies follow the file order
    and can never form a cycle) and not the job itself. Raises :class:`QueueError`
    on the first problem found. Whether ``cwd`` exists is checked when the job starts.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise QueueError(f"queue file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise QueueError(f"{path}: invalid TOML: {exc}") from exc
    _unknown_keys(data, {"queue", "defaults", "job"}, str(path))

    qtable = data.get("queue", {})
    _unknown_keys(qtable, _QUEUE_KEYS, "[queue]")
    root_raw = qtable.get("root", ".")
    if not isinstance(root_raw, str):
        raise QueueError("[queue].root must be a string")
    root = (path.parent / root_raw).resolve()
    if not root.is_dir():
        raise QueueError(f"[queue].root {root} is not a directory")
    programs = _str_list(qtable.get("programs", []), "[queue].programs", non_empty=True)
    kill_grace_s = parse_duration(qtable.get("kill_grace", "30s"), "[queue].kill_grace")
    provenance = _str_list(qtable.get("provenance", []), "[queue].provenance", non_empty=False)
    for item in provenance:
        _check_placeholders(item, "[queue].provenance")

    dtable = data.get("defaults", {})
    _unknown_keys(dtable, _DEFAULT_KEYS, "[defaults]")
    defaults: dict[str, Any] = {
        "env": _env_table(dtable.get("env", {}), "[defaults].env"),
        "timeout": parse_duration(dtable["timeout"], "[defaults].timeout")
        if "timeout" in dtable
        else None,
        "memory": parse_size(dtable["memory"], "[defaults].memory") if "memory" in dtable else None,
    }

    raw_jobs = data.get("job", [])
    if not isinstance(raw_jobs, list):
        raise QueueError("jobs must be written as [[job]] tables")
    jobs = tuple(_parse_job(raw, i, root, programs, defaults) for i, raw in enumerate(raw_jobs))
    seen: set[str] = set()
    for job in jobs:
        if job.id in seen:
            raise QueueError(f"duplicate job id {job.id!r}")
        for dep in job.after:
            if dep not in seen:
                raise QueueError(
                    f"job {job.id!r}: after {dep!r} must name a job earlier in the file"
                )
        seen.add(job.id)
    return Queue(path.resolve(), root, programs, kill_grace_s, jobs, provenance)
