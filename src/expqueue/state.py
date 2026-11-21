"""Results layout, the state file and the locks of the job queue.

Everything lives outside git, under the results directory (``--results-dir`` or
``$EXPQ_RESULTS_DIR``; required, no built-in default, one per project)::

    <results>/_queue/state.json    per-job status (atomic replace, readable at any time)
    <results>/_queue/state.lock    flock held only during a read-modify-write of the state
    <results>/_queue/runner.lock   flock held by the one running runner; contains its pid
    <results>/_queue/STOP          drain request: finish the running jobs, start no new one
    <results>/_queue/runner.log    the runner's progress lines
    <results>/<job id>/            logs, job.json and the job's own outputs

Both locks are ``flock`` locks: the kernel drops them when the holder dies, so a crash or a
reboot never leaves a stale lock behind. The state file is the source of truth for
resuming: ``done`` and ``skipped`` jobs are not run again; a ``running`` entry found by a
new runner means the previous runner died mid-job, and the job is rerun.

Job statuses: ``pending`` (also: no entry yet), ``running``, ``done``, ``failed``,
``timeout``, ``interrupted``, ``skipped``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from expqueue.durable import sync_fd, sync_path

STATE_VERSION = 1
RUNNABLE = frozenset({"pending", "interrupted"})
FINISHED_BAD = frozenset({"failed", "timeout"})
STATUSES = ("pending", "running", "done", "failed", "timeout", "interrupted", "skipped")

JobRecord = dict[str, Any]
State = dict[str, Any]


# Environment variable naming the results directory when ``--results-dir`` is not given.
RESULTS_DIR_ENV = "EXPQ_RESULTS_DIR"


def results_dir_from_env() -> Path | None:
    """``$EXPQ_RESULTS_DIR`` if set and non-empty, else None.

    There is no fallback path on purpose: the results directory holds ``runner.lock``, so a
    default shared by every project would make them one runner.
    """
    env = os.environ.get(RESULTS_DIR_ENV)
    return Path(env).expanduser() if env else None


@dataclass(frozen=True)
class Paths:
    """Locations of the queue's files under one results directory."""

    results_dir: Path

    @property
    def queue_dir(self) -> Path:
        """``<results>/_queue``: state, locks, stop file and runner log."""
        return self.results_dir / "_queue"

    @property
    def state_file(self) -> Path:
        """The JSON state file."""
        return self.queue_dir / "state.json"

    @property
    def state_lock(self) -> Path:
        """Lock serializing read-modify-write cycles of the state file."""
        return self.queue_dir / "state.lock"

    @property
    def runner_lock(self) -> Path:
        """Lock held by the running runner for its whole lifetime."""
        return self.queue_dir / "runner.lock"

    @property
    def stop_file(self) -> Path:
        """Drain request file created by ``expq stop``."""
        return self.queue_dir / "STOP"

    @property
    def runner_log(self) -> Path:
        """Append-only log of the runner's progress lines."""
        return self.queue_dir / "runner.log"

    def job_dir(self, job_id: str) -> Path:
        """``<results>/<job id>``; job ids cannot start with ``_``, so no clash with _queue."""
        return self.results_dir / job_id


def empty_state() -> State:
    """A state with no job records."""
    return {"version": STATE_VERSION, "jobs": {}}


def read_state(paths: Paths) -> State:
    """Read the state file; an absent file is an empty state.

    Safe without the lock because writers replace the file atomically. Raises ``ValueError``
    if the file is not a state written by this package (corrupt or another version).
    """
    try:
        data = json.loads(paths.state_file.read_text())
    except FileNotFoundError:
        return empty_state()
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt state file {paths.state_file}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise ValueError(f"{paths.state_file} is not a version {STATE_VERSION} queue state")
    return data


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to ``path`` via a temporary file, fsync, ``os.replace`` and a directory fsync.

    A reader (or a crash) sees either the old or the new file, never half of one; the
    directory fsync makes the rename itself durable, so after a crash the new content is not
    silently replaced by the old one. On macOS, where ``fsync`` leaves data in the drive's
    cache, ``F_FULLFSYNC`` is used (:func:`expqueue.durable.sync_fd`).

    Raises:
        OSError: the file could not be written or flushed.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        sync_fd(f.fileno())
    os.replace(tmp, path)
    sync_path(path.parent)


@contextlib.contextmanager
def _flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def update_state(paths: Paths, change: Callable[[State], None]) -> State:
    """Apply ``change`` to the state under the state lock and write it back atomically.

    ``change`` mutates the state in place. Used by the runner and by ``retry`` / ``skip``,
    which may run while a runner is active. Returns the new state.
    """
    with _flock(paths.state_lock):
        state = read_state(paths)
        change(state)
        write_json_atomic(paths.state_file, state)
        return state


def job_status(state: State, job_id: str) -> str:
    """Status of a job in ``state``; ``pending`` when it has no record."""
    record = state["jobs"].get(job_id)
    return str(record["status"]) if record else "pending"


class LockedError(RuntimeError):
    """Another runner holds the runner lock."""

    def __init__(self, pid: int | None) -> None:
        super().__init__(f"another runner is active (pid {pid if pid else 'unknown'})")
        self.pid = pid


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    return int(text) if text.isdigit() else None


class RunnerLock:
    """Exclusive, non-blocking lock that makes the runner a singleton per results directory.

    ``acquire`` raises :class:`LockedError` if another process holds it. The lock file holds
    the owner's pid for ``status`` and ``stop --now``; the lock itself is the ``flock``, so
    a dead owner's pid left in the file does not block anyone.
    """

    def __init__(self, paths: Paths) -> None:
        self._path = paths.runner_lock
        self._file: TextIO | None = None

    def acquire(self) -> None:
        """Take the lock or raise :class:`LockedError`."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        f = self._path.open("a+")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise LockedError(_read_pid(self._path)) from None
        f.seek(0)
        f.truncate()
        f.write(f"{os.getpid()}\n")
        f.flush()
        self._file = f

    def release(self) -> None:
        """Release the lock (idempotent). The pid is cleared so it is not mistaken for a runner."""
        if self._file is not None:
            self._file.truncate(0)
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None

    def __enter__(self) -> RunnerLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def active_runner_pid(paths: Paths) -> int | None:
    """Pid of the active runner, or None if no runner holds the lock.

    Probes the lock without keeping it. If a runner holds it but its pid is unreadable,
    returns -1 (a runner is active, pid unknown).
    """
    if not paths.runner_lock.exists():
        return None
    with paths.runner_lock.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = _read_pid(paths.runner_lock)
            return pid if pid else -1
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return None
