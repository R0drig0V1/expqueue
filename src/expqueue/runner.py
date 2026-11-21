"""The queue runner: start jobs in file order, at most N at a time, at low priority.

One ``run_queue`` call is one runner. It takes the runner lock (one runner per results
directory), recovers jobs left ``running`` by a dead runner (their orphaned process group is
killed when it provably belongs to this boot, then they are rerun), and loops:

- reap finished jobs (``os.wait4``: exit status and peak RSS) and record them;
- enforce timeouts: SIGTERM to the job's process group, SIGKILL after ``kill_grace``;
- on SIGINT / SIGTERM / SIGHUP: SIGTERM every running job, mark them ``interrupted`` (they
  are rerun by the next runner) and exit 130; a second signal SIGKILLs at once;
- if the STOP file exists: start nothing new, exit 4 when the running jobs finish;
- otherwise start the next runnable job (``pending`` or ``interrupted``, file order) whose
  ``after`` jobs are all ``done`` and whose memory estimate fits.

Capacity cap: the queue never uses more than half of the machine.
``--jobs`` is refused above half the CPUs the process may use, and a job is not started while
the summed ``memory`` estimates of the running jobs plus its own would exceed half the
physical memory (a job whose estimate alone exceeds it fails without starting). The cap counts
only the queue's own jobs, so a machine loaded by someone else still runs one job at nice 19.

Each job runs as ``Popen(argv_list)`` (never through a shell) in its own session (process
group), with ``nice`` applied in the child before ``exec``, stdin from /dev/null, and stdout
/ stderr to ``<results>/<id>/stdout.log`` / ``stderr.log``. The process group is killed when
the job ends, so nothing a job left behind keeps competing with the next one. The queue file
is reloaded before each start, so jobs can be appended while the runner works; if the file
is invalid at that moment the runner keeps the last valid version and logs a warning.

A job is ``done`` only if it exits 0, every expected output exists and was written
during this attempt, and its files are durable: the whole job directory and the declared
outputs are flushed to disk, with their directories, *before* ``done`` reaches the state file
(:func:`expqueue.durable.sync_job`); exit 0 with missing or stale outputs, or a failed flush,
is ``failed``. The size and sha256 of each flushed file go to ``job.json`` (``files``); before a
job starts, the files of its ``after`` jobs are compared with that manifest, and a dependency
whose files changed or vanished since (a host crash that lost them anyway, a file truncated by
hand) is turned ``failed`` so its dependents are blocked instead of reading broken inputs.
Before an attempt, whatever an earlier attempt left in the job directory is moved to
``attempts/<n>/``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import IO, Any

from expqueue.durable import sync_job, verify_manifest
from expqueue.queuefile import (
    PLACEHOLDER_ENV,
    Job,
    Queue,
    QueueError,
    expand,
    load_queue,
    uses_placeholder,
)
from expqueue.state import (
    FINISHED_BAD,
    RUNNABLE,
    JobRecord,
    Paths,
    RunnerLock,
    State,
    job_status,
    read_state,
    update_state,
    write_json_atomic,
)

# Exit codes of ``expq run`` (also documented in README.md).
EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_USAGE = 2
EXIT_LOCKED = 3
EXIT_STOPPED = 4
EXIT_BLOCKED = 5  # nothing failed, but selected jobs were left waiting for their 'after' jobs
EXIT_INTERRUPTED = 130

DEFAULT_NICE = 19
# Environment variable behind the ``{data_dir}`` placeholder (required when it is used).
DATA_DIR_ENV = PLACEHOLDER_ENV["data_dir"]
# Share of the machine the queue may use: never more than 50%.
CAPACITY_SHARE = 0.5


def cpu_count() -> int:
    """CPUs this process may run on (the affinity mask where the OS has one), at least 1."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def physical_memory() -> int | None:
    """Physical memory in bytes (``sysconf``), or None where it cannot be read."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def max_parallel_jobs() -> int:
    """The largest ``--jobs`` allowed: ``CAPACITY_SHARE`` of :func:`cpu_count`, at least 1."""
    return max(1, int(cpu_count() * CAPACITY_SHARE))


def memory_budget() -> int | None:
    """Bytes of summed job memory estimates allowed at once (None: memory unknown, no cap)."""
    total = physical_memory()
    return None if total is None else int(total * CAPACITY_SHARE)


def now_iso() -> str:
    """Local time with UTC offset, to the second."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float | None) -> str:
    """``1h02m03s`` / ``4m05s`` / ``12s``; ``-`` for None."""
    if seconds is None:
        return "-"
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s" if m else f"{s}s"


def boot_id() -> str | None:
    """An identifier of the current boot, or None if it cannot be read.

    Linux: ``/proc/sys/kernel/random/boot_id``; macOS: the boot time from
    ``sysctl kern.boottime``. Used to tell whether a pid recorded in the state can still
    belong to the job (same boot) or may have been reused after a reboot.
    """
    linux = Path("/proc/sys/kernel/random/boot_id")
    with contextlib.suppress(OSError):
        return linux.read_text().strip()
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.boottime"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        m = re.search(r"sec = (\d+)", out)
        return f"darwin-{m.group(1)}" if m else None
    return None


def peak_rss_bytes(rusage: Any) -> int | None:
    """``ru_maxrss`` in bytes (macOS reports bytes, Linux KiB); None if not positive.

    From ``wait4`` it is the largest resident set of the job's process or of any
    descendant it waited for, not the sum over processes.
    """
    value = int(getattr(rusage, "ru_maxrss", 0))
    if value <= 0:
        return None
    return value if sys.platform == "darwin" else value * 1024


def git_info(root: Path) -> tuple[str | None, bool | None]:
    """``(HEAD commit, working tree dirty)`` of ``root``; ``(None, None)`` outside git."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(dirty)


def data_dir_from_env() -> Path | None:
    """``$EXPQ_DATA_DIR`` if set and non-empty, else None (``{data_dir}`` then has no value)."""
    env = os.environ.get(DATA_DIR_ENV)
    return Path(env).expanduser() if env else None


def _job_texts(job: Job) -> list[str]:
    """Every string of ``job`` in which placeholders are expanded."""
    return [*job.argv, *job.env.values(), *job.outputs]


def file_hashes(paths: Sequence[str], root: Path) -> dict[str, str | None]:
    """sha256 of each file (relative paths are relative to ``root``); None if unreadable.

    Recorded in job.json so a result can be traced to the exact lock files and data
    manifest it was produced with.
    """
    hashes: dict[str, str | None] = {}
    for item in paths:
        path = root / item
        try:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            hashes[item] = digest.hexdigest()
        except OSError:
            hashes[item] = None
    return hashes


def base_env() -> dict[str, str]:
    """The runner's environment minus its own virtual environment.

    ``uv run expq`` puts the runner's ``.venv`` in ``VIRTUAL_ENV`` and first on ``PATH``;
    jobs must not inherit that (``uv run`` in another project would warn, ``python`` would
    resolve to the runner's interpreter), so both are removed.
    """
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    if venv:
        venv_bin = str(Path(venv) / "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if p and p != venv_bin
        )
    return env


def _set_priority(nice: int) -> Callable[[], None]:
    def apply() -> None:
        # Runs in the child between fork and exec. Never lower the priority an
        # already-niced runner gives (that would fail without root anyway).
        current = os.getpriority(os.PRIO_PROCESS, 0)
        os.setpriority(os.PRIO_PROCESS, 0, max(current, nice))

    return apply


def _fresh(path: Path, since: float) -> bool:
    """True if ``path`` exists and was modified after ``since`` (2 s of clock slack)."""
    try:
        return path.stat().st_mtime >= since - 2
    except OSError:
        return False


def _killpg(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_epoch(pid: int) -> float | None:
    """Start time (epoch seconds, 1 s resolution) of process ``pid`` from ``ps``, or None."""
    try:
        out = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.fullmatch(r"(?:(?:(\d+)-)?(\d+):)?(\d+):(\d+)", out)
    if not m:
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return time.time() - (((days * 24 + hours) * 60 + minutes) * 60 + seconds)


def kill_orphan(record: JobRecord, current_boot: str | None, grace_s: float) -> bool:
    """Kill the process group a dead runner left behind for ``record``; True if it did.

    A recorded pid may have been reused, so the group is killed only when all of these hold:
    the record's boot id equals ``current_boot`` (no reboot since), the pid is alive and
    leads its own process group (jobs start as session leaders, pid == pgid), and the
    process started within 60 s of the recorded start. SIGTERM, then SIGKILL after
    ``grace_s``. If the job's leader process already exited, nothing is killed.
    """
    pid = record.get("pid")
    started = record.get("started_epoch")
    if not isinstance(pid, int) or not isinstance(started, int | float):
        return False
    if current_boot is None or record.get("boot_id") != current_boot:
        return False
    try:
        if os.getpgid(pid) != pid:
            return False
    except ProcessLookupError:
        return False
    actual = process_start_epoch(pid)
    if actual is None or abs(actual - started) > 60:
        return False
    _killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while _group_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_alive(pid):
        _killpg(pid, signal.SIGKILL)
    return True


@dataclass
class _Running:
    job: Job
    proc: subprocess.Popen[bytes]
    files: list[IO[bytes]]
    meta: dict[str, Any]
    job_dir: Path
    started_mono: float
    wall_start: float
    deadline: float
    stop_reason: str | None = None  # "timeout" | "interrupted"
    term_at: float | None = None
    killed: bool = False


@dataclass
class Runner:
    """State of one runner invocation; use :func:`run_queue` rather than this class."""

    queue_path: Path
    queue: Queue
    paths: Paths
    max_jobs: int = 1
    nice: int = DEFAULT_NICE
    only: Sequence[str] = ()
    poll_s: float = 0.1
    out: IO[str] = field(default_factory=lambda: sys.stdout)
    running: dict[str, _Running] = field(default_factory=dict)
    signals: int = 0
    bad_this_run: int = 0
    boot: str | None = None
    reload_error: str | None = None
    memory_cap: int | None = None

    def log(self, message: str) -> None:
        """Print a timestamped progress line and append it to ``runner.log``."""
        line = f"[{now_iso()}] {message}"
        with contextlib.suppress(OSError):  # the terminal may be gone (pane closed: SIGHUP)
            print(line, file=self.out, flush=True)
        with contextlib.suppress(OSError), self.paths.runner_log.open("a") as f:
            f.write(line + "\n")

    # --- queue and selection -------------------------------------------------------------

    def reload(self) -> None:
        """Re-read the queue file, keeping the previous version if the new one is invalid."""
        try:
            self.queue = load_queue(self.queue_path)
            self.reload_error = None
        except QueueError as exc:
            if str(exc) != self.reload_error:  # warn once per distinct problem
                self.log(f"warning: queue file invalid, keeping the previous version: {exc}")
            self.reload_error = str(exc)

    def selected(self, job: Job) -> bool:
        """True if ``--only`` is empty or one of its glob patterns matches the job id."""
        return self.selected_id(job.id)

    def selected_id(self, job_id: str) -> bool:
        """True if ``--only`` is empty or one of its glob patterns matches ``job_id``."""
        return not self.only or any(fnmatch.fnmatchcase(job_id, p) for p in self.only)

    def candidates(self, state: State) -> list[Job]:
        """Selected jobs left to run (``pending`` / ``interrupted``, not running), file order."""
        return [
            j
            for j in self.queue.jobs
            if self.selected(j) and j.id not in self.running and job_status(state, j.id) in RUNNABLE
        ]

    def runnable(self, state: State) -> list[Job]:
        """Candidates whose ``after`` jobs are all ``done``: the jobs that may start now."""
        return [
            j
            for j in self.candidates(state)
            if all(job_status(state, d) == "done" for d in j.after)
        ]

    def blockers(self, state: State, job: Job) -> str:
        """The ``after`` jobs of ``job`` that are not ``done``, with their status, for the log.

        A dependency outside the selection (``--only``) is marked as such: this run will never
        start it, so the job waits until a run selects both.
        """
        parts = []
        for d in job.after:
            status = job_status(state, d)
            if status == "done":
                continue
            outside = "" if self.selected_id(d) else ", not selected"
            parts.append(f"{d} ({status}{outside})")
        return "waits for " + ", ".join(parts)

    def running_memory(self) -> int:
        """Summed memory estimates of the running jobs (a job without one counts 0)."""
        return sum(r.job.memory_bytes or 0 for r in self.running.values())

    def fits(self, job: Job) -> bool:
        """Whether ``job`` may start next to the running jobs under the memory cap."""
        if self.memory_cap is None or job.memory_bytes is None:
            return True
        return self.running_memory() + job.memory_bytes <= self.memory_cap

    # --- recovery ------------------------------------------------------------------------

    def recover(self) -> None:
        """Turn ``running`` records of a dead runner into ``interrupted`` (rerun later).

        Orphans are killed before the state is locked: waiting for them can take up to
        ``kill_grace`` and must not block ``retry`` / ``skip`` meanwhile.
        """
        stale = {
            job_id: record
            for job_id, record in read_state(self.paths)["jobs"].items()
            if record.get("status") == "running"
        }
        killed = {
            job_id
            for job_id, record in stale.items()
            if kill_orphan(record, self.boot, self.queue.kill_grace_s)
        }

        def change(state: State) -> None:
            for job_id in stale:
                record = state["jobs"].get(job_id)
                if record and record.get("status") == "running":
                    record["status"] = "interrupted"
                    record["reason"] = "the runner stopped while the job was running"
                    record["pid"] = None

        update_state(self.paths, change)
        for job_id in stale:
            extra = " (killed its orphaned process group)" if job_id in killed else ""
            self.log(f"recover: {job_id} was interrupted{extra}; it will be rerun")

    # --- starting --------------------------------------------------------------------------

    def _rotate(self, job_dir: Path) -> None:
        """Move everything a previous attempt left in ``job_dir`` to ``attempts/<n>/``.

        Each attempt starts with an empty job directory, so outputs of an earlier attempt can
        never make a new one look successful, and nothing of the earlier one is lost.
        """
        entries = [e for e in job_dir.iterdir() if e.name != "attempts"]
        if not entries:
            return
        try:
            attempt = str(int(json.loads((job_dir / "job.json").read_text())["attempt"]))
        except (ValueError, OSError, KeyError, TypeError):
            attempt = "unknown"
        dest = job_dir / "attempts" / attempt
        n = 0
        while dest.exists():
            n += 1
            dest = job_dir / "attempts" / f"{attempt}.{n}"
        dest.mkdir(parents=True)
        for entry in entries:
            entry.rename(dest / entry.name)

    def _record(self, job: Job, **fields: Any) -> None:
        self._record_id(job.id, **fields)

    def _record_id(self, job_id: str, **fields: Any) -> None:
        def change(state: State) -> None:
            record = state["jobs"].setdefault(job_id, {"attempts": 0})
            record.update(fields)

        update_state(self.paths, change)

    def _fail_to_start(self, job: Job, job_dir: Path, meta: dict[str, Any], reason: str) -> None:
        meta.update(status="failed", reason=reason, ended=now_iso(), exit_code=None)
        write_json_atomic(job_dir / "job.json", meta)
        self._record(
            job, status="failed", reason=reason, ended=meta["ended"], exit_code=None, pid=None
        )
        self.bad_this_run += 1
        self.log(f"FAILED {job.id}: {reason}")

    def start(self, job: Job) -> None:
        """Start ``job``: prepare its directory, record it, launch the process group."""
        root = self.queue.root
        job_dir = self.paths.job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._rotate(job_dir)
        values = {
            "job_dir": str(job_dir),
            "repo": str(root),
            "results_dir": str(self.paths.results_dir),
        }
        if (data_dir := data_dir_from_env()) is not None:
            values["data_dir"] = str(data_dir)
        try:
            argv = [expand(a, values) for a in job.argv]
            overrides = {k: expand(v, values) for k, v in job.env.items()}
            outputs = [expand(o, values) for o in job.outputs]
            provenance_paths = [expand(x, values) for x in self.queue.provenance]
        except QueueError as exc:
            # Only a queue file edited while the runner works gets here: run_queue refuses to
            # start when a selected job or the provenance needs a value that is not set.
            self._record(job, status="failed", reason=str(exc), ended=now_iso(), exit_code=None)
            self.bad_this_run += 1
            self.log(f"FAILED {job.id}: {exc}")
            return
        cwd = (root / job.cwd).resolve()
        env = base_env()
        env.update(overrides)
        state = read_state(self.paths)
        attempt = int(state["jobs"].get(job.id, {}).get("attempts", 0)) + 1
        commit, dirty = git_info(root)
        provenance = file_hashes(provenance_paths, root)
        started = now_iso()
        meta: dict[str, Any] = {
            "id": job.id,
            "attempt": attempt,
            "status": "running",
            "argv": argv,
            "command": shlex.join(argv),
            "cwd": str(cwd),
            "env_overrides": overrides,
            "outputs": outputs,
            "timeout_s": job.timeout_s,
            "nice": self.nice,
            "git_commit": commit,
            "git_dirty": dirty,
            "provenance_sha256": provenance,
            "queue_file": str(self.queue.path),
            "spec_hash": job.spec_hash,
            "host": socket.gethostname(),
            "started": started,
            "ended": None,
            "duration_s": None,
            "exit_code": None,
            "signal": None,
            "peak_rss_bytes": None,
            "reason": None,
            "pid": None,
            "program": None,
            "missing_outputs": [],
            "files": None,
        }
        self._record(
            job,
            status="running",
            attempts=attempt,
            started=started,
            ended=None,
            exit_code=None,
            signal=None,
            peak_rss_bytes=None,
            duration_s=None,
            reason=None,
            pid=None,
            boot_id=self.boot,
            spec_hash=job.spec_hash,
            started_epoch=None,
        )
        self.log(f"start {job.id} (attempt {attempt}): {shlex.join(argv)}  [cwd {cwd}]")

        if not cwd.is_dir() or not (cwd == root or root in cwd.parents):
            self._fail_to_start(job, job_dir, meta, f"working directory {cwd} not found in {root}")
            return
        program = argv[0] if "/" in argv[0] else shutil.which(argv[0], path=env.get("PATH"))
        if program is None:
            self._fail_to_start(job, job_dir, meta, f"program {argv[0]!r} not found on PATH")
            return
        meta["program"] = program
        files: list[IO[bytes]] = []
        wall_start = time.time()
        # The duration and the timeout count from the start, not from after job.json and the
        # state are written (both are fsynced and can take longer than a short job).
        mono = time.monotonic()
        try:
            stdout = (job_dir / "stdout.log").open("wb")
            files.append(stdout)
            stderr = (job_dir / "stderr.log").open("wb")
            files.append(stderr)
            proc = subprocess.Popen(
                [program, *argv[1:]],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                preexec_fn=_set_priority(self.nice),  # safe: the runner has no threads
            )
        except OSError as exc:
            for f in files:
                f.close()
            self._fail_to_start(job, job_dir, meta, f"could not start: {exc}")
            return
        meta["pid"] = proc.pid
        write_json_atomic(job_dir / "job.json", meta)
        self._record(job, pid=proc.pid, started_epoch=wall_start)
        self.running[job.id] = _Running(
            job, proc, files, meta, job_dir, mono, wall_start, mono + job.timeout_s
        )

    # --- supervising -----------------------------------------------------------------------

    def _terminate(self, r: _Running, reason: str) -> None:
        if r.term_at is None:
            r.stop_reason = r.stop_reason or reason
            r.term_at = time.monotonic()
            _killpg(r.proc.pid, signal.SIGTERM)

    def supervise(self) -> None:
        """Timeouts, escalation to SIGKILL and signal-driven stops of the running jobs."""
        now = time.monotonic()
        for r in self.running.values():
            if self.signals:
                self._terminate(r, "interrupted")
                if self.signals >= 2 and not r.killed:
                    r.killed = True
                    _killpg(r.proc.pid, signal.SIGKILL)
            elif now >= r.deadline:
                if r.term_at is None:
                    self.log(
                        f"timeout {r.job.id} after {format_duration(r.job.timeout_s)}: SIGTERM"
                    )
                self._terminate(r, "timeout")
            if (
                r.term_at is not None
                and not r.killed
                and now >= r.term_at + self.queue.kill_grace_s
            ):
                r.killed = True
                _killpg(r.proc.pid, signal.SIGKILL)

    def reap(self) -> None:
        """Collect every finished job and record its outcome."""
        for job_id in list(self.running):
            r = self.running[job_id]
            try:
                pid, status, rusage = os.wait4(r.proc.pid, os.WNOHANG)
            except ChildProcessError:  # already reaped elsewhere; should not happen
                pid, status, rusage = r.proc.pid, 0, None
            if pid == 0:
                continue
            r.proc.returncode = os.waitstatus_to_exitcode(status)
            # Anything the job left in its process group must not outlive it.
            _killpg(r.proc.pid, signal.SIGKILL)
            for f in r.files:
                f.close()
            del self.running[job_id]
            self.finish(r, r.proc.returncode, rusage)

    def finish(self, r: _Running, code: int, rusage: Any) -> None:
        """Decide the final status of a reaped job and write job.json and the state."""
        duration = time.monotonic() - r.started_mono
        missing = [
            o for o in r.meta["outputs"] if not _fresh(Path(r.meta["cwd"]) / o, r.wall_start)
        ]
        sig = -code if code < 0 else None
        if r.stop_reason == "interrupted":
            status, reason = "interrupted", "stopped by a signal to the runner"
        elif r.stop_reason == "timeout":
            status, reason = "timeout", f"timeout after {format_duration(r.job.timeout_s)}"
        elif code != 0:
            status = "failed"
            reason = f"killed by signal {sig}" if sig else f"exit code {code}"
        elif missing:
            status, reason = "failed", f"exit code 0 but missing outputs: {missing}"
        else:
            status, reason = "done", None
        manifest = None
        if status == "done":
            # Durable before "done" is written anywhere: a crash between the two must leave
            # the job rerunnable, never "done" with files still only in the page cache.
            outputs = [Path(r.meta["cwd"]) / o for o in r.meta["outputs"]]
            try:
                manifest = sync_job(r.job_dir, outputs)
            except OSError as exc:
                status, reason = "failed", f"exit code 0 but outputs not durable: {exc}"
        fields: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "ended": now_iso(),
            "duration_s": round(duration, 3),
            "exit_code": code,
            "signal": sig,
            "peak_rss_bytes": peak_rss_bytes(rusage) if rusage is not None else None,
            "pid": None,
        }
        r.meta.update(fields, missing_outputs=missing, files=manifest)
        write_json_atomic(r.job_dir / "job.json", r.meta)
        self._record(r.job, **fields)
        if status in FINISHED_BAD:
            self.bad_this_run += 1
        rss = fields["peak_rss_bytes"]
        mem = f", peak RSS {rss / 2**30:.2f} GB" if rss else ""
        word = status.upper() if status != "done" else "done"
        self.log(
            f"{word} {r.job.id} in {format_duration(duration)} "
            f"(exit {code}{mem}){': ' + reason if reason else ''}"
        )

    def _too_big(self, job: Job) -> bool:
        """Fail ``job`` without starting it if its memory estimate alone exceeds the cap."""
        if self.memory_cap is None or job.memory_bytes is None:
            return False
        if job.memory_bytes <= self.memory_cap:
            return False
        reason = (
            f"memory estimate {job.memory_bytes / 2**30:.1f} GB exceeds the queue's cap of "
            f"{self.memory_cap / 2**30:.1f} GB ({CAPACITY_SHARE:.0%} of physical memory)"
        )
        self._record(job, status="failed", reason=reason, ended=now_iso(), exit_code=None)
        self.bad_this_run += 1
        self.log(f"FAILED {job.id}: {reason}")
        return True

    def _lost_dependencies(self, job: Job) -> bool:
        """Fail every ``done`` dependency of ``job`` whose files no longer match job.json.

        Returns True if one was failed (``job`` is then blocked, not started). A dependency
        whose ``job.json`` has no ``files`` (finished before the manifest existed) cannot be
        checked and counts as intact; one whose ``job.json`` is gone or unreadable does not.
        """
        state = read_state(self.paths)
        lost = False
        for dep in job.after:
            if job_status(state, dep) != "done":
                continue
            dep_dir = self.paths.job_dir(dep)
            try:
                meta = json.loads((dep_dir / "job.json").read_text())
                manifest = meta.get("files") if isinstance(meta, dict) else None
                problems = [] if manifest is None else verify_manifest(dep_dir, manifest)
            except (OSError, ValueError) as exc:
                problems = [f"job.json unreadable ({exc})"]
            if not problems:
                continue
            more = f"; {len(problems) - 3} more" if len(problems) > 3 else ""
            reason = f"files changed after it was done: {'; '.join(problems[:3])}{more}"
            self._record_id(dep, status="failed", reason=reason)
            self.bad_this_run += 1
            self.log(f"LOST {dep} (needed by {job.id}): {reason}; 'expq retry {dep}' reruns it")
            lost = True
        return lost

    # --- main loop ---------------------------------------------------------------------------

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        self.signals += 1

    def _finish(self, draining: bool) -> int:
        """End the run when nothing is running and nothing may start; return the exit code."""
        state = read_state(self.paths)
        left = self.runnable(state)
        blocked = [j for j in self.candidates(state) if j not in left]
        if blocked and not draining:
            self.log(
                f"{len(blocked)} job(s) not started: their 'after' jobs are not "
                "done (retry a failed one, or select it with --only)"
            )
            for j in blocked:
                self.log(f"BLOCKED {j.id}: {self.blockers(state, j)}")
        if draining:
            with contextlib.suppress(FileNotFoundError):
                self.paths.stop_file.unlink()
            if left:
                self.log(f"stopped on request; {len(left)} job(s) left")
                return EXIT_STOPPED
        self.log(f"queue finished; {self.bad_this_run} job(s) failed or timed out in this run")
        if self.bad_this_run:
            return EXIT_JOB_FAILED
        # Blocked jobs must not look like success to a script gating on the exit code.
        return EXIT_BLOCKED if blocked else EXIT_OK

    def loop(self) -> int:
        """Run until nothing is left to run; return the exit code of ``expq run``."""
        while True:
            self.reap()
            self.supervise()
            if self.signals:
                if not self.running:
                    self.log(
                        "stopped by a signal; interrupted jobs will be rerun by the next 'expq run'"
                    )
                    return EXIT_INTERRUPTED
            else:
                draining = self.paths.stop_file.exists()
                if not draining and len(self.running) < self.max_jobs:
                    self.reload()
                    todo = self.runnable(read_state(self.paths))
                    if todo and self._too_big(todo[0]):
                        continue  # failed without starting; look at the next one
                    if todo and self.fits(todo[0]):
                        if self._lost_dependencies(todo[0]):
                            continue  # a dependency was failed; the job is blocked now
                        self.start(todo[0])
                        continue  # fill the remaining slots before sleeping
                    # Strict file order: a job waiting for memory is not overtaken.
                if not self.running:
                    return self._finish(draining)
            time.sleep(self.poll_s)

    def run(self) -> int:
        """Lock, recover, install signal handlers, loop; always release the lock."""
        results = self.paths.results_dir.expanduser().resolve()
        root = self.queue.root
        if results == root or root in results.parents:
            raise QueueError(
                f"results directory {results} is inside the repository {root}; "
                "results must stay outside git"
            )
        self.paths = Paths(results)
        lock = RunnerLock(self.paths)
        lock.acquire()
        handled = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous = {s: signal.getsignal(s) for s in handled}
        try:
            for s in handled:
                signal.signal(s, self._on_signal)
            with contextlib.suppress(FileNotFoundError):
                self.paths.stop_file.unlink()  # a stale request must not stop a new run
            self.boot = boot_id()
            cap = "unknown" if self.memory_cap is None else f"{self.memory_cap / 2**30:.1f} GB"
            self.log(
                f"runner pid {os.getpid()}, queue {self.queue.path}, results "
                f"{results}, max jobs {self.max_jobs}, nice {self.nice}, memory cap {cap}"
            )
            self.recover()
            return self.loop()
        finally:
            # Normally empty here. After an unexpected error (disk full, a bug) the jobs are
            # stopped rather than left running unsupervised; their records stay "running"
            # and the next runner reruns them.
            for r in self.running.values():
                _killpg(r.proc.pid, signal.SIGTERM)
            for s, handler in previous.items():
                signal.signal(s, handler)
            lock.release()


def run_queue(
    queue_path: Path,
    paths: Paths,
    *,
    max_jobs: int = 1,
    nice: int = DEFAULT_NICE,
    only: Sequence[str] = (),
    poll_s: float = 0.1,
    out: IO[str] | None = None,
) -> int:
    """Run the queue until it is empty, stopped or interrupted; return the exit code.

    Raises :class:`~expqueue.queuefile.QueueError` for an invalid queue file, a results
    directory inside the repository, ``max_jobs < 1`` or above :func:`max_parallel_jobs`
    (half the CPUs), ``nice`` outside 0-20, an ``--only`` pattern that matches no job or a
    ``{data_dir}`` placeholder (in a selected job or in ``provenance``) while
    ``$EXPQ_DATA_DIR`` is not set, and :class:`~expqueue.state.LockedError` if another
    runner is active (the CLI maps them to exit codes 2 and 3).
    """
    if max_jobs < 1:
        raise QueueError(f"--jobs must be at least 1, got {max_jobs}")
    if max_jobs > (limit := max_parallel_jobs()):
        raise QueueError(
            f"--jobs {max_jobs} exceeds the queue's cap of {limit}: at most "
            f"{CAPACITY_SHARE:.0%} of the {cpu_count()} CPUs"
        )
    if not 0 <= nice <= 20:
        raise QueueError(f"--nice must be between 0 and 20, got {nice}")
    queue = load_queue(Path(queue_path))
    for pattern in only:  # a typo in --only would otherwise silently run nothing
        if not any(fnmatch.fnmatchcase(j.id, pattern) for j in queue.jobs):
            raise QueueError(f"--only {pattern!r} matches no job of {queue.path}")
    if data_dir_from_env() is None:
        texts = list(queue.provenance)
        for j in queue.jobs:
            if not only or any(fnmatch.fnmatchcase(j.id, p) for p in only):
                texts += _job_texts(j)
        if uses_placeholder(texts, "data_dir"):
            raise QueueError(
                f"{queue.path} uses {{data_dir}} but ${DATA_DIR_ENV} is not set (it has no default)"
            )
    runner = Runner(
        Path(queue_path),
        queue,
        paths,
        max_jobs,
        nice,
        tuple(only),
        poll_s,
        out if out is not None else sys.stdout,
        memory_cap=memory_budget(),
    )
    return runner.run()
