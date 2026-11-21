"""Durable job results: flush a finished job's files to disk and fingerprint them.

Why: a host reboot (ext4 under WSL2) once left the outputs of two finished jobs at 0 bytes
although the queue had recorded both ``done``. The runner wrote ``done`` (its state file is
fsynced) while the jobs' own files were still only in the page cache: ext4's delayed
allocation commits a new file's metadata before its data, so after the crash the files
existed, empty. The dependent jobs failed three days later. Two guards, both used by
:mod:`expqueue.runner`:

- :func:`sync_job`, before ``done`` is recorded: every regular file of the job directory
  (``attempts/`` and ``job.json`` excluded: earlier attempts were synced when they finished,
  and ``job.json`` is written atomically and fsynced by itself) and every declared output
  outside it are flushed, then the directories that hold them. A job may declare no outputs or
  only some of them, so the whole directory is flushed, not
  the declared list. On macOS ``fsync`` does not flush the drive's cache, so ``F_FULLFSYNC`` is
  used there (falling back to ``fsync`` where the file system refuses it). Any error makes the
  job ``failed``: the outputs are not known to be on disk.
- the manifest :func:`sync_job` returns (size and sha256 of each of those files), written to
  ``job.json`` as ``files``; :func:`verify_manifest` compares the files on disk with it before a
  dependent job starts, so outputs lost anyway (a disk that ignores flushes, a file truncated or
  deleted by hand) stop the dependents instead of feeding them.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

Manifest = dict[str, dict[str, Any]]

# The runner's own file in the job directory: rewritten after the manifest, synced on its own.
JOB_JSON = "job.json"
ATTEMPTS = "attempts"


def sync_fd(fd: int) -> None:
    """Flush ``fd`` to stable storage: ``F_FULLFSYNC`` where it exists (macOS), else ``fsync``.

    Raises:
        OSError: the flush failed (the data may not be on disk).

    """
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return
        except OSError:
            pass  # e.g. a file system without F_FULLFSYNC support: fsync is the best left
    os.fsync(fd)


def sync_path(path: Path) -> None:
    """Flush the file or directory ``path`` (opened read-only, which is enough for both).

    Raises:
        OSError: it cannot be opened or flushed.

    """
    fd = os.open(path, os.O_RDONLY)
    try:
        sync_fd(fd)
    finally:
        os.close(fd)


def _regular_files(top: Path, skip_top: Iterable[str] = ()) -> list[Path]:
    """Regular files under ``top`` (symlinks not followed), without the names in ``skip_top``.

    A regular-file ``top`` is returned alone; a missing one gives nothing (the runner has
    already failed the job for a missing declared output).
    """
    try:
        mode = os.lstat(top).st_mode
    except FileNotFoundError:
        return []
    if stat.S_ISREG(mode):
        return [top]
    if not stat.S_ISDIR(mode):
        return []
    skip = set(skip_top)
    found: list[Path] = []
    for entry in sorted(os.scandir(top), key=lambda e: e.name):
        if entry.name in skip:
            continue
        if entry.is_dir(follow_symlinks=False):
            found += _regular_files(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):
            found.append(Path(entry.path))
    return found


def job_files(job_dir: Path, outputs: Sequence[Path]) -> list[Path]:
    """The files :func:`sync_job` flushes: the job directory's and the declared outputs'.

    Without ``attempts/`` and ``job.json``; outputs inside the job directory are already part
    of it. Declared outputs may be files or directories (walked). Sorted, no duplicates.
    """
    files = set(_regular_files(job_dir, skip_top=(ATTEMPTS, JOB_JSON)))
    job_dir = job_dir.resolve()
    for out in outputs:
        resolved = out.resolve()
        if resolved == job_dir or job_dir in resolved.parents:
            continue
        files.update(_regular_files(out))
    return sorted(files)


def _key(path: Path, job_dir: Path) -> str:
    """Manifest key: the path relative to the job directory, else the absolute path."""
    try:
        return path.relative_to(job_dir).as_posix()
    except ValueError:
        return str(path.resolve())


def _path_of(key: str, job_dir: Path) -> Path:
    return Path(key) if os.path.isabs(key) else job_dir / key


def _fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def sync_job(job_dir: Path, outputs: Sequence[Path]) -> Manifest:
    """Flush a finished job's files and their directories; return their manifest.

    Files first (each fingerprinted, then flushed), then every directory that holds one, the
    job directory and its parent (so the new entries themselves survive a crash). The manifest
    maps each file (relative to ``job_dir``, or absolute outside it) to ``{"bytes", "sha256"}``.

    Raises:
        OSError: a file or directory could not be read or flushed.

    """
    job_dir = job_dir.resolve()
    manifest: Manifest = {}
    dirs = {job_dir, job_dir.parent}
    for path in job_files(job_dir, outputs):
        size, sha = _fingerprint(path)
        sync_path(path)
        manifest[_key(path.resolve(), job_dir)] = {"bytes": size, "sha256": sha}
        dirs.add(path.resolve().parent)
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):  # deepest first
        sync_path(d)
    return manifest


def verify_manifest(job_dir: Path, manifest: Manifest) -> list[str]:
    """Compare the files on disk with ``manifest``; return the differences (empty: intact).

    Each problem names the file: missing, another size, or the same size with other content.
    """
    problems = []
    for key, expected in sorted(manifest.items()):
        path = _path_of(key, job_dir)
        try:
            size, sha = _fingerprint(path)
        except FileNotFoundError:
            problems.append(f"{key}: missing")
            continue
        except OSError as exc:
            problems.append(f"{key}: unreadable ({exc.strerror or exc})")
            continue
        if size != expected.get("bytes"):
            problems.append(f"{key}: {size} bytes, recorded {expected.get('bytes')}")
        elif sha != expected.get("sha256"):
            problems.append(f"{key}: content changed (sha256)")
    return problems
