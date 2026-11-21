"""Command line of the job queue: ``expq run | status | list | retry | skip | stop``.

Examples (with ``EXPQ_RESULTS_DIR`` set, from the directory of ``queue.toml`` or below it;
full guide in ``README.md``)::

    expq list                      # the jobs of queue.toml, in order, with estimates
    expq run                       # run everything left, one job at a time, nice 19
    expq run --only 'train-*-60'   # only the jobs whose id matches a glob
    expq status                    # progress; --json for machines
    expq retry train-a-60          # failed / timed-out / done -> pending
    expq retry --failed            # every failed or timed-out job -> pending
    expq skip eval-b               # never run it (until retried)
    expq stop                      # drain: finish the running job, start no new one
    expq stop --now                # SIGTERM the runner: jobs interrupted, rerun later

Global options: ``--queue PATH`` (default: ``queue.toml`` in the current directory or the
nearest parent that has one) and ``--results-dir DIR`` (default ``$EXPQ_RESULTS_DIR``). The
results directory is required and has no built-in default: it holds the runner lock, so two
projects must never share one.

Exit codes: 0 ok; 1 at least one job failed or timed out in this run; 2 bad arguments,
invalid queue file or unknown job id; 3 another runner is active, or the job is running;
4 stopped on request (``stop``) with jobs left; 130 interrupted by a signal (jobs that were
running are marked ``interrupted`` and rerun by the next ``run``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from expqueue.queuefile import Queue, QueueError, load_queue
from expqueue.runner import (
    DEFAULT_NICE,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_USAGE,
    format_duration,
    run_queue,
)
from expqueue.state import (
    FINISHED_BAD,
    RESULTS_DIR_ENV,
    RUNNABLE,
    STATUSES,
    LockedError,
    Paths,
    State,
    active_runner_pid,
    job_status,
    read_state,
    results_dir_from_env,
    update_state,
)


def find_default_queue(start: Path) -> Path:
    """``queue.toml`` in ``start`` or the nearest parent that has one.

    Raises :class:`QueueError` if no parent has it (then ``--queue`` is needed).
    """
    for d in (start, *start.parents):
        candidate = d / "queue.toml"
        if candidate.is_file():
            return candidate
    raise QueueError(f"no queue.toml in {start} or above it; pass --queue")


def resolve_results_dir(option: Path | None) -> Path:
    """The results directory: ``--results-dir`` if given, else ``$EXPQ_RESULTS_DIR``.

    Raises :class:`QueueError` if neither is set. There is deliberately no default: the
    directory holds ``runner.lock`` and the state, and a shared default would make two
    projects on one machine share a runner.
    """
    results = option or results_dir_from_env()
    if results is None:
        raise QueueError(
            f"no results directory: pass --results-dir or set ${RESULTS_DIR_ENV} "
            "(one directory per project, outside the repository)"
        )
    return results.expanduser()


def _effective_status(state: State, job_id: str, runner_alive: bool) -> str:
    status = job_status(state, job_id)
    # A "running" record without a live runner is a dead runner's job: rerun on next run.
    return "interrupted" if status == "running" and not runner_alive else status


def _status_rows(queue: Queue, state: State, runner_alive: bool) -> list[dict[str, Any]]:
    rows = []
    for job in queue.jobs:
        record = state["jobs"].get(job.id, {})
        rows.append(
            {
                "id": job.id,
                "status": _effective_status(state, job.id, runner_alive),
                "attempts": record.get("attempts", 0),
                "started": record.get("started"),
                "ended": record.get("ended"),
                "duration_s": record.get("duration_s"),
                "exit_code": record.get("exit_code"),
                "peak_rss_bytes": record.get("peak_rss_bytes"),
                "reason": record.get("reason"),
                "estimate_s": job.estimate_s,
                "spec_changed": bool(record.get("spec_hash"))
                and record.get("spec_hash") != job.spec_hash,
            }
        )
    return rows


def cmd_status(queue: Queue, paths: Paths, as_json: bool) -> int:
    """Print the runner, the per-job status and the estimated remaining time."""
    pid = active_runner_pid(paths)
    state = read_state(paths)
    rows = _status_rows(queue, state, pid is not None)
    counts = Counter(r["status"] for r in rows)
    remaining = sum(r["estimate_s"] or 0 for r in rows if r["status"] in RUNNABLE | {"running"})
    unknown = sum(
        1 for r in rows if r["status"] in RUNNABLE | {"running"} and r["estimate_s"] is None
    )
    orphans = sorted(set(state["jobs"]) - {j.id for j in queue.jobs})
    if as_json:
        print(
            json.dumps(
                {
                    "runner_pid": pid,
                    "results_dir": str(paths.results_dir),
                    "queue_file": str(queue.path),
                    "counts": dict(counts),
                    "remaining_estimate_s": remaining,
                    "jobs": rows,
                    "records_not_in_queue": orphans,
                },
                indent=2,
            )
        )
        return EXIT_OK
    runner = f"running (pid {pid})" if pid else "not running"
    if pid and paths.stop_file.exists():
        runner += ", stop requested (draining)"
    print(f"runner: {runner}\nresults: {paths.results_dir}\nqueue: {queue.path}")
    print(f"{'id':<34} {'status':<12} {'try':>3} {'time':>9} {'exit':>5} {'peak GB':>7}  note")
    for r in rows:
        rss = f"{r['peak_rss_bytes'] / 2**30:.2f}" if r["peak_rss_bytes"] else "-"
        exit_code = "-" if r["exit_code"] is None else str(r["exit_code"])
        if r["status"] == "running":
            note = f"started {r['started']}, estimate {format_duration(r['estimate_s'])}"
        else:
            note = r["reason"] or ""
        if r["spec_changed"]:
            note = ("definition changed since it ran; " + note).strip("; ")
        print(
            f"{r['id']:<34} {r['status']:<12} {r['attempts']:>3} "
            f"{format_duration(r['duration_s']):>9} {exit_code:>5} {rss:>7}  {note}"
        )
    summary = ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s])
    print(
        f"\n{summary}; estimated remaining {format_duration(remaining)}"
        + (f" (+{unknown} job(s) without estimate)" if unknown else "")
    )
    if orphans:
        print(f"state records of jobs no longer in the queue: {', '.join(orphans)}")
    return EXIT_OK


def cmd_list(queue: Queue) -> int:
    """Print every job with its working directory, timeout, estimate and exact argv."""

    def env_text(env: dict[str, str]) -> str:
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))

    shared = len({env_text(j.env) for j in queue.jobs}) == 1
    if shared and queue.jobs and queue.jobs[0].env:
        print(f"env of every job: {env_text(queue.jobs[0].env)}\n")
    total = 0.0
    for i, job in enumerate(queue.jobs, 1):
        total += job.estimate_s or 0
        print(
            f"{i:>3}. {job.id}  [cwd {job.cwd}, timeout {format_duration(job.timeout_s)}, "
            f"estimate {format_duration(job.estimate_s)}]"
        )
        print(f"     {shlex.join(job.argv)}")
        if not shared and job.env:
            print(f"     env: {env_text(job.env)}")
    print(f"\n{len(queue.jobs)} jobs, estimated total {format_duration(total)}")
    return EXIT_OK


def _check_ids(queue: Queue, ids: list[str]) -> str | None:
    known = {j.id for j in queue.jobs}
    unknown = [i for i in ids if i not in known]
    return f"unknown job id(s): {', '.join(unknown)}" if unknown else None


def cmd_retry(queue: Queue, paths: Paths, ids: list[str], failed: bool) -> int:
    """Set jobs back to ``pending`` (``--failed``: every failed or timed-out job)."""
    if not ids and not failed:
        print("expq: retry needs job ids or --failed", file=sys.stderr)
        return EXIT_USAGE
    if error := _check_ids(queue, ids):
        print(f"expq: {error}", file=sys.stderr)
        return EXIT_USAGE
    alive = active_runner_pid(paths) is not None
    result: dict[str, Any] = {"busy": [], "changed": []}

    def change(state: State) -> None:
        targets = list(ids)
        if failed:
            targets += [
                j.id
                for j in queue.jobs
                if job_status(state, j.id) in FINISHED_BAD and j.id not in targets
            ]
        for job_id in targets:
            status = job_status(state, job_id)
            if status == "running" and alive:
                result["busy"].append(job_id)
                continue
            record = state["jobs"].setdefault(job_id, {"attempts": 0})
            record.update(status="pending", reason=f"retry requested (was {status})", pid=None)
            result["changed"].append(f"{job_id} ({status} -> pending)")

    update_state(paths, change)
    for line in result["changed"]:
        print(line)
    if not result["changed"] and not result["busy"]:
        print("nothing to retry")
    if result["busy"]:
        print(f"expq: running, not changed: {', '.join(result['busy'])}", file=sys.stderr)
        return EXIT_LOCKED
    return EXIT_OK


def cmd_skip(queue: Queue, paths: Paths, ids: list[str]) -> int:
    """Mark jobs ``skipped`` so no runner starts them (``retry`` undoes it)."""
    if error := _check_ids(queue, ids):
        print(f"expq: {error}", file=sys.stderr)
        return EXIT_USAGE
    alive = active_runner_pid(paths) is not None
    busy: list[str] = []
    lines: list[str] = []

    def change(state: State) -> None:
        for job_id in ids:
            status = job_status(state, job_id)
            if status == "running" and alive:
                busy.append(job_id)
            elif status == "done":
                lines.append(f"{job_id} is done, unchanged")
            else:
                record = state["jobs"].setdefault(job_id, {"attempts": 0})
                record.update(status="skipped", reason=f"skip requested (was {status})", pid=None)
                lines.append(f"{job_id} ({status} -> skipped)")

    update_state(paths, change)
    for line in lines:
        print(line)
    if busy:
        print(
            f"expq: running, not skipped (stop the runner first): {', '.join(busy)}",
            file=sys.stderr,
        )
        return EXIT_LOCKED
    return EXIT_OK


def cmd_stop(paths: Paths, now: bool) -> int:
    """Ask the active runner to drain (STOP file) or, with ``now``, send it SIGTERM."""
    pid = active_runner_pid(paths)
    if pid is None:
        print("no runner is active")
        return EXIT_OK
    if now:
        if pid < 0:
            print("expq: a runner is active but its pid is unknown", file=sys.stderr)
            return EXIT_LOCKED
        os.kill(pid, signal.SIGTERM)
        print(
            f"sent SIGTERM to runner {pid}: running jobs are interrupted and will be "
            "rerun by the next 'expq run'"
        )
        return EXIT_OK
    paths.stop_file.touch()
    print(f"stop requested: runner {pid} finishes its running job(s) and starts no new one")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser of ``expq``."""
    parser = argparse.ArgumentParser(prog="expq", description=__doc__.split("\n")[0])
    parser.add_argument("--queue", type=Path, help="queue file (default: nearest queue.toml)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help=f"results directory, required (default: ${RESULTS_DIR_ENV}; no built-in default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the jobs left, in file order")
    p_run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="jobs at a time (default 1; at most half the CPUs, and the running jobs' memory "
        "estimates stay within half the physical memory)",
    )
    p_run.add_argument(
        "--nice",
        type=int,
        default=DEFAULT_NICE,
        help=f"niceness of the jobs (default {DEFAULT_NICE})",
    )
    p_run.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="GLOB",
        help="run only jobs whose id matches (repeatable)",
    )
    p_run.add_argument("--poll", type=float, default=0.5, help=argparse.SUPPRESS)
    p_status = sub.add_parser("status", help="progress of the queue")
    p_status.add_argument("--json", action="store_true", help="machine-readable output")
    sub.add_parser("list", help="the jobs of the queue file")
    p_retry = sub.add_parser("retry", help="set jobs back to pending")
    p_retry.add_argument("ids", nargs="*")
    p_retry.add_argument("--failed", action="store_true", help="every failed or timed-out job")
    p_skip = sub.add_parser("skip", help="never run these jobs (until retried)")
    p_skip.add_argument("ids", nargs="+")
    p_stop = sub.add_parser("stop", help="drain the runner (or --now: interrupt it)")
    p_stop.add_argument("--now", action="store_true", help="SIGTERM the runner")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``expq`` script; returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        paths = Paths(resolve_results_dir(args.results_dir))
        if args.command == "stop":
            return cmd_stop(paths, args.now)
        queue_path = args.queue or find_default_queue(Path.cwd())
        if args.command == "run":
            return run_queue(
                queue_path,
                paths,
                max_jobs=args.jobs,
                nice=args.nice,
                only=args.only,
                poll_s=args.poll,
            )
        queue = load_queue(queue_path)
        if args.command == "status":
            return cmd_status(queue, paths, args.json)
        if args.command == "list":
            return cmd_list(queue)
        if args.command == "retry":
            return cmd_retry(queue, paths, args.ids, args.failed)
        return cmd_skip(queue, paths, args.ids)
    except QueueError as exc:
        print(f"expq: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except LockedError as exc:
        print(f"expq: {exc}", file=sys.stderr)
        return EXIT_LOCKED
    except ValueError as exc:  # corrupt state file
        print(f"expq: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
