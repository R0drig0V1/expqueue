"""The runner end to end with dummy ``python -c`` jobs.

Order, isolation from the shell, outcomes and metadata, timeouts, the lock, resuming after
a crash, stopping and the CLI.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from expqueue import runner as runner_mod
from expqueue.cli import main
from expqueue.state import RunnerLock, read_state, update_state, write_json_atomic
from tests.conftest import Sandbox, py_job

# Appends "<id> <start> <end>" to the file in argv[1] after sleeping argv[3] seconds.
TRACE = (
    "import sys, time; t0 = time.time(); time.sleep(float(sys.argv[3])); "
    "open(sys.argv[1], 'a').write(f'{sys.argv[2]} {{t0}} {time.time()}\\n')"
)
Wait = Callable[..., None]


def status_of(sandbox: Sandbox, job_id: str) -> str:
    return str(read_state(sandbox.paths)["jobs"][job_id]["status"])


def test_jobs_run_serially_in_file_order(sandbox: Sandbox) -> None:
    trace = sandbox.root / "trace.txt"
    sandbox.write([py_job(i, TRACE, str(trace), i, "0.15") for i in ("c", "a", "b")])
    assert main(sandbox.args("run")) == 0
    rows = [line.split() for line in trace.read_text().splitlines()]
    assert [r[0] for r in rows] == ["c", "a", "b"]
    for prev, nxt in itertools.pairwise(rows):
        assert float(prev[2]) <= float(nxt[1])  # never overlapping
    assert all(status_of(sandbox, i) == "done" for i in "abc")


def spans(trace: Path) -> dict[str, tuple[float, float]]:
    """``id -> (start, end)`` from a TRACE file."""
    return {
        r[0]: (float(r[1]), float(r[2])) for r in map(str.split, trace.read_text().splitlines())
    }


@pytest.fixture
def eight_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the machine has 8 CPUs and 8 GiB, so the 50% cap allows 4 jobs and 4 GiB."""
    monkeypatch.setattr(runner_mod, "cpu_count", lambda: 8)
    monkeypatch.setattr(runner_mod, "physical_memory", lambda: 8 * 2**30)


def test_two_slots_run_two_jobs_at_once(sandbox: Sandbox, eight_cpus: None) -> None:
    # Each job only succeeds if it sees the other one's marker: they must overlap.
    code = (
        "import pathlib, sys, time; d = pathlib.Path(sys.argv[1]); "
        "(d / sys.argv[2]).touch(); t = time.time() + 5\n"
        "while not (d / sys.argv[3]).exists():\n"
        "    time.sleep(0.02)\n"
        "    if time.time() > t: sys.exit(1)"
    )
    d = str(sandbox.root)
    sandbox.write([py_job("a", code, d, "a", "b"), py_job("b", code, d, "b", "a")])
    assert main(sandbox.args("run", "--jobs", "2")) == 0
    assert status_of(sandbox, "a") == status_of(sandbox, "b") == "done"


def test_argv_is_never_passed_through_a_shell(sandbox: Sandbox) -> None:
    hostile = [
        "; touch pwned1",
        "$(touch pwned2)",
        "`touch pwned3`",
        "a | touch pwned4",
        "*",
        "&& touch pwned5",
        "'quoted' \"double\"",
        "{job_dir}",
    ]
    code = "import json, sys; open('args.json', 'w').write(json.dumps(sys.argv[1:]))"
    sandbox.write([py_job("a", code, *hostile)])
    assert main(sandbox.args("run")) == 0
    received = json.loads((sandbox.root / "args.json").read_text())
    assert received == [*hostile[:-1], str(sandbox.results / "a")]  # only placeholders change
    assert not list(sandbox.root.glob("pwned*"))


def test_priority_threads_env_and_logs(sandbox: Sandbox) -> None:
    code = (
        "import os, sys; print(os.getpriority(os.PRIO_PROCESS, 0), "
        "os.environ['MKL_NUM_THREADS'], os.environ['OMP_NUM_THREADS'], "
        "os.environ.get('VIRTUAL_ENV', 'none')); print('to stderr', file=sys.stderr)"
    )
    sandbox.write(
        [py_job("a", code, env={"OMP_NUM_THREADS": "2"})],
        env={"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    assert main(sandbox.args("run")) == 0
    out = (sandbox.results / "a" / "stdout.log").read_text().split()
    assert out == ["19", "1", "2", "none"]
    assert (sandbox.results / "a" / "stderr.log").read_text() == "to stderr\n"


def test_metadata_of_a_finished_job(sandbox: Sandbox) -> None:
    code = "import sys; x = b'x' * (80 * 2**20); open(sys.argv[1], 'w').write('ok')"
    (sandbox.root / "deps.lock").write_text("deps\n")
    sandbox.write(
        [py_job("a", code, "{job_dir}/out.txt", outputs=["{job_dir}/out.txt"])],
        provenance=["deps.lock", "absent.lock"],
    )
    assert main(sandbox.args("run")) == 0
    meta = sandbox.job_meta("a")
    assert meta["status"] == "done" and meta["exit_code"] == 0 and meta["attempt"] == 1
    assert meta["argv"][-1] == str(sandbox.results / "a" / "out.txt")
    assert meta["command"].startswith(sys.executable)
    assert meta["cwd"] == str(sandbox.root.resolve())
    assert "git_commit" in meta and meta["started"] <= meta["ended"]
    assert meta["duration_s"] > 0 and meta["nice"] == 19
    assert meta["peak_rss_bytes"] > 80 * 2**20  # the 80 MB buffer
    assert meta["missing_outputs"] == []
    assert meta["provenance_sha256"] == {
        "deps.lock": hashlib.sha256(b"deps\n").hexdigest(),
        "absent.lock": None,
    }


def test_nonzero_exit_is_recorded_and_the_queue_continues(sandbox: Sandbox) -> None:
    sandbox.write(
        [
            py_job("bad", "import sys; sys.exit(3)"),
            py_job("good", "pass"),
            py_job("noout", "pass", outputs=["{job_dir}/missing.csv"]),
        ]
    )
    assert main(sandbox.args("run")) == 1
    state = read_state(sandbox.paths)["jobs"]
    assert state["bad"]["status"] == "failed" and state["bad"]["exit_code"] == 3
    assert state["good"]["status"] == "done"
    assert state["noout"]["status"] == "failed" and "missing outputs" in state["noout"]["reason"]
    # A failed job is not retried automatically.
    assert main(sandbox.args("run")) == 0
    assert read_state(sandbox.paths)["jobs"]["bad"]["attempts"] == 1


def test_missing_program_or_cwd_fails_the_job_only(sandbox: Sandbox) -> None:
    sandbox.write([py_job("nodir", "pass", cwd="absent"), py_job("ok", "pass")])
    assert main(sandbox.args("run")) == 1
    assert "not found" in read_state(sandbox.paths)["jobs"]["nodir"]["reason"]
    assert status_of(sandbox, "ok") == "done"


def test_timeout_terminates_then_kills(sandbox: Sandbox) -> None:
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    sandbox.write(
        [
            py_job("slow", "import time; time.sleep(30)", timeout="0.5s"),
            py_job("stubborn", stubborn, timeout="0.5s"),
            py_job("after", "pass"),
        ],
        kill_grace="0.5s",
    )
    t0 = time.monotonic()
    assert main(sandbox.args("run")) == 1
    assert time.monotonic() - t0 < 10
    state = read_state(sandbox.paths)["jobs"]
    assert state["slow"]["status"] == "timeout" and state["slow"]["exit_code"] == -signal.SIGTERM
    assert state["stubborn"]["status"] == "timeout"
    assert state["stubborn"]["exit_code"] == -signal.SIGKILL
    assert state["after"]["status"] == "done"


def test_lock_refuses_a_second_runner(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    sandbox.write([py_job("a", "pass")])
    with RunnerLock(sandbox.paths):
        assert main(sandbox.args("run")) == 3
        assert f"pid {os.getpid()}" in capsys.readouterr().err
        assert main(sandbox.args("status")) == 0
        assert f"running (pid {os.getpid()})" in capsys.readouterr().out
    assert "a" not in read_state(sandbox.paths)["jobs"]
    assert main(sandbox.args("run")) == 0  # released


def test_resume_skips_done_and_reruns_interrupted(sandbox: Sandbox) -> None:
    trace = sandbox.root / "trace.txt"
    sandbox.write([py_job(i, TRACE, str(trace), i, "0") for i in ("a", "b", "c")])
    assert main(sandbox.args("run", "--only", "a")) == 0
    # Simulated crash: the state says b is running, but no runner and no process exist.

    def crashed(state: dict[str, object]) -> None:
        jobs = state["jobs"]
        assert isinstance(jobs, dict)
        jobs["b"] = {"status": "running", "attempts": 1, "pid": 999_999, "boot_id": "other"}

    update_state(sandbox.paths, crashed)
    assert main(sandbox.args("run")) == 0
    assert [line.split()[0] for line in trace.read_text().splitlines()] == ["a", "b", "c"]
    jobs = read_state(sandbox.paths)["jobs"]
    assert jobs["a"]["attempts"] == 1 and jobs["b"]["attempts"] == 2


def _spawn_runner(sandbox: Sandbox) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "expqueue", *sandbox.args("run")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_crash_of_the_runner_kills_the_orphan_and_reruns(sandbox: Sandbox, wait_for: Wait) -> None:
    # First attempt: record the pid and hang; second attempt: succeed at once.
    code = (
        "import os, pathlib, time; p = pathlib.Path('pid.txt')\n"
        "if p.exists(): raise SystemExit(0)\n"
        "p.write_text(str(os.getpid())); time.sleep(60)"
    )
    sandbox.write([py_job("a", code), py_job("b", "pass")])
    runner = _spawn_runner(sandbox)
    pid_file = sandbox.root / "pid.txt"
    wait_for(lambda: pid_file.exists() and pid_file.read_text(), what="job a to start")
    orphan = int(pid_file.read_text())
    runner.kill()  # SIGKILL: no cleanup at all, as in a crash
    runner.wait()
    assert _alive(orphan) and status_of(sandbox, "a") == "running"
    assert main(sandbox.args("run")) == 0
    wait_for(lambda: not _alive(orphan), what="the orphan to die")
    jobs = read_state(sandbox.paths)["jobs"]
    assert jobs["a"]["status"] == "done" and jobs["a"]["attempts"] == 2
    assert jobs["b"]["status"] == "done"
    assert (sandbox.results / "a" / "attempts" / "1" / "job.json").exists()  # first try kept


def test_stop_now_interrupts_and_the_next_run_reruns(sandbox: Sandbox, wait_for: Wait) -> None:
    code = "import pathlib, time; pathlib.Path('started').touch(); time.sleep(60)"
    sandbox.write([py_job("a", code)])
    runner = _spawn_runner(sandbox)
    wait_for(lambda: (sandbox.root / "started").exists(), what="job a to start")
    assert main(sandbox.args("stop", "--now")) == 0
    assert runner.wait(timeout=10) == 130
    assert status_of(sandbox, "a") == "interrupted"
    assert sandbox.job_meta("a")["status"] == "interrupted"
    sandbox.write([py_job("a", "pass")])  # the definition may change before the rerun
    assert main(sandbox.args("run")) == 0
    assert status_of(sandbox, "a") == "done"


def test_stop_drains_after_the_running_job(sandbox: Sandbox, wait_for: Wait) -> None:
    go = sandbox.root / "go"
    code = (
        "import pathlib, time; pathlib.Path('started').touch()\n"
        "while not pathlib.Path('go').exists(): time.sleep(0.02)"
    )
    sandbox.write([py_job("a", code), py_job("b", "pass")])
    runner = _spawn_runner(sandbox)
    wait_for(lambda: (sandbox.root / "started").exists(), what="job a to start")
    assert main(sandbox.args("stop")) == 0
    go.touch()
    assert runner.wait(timeout=10) == 4
    assert status_of(sandbox, "a") == "done"
    assert "b" not in read_state(sandbox.paths)["jobs"]
    assert not sandbox.paths.stop_file.exists()
    assert main(sandbox.args("run")) == 0  # the request does not outlive the runner
    assert status_of(sandbox, "b") == "done"


def test_retry_and_skip(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    flag = sandbox.root / "fixed"
    code = "import pathlib, sys; sys.exit(0 if pathlib.Path('fixed').exists() else 5)"
    sandbox.write([py_job("flaky", code), py_job("skipme", "pass"), py_job("ok", "pass")])
    assert main(sandbox.args("skip", "skipme")) == 0
    assert main(sandbox.args("run")) == 1
    assert status_of(sandbox, "flaky") == "failed" and status_of(sandbox, "skipme") == "skipped"
    flag.touch()
    assert main(sandbox.args("retry", "--failed")) == 0
    assert main(sandbox.args("run")) == 0
    jobs = read_state(sandbox.paths)["jobs"]
    assert jobs["flaky"]["status"] == "done" and jobs["flaky"]["attempts"] == 2
    assert jobs["skipme"]["status"] == "skipped"
    assert main(sandbox.args("skip", "ok")) == 0 and status_of(sandbox, "ok") == "done"
    assert main(sandbox.args("retry", "skipme", "ok")) == 0
    assert main(sandbox.args("run")) == 0
    assert status_of(sandbox, "skipme") == "done"
    assert read_state(sandbox.paths)["jobs"]["ok"]["attempts"] == 2
    capsys.readouterr()
    assert main(sandbox.args("retry", "nope")) == 2
    assert main(sandbox.args("skip", "nope")) == 2
    assert "unknown job id" in capsys.readouterr().err


def test_status_list_and_json(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    sandbox.write([py_job("a", "pass", estimate="10m"), py_job("b", "pass", estimate="5m")])
    assert main(sandbox.args("run", "--only", "a")) == 0
    capsys.readouterr()
    assert main(sandbox.args("status", "--json")) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runner_pid"] is None and data["counts"] == {"done": 1, "pending": 1}
    assert data["remaining_estimate_s"] == 300
    assert main(sandbox.args("status")) == 0
    text = capsys.readouterr().out
    assert "1 pending, 1 done" in text and "estimated remaining 5m00s" in text
    assert main(sandbox.args("list")) == 0
    text = capsys.readouterr().out
    assert "2 jobs, estimated total 15m00s" in text and sys.executable in text


def test_status_flags_a_stale_running_record(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox.write([py_job("a", "pass")])
    write_json_atomic(
        sandbox.paths.state_file,
        {"version": 1, "jobs": {"a": {"status": "running", "attempts": 1}}},
    )
    assert main(sandbox.args("status", "--json")) == 0
    assert json.loads(capsys.readouterr().out)["jobs"][0]["status"] == "interrupted"


def test_configuration_errors_exit_2(sandbox: Sandbox, tmp_path: Path) -> None:
    sandbox.write([py_job("a", "pass")])
    inside = ["--queue", str(sandbox.queue), "--results-dir", str(sandbox.root / "res"), "run"]
    assert main(inside) == 2  # results inside the repository
    assert main(sandbox.args("run", "--jobs", "0")) == 2
    assert main(sandbox.args("run", "--nice", "25")) == 2
    assert main(sandbox.args("run", "--only", "typo*")) == 2
    sandbox.queue.write_text("not toml [")
    assert main(sandbox.args("run")) == 2
    assert main(sandbox.args("status")) == 2
    sandbox.write([py_job("a", "pass")])
    sandbox.paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    sandbox.paths.state_file.write_text("{broken")
    assert main(sandbox.args("status")) == 2


def test_outputs_of_an_earlier_attempt_do_not_count(sandbox: Sandbox) -> None:
    write = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('v1')"
    sandbox.write([py_job("a", write, "{job_dir}/out.csv", outputs=["{job_dir}/out.csv"])])
    assert main(sandbox.args("run")) == 0
    # The new definition no longer writes the output: the old file must not satisfy it.
    sandbox.write([py_job("a", "pass", outputs=["{job_dir}/out.csv"])])
    assert main(sandbox.args("retry", "a")) == 0
    assert main(sandbox.args("run")) == 1
    assert "missing outputs" in read_state(sandbox.paths)["jobs"]["a"]["reason"]
    archived = sandbox.results / "a" / "attempts" / "1"
    assert (archived / "out.csv").read_text() == "v1" and (archived / "job.json").exists()
    # An output outside the job directory must be written during the attempt, too.
    old = sandbox.root / "old.csv"
    old.write_text("old")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    sandbox.write([py_job("b", "pass", outputs=["old.csv"])])
    assert main(sandbox.args("run")) == 1
    assert status_of(sandbox, "b") == "failed"


def test_jobs_above_half_the_cpus_are_refused(
    sandbox: Sandbox, eight_cpus: None, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox.write([py_job("a", "pass")])
    assert main(sandbox.args("run", "--jobs", "5")) == 2
    assert "exceeds the queue's cap of 4" in capsys.readouterr().err
    assert main(sandbox.args("run", "--jobs", "4")) == 0


def test_after_waits_for_its_dependency(sandbox: Sandbox, eight_cpus: None) -> None:
    trace = sandbox.root / "trace.txt"
    sandbox.write(
        [
            py_job("fc", TRACE, str(trace), "fc", "0.4"),
            py_job("mpc", TRACE, str(trace), "mpc", "0", after=["fc"]),
            py_job("free", TRACE, str(trace), "free", "0"),
        ]
    )
    assert main(sandbox.args("run", "--jobs", "3")) == 0
    rows = spans(trace)
    assert rows["mpc"][0] >= rows["fc"][1]  # started only after its dependency ended
    assert rows["free"][0] < rows["fc"][1]  # an independent later job did not wait


def test_a_failed_dependency_blocks_its_dependents(sandbox: Sandbox, eight_cpus: None) -> None:
    sandbox.write(
        [
            py_job("fc", "import sys; sys.exit(3)"),
            py_job("mpc", "pass", after=["fc"]),
            py_job("other", "pass"),
        ]
    )
    assert main(sandbox.args("run", "--jobs", "2")) == 1
    assert status_of(sandbox, "fc") == "failed" and status_of(sandbox, "other") == "done"
    assert "mpc" not in read_state(sandbox.paths)["jobs"]  # never started, still pending
    log = sandbox.paths.runner_log.read_text()
    assert "1 job(s) not started: their 'after' jobs are not done" in log
    assert "BLOCKED mpc: waits for fc (failed)" in log
    # A later run with the dependency still failed runs nothing; it must not exit 0.
    assert main(sandbox.args("run", "--jobs", "2")) == runner_mod.EXIT_BLOCKED
    assert "mpc" not in read_state(sandbox.paths)["jobs"]


def test_a_dependency_outside_only_blocks_with_its_own_exit_code(
    sandbox: Sandbox, eight_cpus: None
) -> None:
    sandbox.write([py_job("fc", "pass"), py_job("mpc", "pass", after=["fc"])])
    assert main(sandbox.args("run", "--only", "mpc")) == runner_mod.EXIT_BLOCKED
    assert read_state(sandbox.paths)["jobs"] == {} or "mpc" not in read_state(sandbox.paths)["jobs"]
    log = sandbox.paths.runner_log.read_text()
    assert "BLOCKED mpc: waits for fc (pending, not selected)" in log
    assert main(sandbox.args("run", "--only", "fc")) == 0
    assert main(sandbox.args("run", "--only", "mpc")) == 0
    assert status_of(sandbox, "mpc") == "done"


def test_memory_cap_keeps_jobs_apart(sandbox: Sandbox, eight_cpus: None) -> None:
    # 4 GiB cap: two 3 GiB jobs never overlap even with free slots; a 5 GiB job never starts.
    trace = sandbox.root / "trace.txt"
    sandbox.write(
        [
            py_job("a", TRACE, str(trace), "a", "0.2", memory="3G"),
            py_job("b", TRACE, str(trace), "b", "0.2", memory="3G"),
            py_job("huge", "pass", memory="5G"),
        ]
    )
    assert main(sandbox.args("run", "--jobs", "4")) == 1
    rows = spans(trace)
    assert rows["b"][0] >= rows["a"][1]
    record = read_state(sandbox.paths)["jobs"]["huge"]
    assert record["status"] == "failed" and "exceeds the queue's cap" in record["reason"]
