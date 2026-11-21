"""Durable results: a job is ``done`` only once its files are on disk, and stays checkable.

The incident behind these tests: a host reboot left outputs of jobs the
queue had already recorded ``done`` at 0 bytes, and their dependents failed days later. The
runner must flush every file of a finished job before ``done`` reaches the state file, fail the
job if that flush fails, and refuse to start a dependent whose dependency's files no longer
match the manifest recorded in its ``job.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from expqueue import durable
from expqueue.cli import main
from expqueue.state import read_state
from tests.conftest import Sandbox, py_job

# Writes out.csv (declared), eval.csv.gz (undeclared, like the incident's eval_*.csv.gz) and
# a nested file into the job directory argv[1], and outside.csv into argv[2].
WRITE = (
    "import pathlib, sys; d = pathlib.Path(sys.argv[1]); "
    "(d / 'out.csv').write_text('a,b\\n1,2\\n'); "
    "(d / 'eval.csv.gz').write_bytes(b'x' * 1000); "
    "(d / 'run').mkdir(); (d / 'run' / 'results.json').write_text('{}'); "
    "pathlib.Path(sys.argv[2]).write_text('outside'); print('hello')"
)


def producer(sandbox: Sandbox, job_id: str = "fc") -> dict[str, object]:
    outside = str(sandbox.root / "outside.csv")
    return py_job(job_id, WRITE, "{job_dir}", outside, outputs=["{job_dir}/out.csv", "outside.csv"])


def status_of(sandbox: Sandbox, job_id: str) -> str:
    return str(read_state(sandbox.paths)["jobs"][job_id]["status"])


def test_every_file_is_synced_before_done_is_recorded(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[tuple[Path, str | None]] = []
    real = durable.sync_path

    def spy(path: Path) -> None:
        record = read_state(sandbox.paths)["jobs"].get("fc", {})
        synced.append((Path(path).resolve(), record.get("status")))
        real(path)

    monkeypatch.setattr(durable, "sync_path", spy)
    sandbox.write([producer(sandbox)])
    assert main(sandbox.args("run")) == 0
    assert status_of(sandbox, "fc") == "done"
    job_dir = (sandbox.results / "fc").resolve()
    expected = {
        job_dir / "out.csv",
        job_dir / "eval.csv.gz",  # not declared: flushed all the same
        job_dir / "run" / "results.json",
        job_dir / "stdout.log",
        job_dir / "stderr.log",
        (sandbox.root / "outside.csv").resolve(),  # declared output outside the job dir
        job_dir / "run",
        job_dir,
        sandbox.root.resolve(),
    }
    seen = {p for p, _ in synced}
    assert expected <= seen, expected - seen
    # Every job file was flushed while the state still said "running", i.e. before "done".
    for path, status in synced:
        if path in expected:
            assert status == "running", (path, status)


def test_a_failed_flush_fails_the_job(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    real = durable.sync_path

    def failing(path: Path) -> None:
        if Path(path).name == "eval.csv.gz":
            raise OSError(5, "Input/output error")
        real(path)

    monkeypatch.setattr(durable, "sync_path", failing)
    sandbox.write([producer(sandbox), py_job("mpc", "pass", after=["fc"])])
    assert main(sandbox.args("run")) == 1
    record = read_state(sandbox.paths)["jobs"]["fc"]
    assert record["status"] == "failed"
    assert "outputs not durable" in record["reason"] and "Input/output error" in record["reason"]
    assert sandbox.job_meta("fc")["files"] is None
    assert "mpc" not in read_state(sandbox.paths)["jobs"]  # never started


def test_job_json_records_the_manifest(sandbox: Sandbox) -> None:
    sandbox.write([producer(sandbox)])
    assert main(sandbox.args("run")) == 0
    files = sandbox.job_meta("fc")["files"]
    job_dir = sandbox.results / "fc"
    outside = str((sandbox.root / "outside.csv").resolve())
    assert set(files) == {
        "out.csv",
        "eval.csv.gz",
        "run/results.json",
        "stdout.log",
        "stderr.log",
        outside,
    }  # job.json itself is not in it
    data = (job_dir / "eval.csv.gz").read_bytes()
    assert files["eval.csv.gz"] == {"bytes": 1000, "sha256": hashlib.sha256(data).hexdigest()}
    assert files["stdout.log"]["bytes"] == len("hello\n")
    assert files[outside]["bytes"] == len("outside")


def test_a_rerun_does_not_list_earlier_attempts(sandbox: Sandbox) -> None:
    sandbox.write([producer(sandbox)])
    assert main(sandbox.args("run")) == 0
    assert main(sandbox.args("retry", "fc")) == 0
    assert main(sandbox.args("run")) == 0
    assert (sandbox.results / "fc" / "attempts" / "1" / "out.csv").exists()
    assert not any(k.startswith("attempts") for k in sandbox.job_meta("fc")["files"])


def test_a_dependency_truncated_after_done_blocks_its_dependent(sandbox: Sandbox) -> None:
    sandbox.write([producer(sandbox), py_job("mpc", "pass", after=["fc"])])
    assert main(sandbox.args("run", "--only", "fc")) == 0
    # What the reboot did: the file is still there, with no data.
    (sandbox.results / "fc" / "eval.csv.gz").write_bytes(b"")
    assert main(sandbox.args("run")) == 1
    record = read_state(sandbox.paths)["jobs"]["fc"]
    assert record["status"] == "failed"
    assert "eval.csv.gz: 0 bytes, recorded 1000" in record["reason"]
    assert "mpc" not in read_state(sandbox.paths)["jobs"]  # not started on broken inputs
    log = sandbox.paths.runner_log.read_text()
    assert "LOST fc (needed by mpc)" in log and "BLOCKED mpc: waits for fc (failed)" in log
    # The documented way out: rerun the dependency, then the dependent runs.
    assert main(sandbox.args("retry", "fc")) == 0
    assert main(sandbox.args("run")) == 0
    assert status_of(sandbox, "fc") == status_of(sandbox, "mpc") == "done"


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("same_size", "out.csv: content changed (sha256)"),
        ("delete_output", "outside.csv: missing"),
        ("delete_job_json", "job.json unreadable"),
    ],
)
def test_other_damage_to_a_dependency_blocks_its_dependent(
    sandbox: Sandbox, damage: str, expected: str
) -> None:
    sandbox.write([producer(sandbox), py_job("mpc", "pass", after=["fc"])])
    assert main(sandbox.args("run", "--only", "fc")) == 0
    if damage == "same_size":
        (sandbox.results / "fc" / "out.csv").write_text("a,b\n9,9\n")
    elif damage == "delete_output":
        (sandbox.root / "outside.csv").unlink()
    else:
        (sandbox.results / "fc" / "job.json").unlink()
    assert main(sandbox.args("run")) == 1
    assert expected in read_state(sandbox.paths)["jobs"]["fc"]["reason"]
    assert "mpc" not in read_state(sandbox.paths)["jobs"]


def test_a_dependency_without_a_manifest_still_counts_as_done(sandbox: Sandbox) -> None:
    # Jobs finished before the manifest existed have no "files" in job.json: not checkable.
    sandbox.write([producer(sandbox), py_job("mpc", "pass", after=["fc"])])
    assert main(sandbox.args("run", "--only", "fc")) == 0
    meta_path = sandbox.results / "fc" / "job.json"
    meta = json.loads(meta_path.read_text())
    del meta["files"]
    meta_path.write_text(json.dumps(meta))
    assert main(sandbox.args("run")) == 0
    assert status_of(sandbox, "mpc") == "done"


def test_job_files_skips_symlinks_attempts_and_job_json(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "attempts" / "1").mkdir(parents=True)
    (job / "attempts" / "1" / "old.csv").write_text("old")
    (job / "job.json").write_text("{}")
    (job / "a.csv").write_text("a")
    (job / "link.csv").symlink_to(job / "a.csv")
    assert durable.job_files(job, [job / "a.csv"]) == [job / "a.csv"]
    manifest = durable.sync_job(job, [])
    assert set(manifest) == {"a.csv"}
    assert durable.verify_manifest(job, manifest) == []
