"""The required, generic settings: the results directory, ``{data_dir}`` and the default queue.

Neither setting has a built-in default: the results directory holds ``runner.lock``, so a
shared default would make two projects on one machine share a runner, and a data directory
guessed by the queue would hide a missing setting until a job reads the wrong files.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from expqueue.cli import find_default_queue, main
from expqueue.queuefile import QueueError, expand, uses_placeholder
from expqueue.state import read_state
from tests.conftest import Sandbox, py_job

ECHO = "import sys; open(sys.argv[1], 'w').write(sys.argv[2])"


def test_results_dir_is_required(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    sandbox.write([py_job("a", "pass")])
    for command in (["run"], ["status"], ["list"], ["stop"], ["retry", "a"], ["skip", "a"]):
        assert main(["--queue", str(sandbox.queue), *command]) == 2
        assert "EXPQ_RESULTS_DIR" in capsys.readouterr().err
    assert not sandbox.results.exists()


def test_results_dir_from_the_environment(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox.write([py_job("a", "pass")])
    monkeypatch.setenv("EXPQ_RESULTS_DIR", str(sandbox.results))
    assert main(["--queue", str(sandbox.queue), "run", "--poll", "0.05"]) == 0
    assert read_state(sandbox.paths)["jobs"]["a"]["status"] == "done"
    # the option wins over the variable
    other = tmp_path / "other"
    assert main(["--queue", str(sandbox.queue), "--results-dir", str(other), "status"]) == 0
    assert not (other / "a").exists()


def test_data_dir_is_required_when_used(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox.write(
        [py_job("plain", "pass"), py_job("reader", ECHO, "{job_dir}/out.txt", "{data_dir}/in")]
    )
    assert main(sandbox.args("run")) == 2  # refused before any job starts
    assert "EXPQ_DATA_DIR" in capsys.readouterr().err
    assert read_state(sandbox.paths)["jobs"] == {}
    # a run that does not select the job needs no data directory
    assert main(sandbox.args("run", "--only", "plain")) == 0
    # ... unless the provenance of every job needs it
    sandbox.write([py_job("plain2", "pass")], provenance=["{data_dir}/MANIFEST.json"])
    assert main(sandbox.args("run")) == 2
    # reading the queue never needs it
    assert main(sandbox.args("list")) == 0
    assert main(sandbox.args("status")) == 0


def test_data_dir_expands_in_argv_and_provenance(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "MANIFEST.json").write_text("{}\n")
    monkeypatch.setenv("EXPQ_DATA_DIR", str(data))
    sandbox.write(
        [py_job("reader", ECHO, "{job_dir}/out.txt", "{data_dir}/in")],
        provenance=["{data_dir}/MANIFEST.json"],
    )
    assert main(sandbox.args("run")) == 0
    assert (sandbox.results / "reader" / "out.txt").read_text() == f"{data}/in"
    assert sandbox.job_meta("reader")["provenance_sha256"] == {
        f"{data}/MANIFEST.json": hashlib.sha256(b"{}\n").hexdigest()
    }


def test_a_job_added_mid_run_without_its_data_dir_fails(sandbox: Sandbox) -> None:
    """The queue file is reloaded before each start, after the check made when the run began."""
    addition = (
        f'\n[[job]]\nid = "late"\nargv = ["{sys.executable}", "-c", "pass", "{{{{data_dir}}}}/x"]\n'
    )
    # Escaped braces: the first job's own argument reaches it as a literal "{data_dir}".
    code = "import sys; open(sys.argv[1], 'a').write(sys.argv[2])"
    sandbox.write([py_job("first", code, str(sandbox.queue), addition)])
    assert main(sandbox.args("run")) == 1
    late = read_state(sandbox.paths)["jobs"]["late"]
    assert late["status"] == "failed" and "EXPQ_DATA_DIR" in late["reason"]
    assert not (sandbox.results / "late" / "job.json").exists()  # it never started


def test_expand_and_uses_placeholder() -> None:
    with pytest.raises(QueueError, match="EXPQ_DATA_DIR"):
        expand("{data_dir}/x", {"job_dir": "/r/a", "repo": "/repo", "results_dir": "/r"})
    assert expand("{data_dir}/x", {"data_dir": "/d"}) == "/d/x"
    assert uses_placeholder(["a", "{data_dir}/x"], "data_dir")
    assert not uses_placeholder(["{{data_dir}}", "{job_dir}"], "data_dir")


def test_default_queue_is_the_nearest_queue_toml(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox.write([py_job("a", "pass")])
    below = sandbox.root / "pkg" / "sub"
    below.mkdir(parents=True)
    assert find_default_queue(below) == sandbox.queue
    with pytest.raises(QueueError, match="--queue"):
        find_default_queue(tmp_path / "results")
    monkeypatch.chdir(below)
    assert main(["--results-dir", str(sandbox.results), "run", "--poll", "0.05"]) == 0
    assert read_state(sandbox.paths)["jobs"]["a"]["status"] == "done"
