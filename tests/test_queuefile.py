"""Validation of the queue file: what is accepted, and every rule that rejects a job."""

from __future__ import annotations

from typing import Any

import pytest

from expqueue.queuefile import QueueError, expand, load_queue, parse_duration, parse_size
from tests.conftest import PY, Sandbox, py_job


def test_valid_queue_with_defaults_and_overrides(sandbox: Sandbox) -> None:
    (sandbox.root / "sub").mkdir()
    sandbox.write(
        [
            py_job(
                "a",
                "pass",
                "{job_dir}/x",
                cwd="sub",
                env={"B": "2", "A": "9"},
                outputs=["{job_dir}/out.csv"],
                estimate="2m",
            ),
            py_job("b", "pass", timeout="3h"),
        ],
        timeout="45m",
        env={"A": "1", "MKL_NUM_THREADS": "1"},
    )
    q = load_queue(sandbox.queue)
    assert q.root == sandbox.root.resolve()
    assert [j.id for j in q.jobs] == ["a", "b"]
    a, b = q.jobs
    assert a.cwd == "sub" and a.argv == (PY, "-c", "pass", "{job_dir}/x")
    assert a.env == {"A": "9", "B": "2", "MKL_NUM_THREADS": "1"}  # job overrides defaults
    assert a.timeout_s == 45 * 60 and a.estimate_s == 120
    assert b.timeout_s == 3 * 3600 and b.estimate_s is None and b.cwd == "."
    assert a.spec_hash != b.spec_hash
    assert q.kill_grace_s == 1


def test_spec_hash_changes_with_the_definition(sandbox: Sandbox) -> None:
    sandbox.write([py_job("a", "pass")])
    first = load_queue(sandbox.queue).jobs[0].spec_hash
    sandbox.write([py_job("a", "pass", estimate="1m")])  # estimate is not part of the spec
    assert load_queue(sandbox.queue).jobs[0].spec_hash == first
    sandbox.write([py_job("a", "pass", "--extra")])
    assert load_queue(sandbox.queue).jobs[0].spec_hash != first


@pytest.mark.parametrize(
    ("job", "message"),
    [
        ({"id": "a", "argv": f"{PY} -c 'pass'"}, "not a single string"),
        ({"id": "a", "argv": []}, "must not be empty"),
        ({"id": "a", "argv": [PY, 3]}, "list of strings"),
        ({"id": "a"}, "argv is required"),
        ({"id": "a", "argv": ["sh", "-c", "echo hi"]}, "not in [queue].programs"),
        ({"id": "a", "argv": ["/bin/" + "sh", "-c", "x"]}, "not in [queue].programs"),
        ({"id": "a", "argv": [PY], "cwd": "../outside"}, "leaves the repository root"),
        ({"id": "a", "argv": [PY], "cwd": "/tmp"}, "relative to the repository root"),
        ({"id": "../a", "argv": [PY]}, "id must match"),
        ({"id": "_queue", "argv": [PY]}, "id must match"),
        ({"id": "a b", "argv": [PY]}, "id must match"),
        ({"id": "a", "argv": [PY], "timout": "1h"}, "unknown key"),
        ({"id": "a", "argv": [PY, "{jobdir}"]}, "unknown placeholder"),
        ({"id": "a", "argv": [PY], "outputs": ["{home}/x"]}, "unknown placeholder"),
        ({"id": "a", "argv": [PY], "env": {"1BAD": "x"}}, "invalid variable name"),
        ({"id": "a", "argv": [PY], "env": {"N": 1}}, "must be a string"),
        ({"id": "a", "argv": [PY], "timeout": "soon"}, "expected a duration"),
        ({"id": "a", "argv": [PY], "timeout": 0}, "must be positive"),
        ({"id": "a", "argv": [PY], "memory": "6GB"}, "expected a size"),
        ({"id": "a", "argv": [PY], "memory": 6}, "expected a size"),
        ({"id": "a", "argv": [PY], "after": "b"}, "not a single string"),
        ({"id": "a", "argv": [PY], "after": ["a"]}, "earlier in the file"),
    ],
)
def test_invalid_jobs_are_rejected(sandbox: Sandbox, job: dict[str, Any], message: str) -> None:
    sandbox.write([job])
    with pytest.raises(QueueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_queue(sandbox.queue)


def test_duplicate_ids_and_missing_timeout(sandbox: Sandbox) -> None:
    sandbox.write([py_job("a", "pass"), py_job("a", "pass")])
    with pytest.raises(QueueError, match="duplicate job id"):
        load_queue(sandbox.queue)
    sandbox.queue.write_text(f'[queue]\nprograms = ["{PY}"]\n[[job]]\nid = "a"\nargv = ["{PY}"]\n')
    with pytest.raises(QueueError, match="no timeout"):
        load_queue(sandbox.queue)


def test_file_level_errors(sandbox: Sandbox) -> None:
    with pytest.raises(QueueError, match="not found"):
        load_queue(sandbox.queue)
    sandbox.queue.write_text("[queue\n")
    with pytest.raises(QueueError, match="invalid TOML"):
        load_queue(sandbox.queue)
    sandbox.queue.write_text('[queue]\nprograms = ["x"]\n[jobs]\n')
    with pytest.raises(QueueError, match="unknown key"):
        load_queue(sandbox.queue)
    sandbox.queue.write_text('[queue]\nroot = "missing"\nprograms = ["x"]\n')
    with pytest.raises(QueueError, match="not a directory"):
        load_queue(sandbox.queue)


def test_durations_and_expansion() -> None:
    assert parse_duration("90s", "t") == 90
    assert parse_duration("1.5h", "t") == 5400
    assert parse_duration(12, "t") == 12
    for bad in ("1d", "", True, "-1m"):
        with pytest.raises(QueueError):
            parse_duration(bad, "t")
    values = {"job_dir": "/r/a", "repo": "/repo", "results_dir": "/r"}
    assert expand("{job_dir}/x;{repo} $(id) {results_dir}", values) == "/r/a/x;/repo $(id) /r"
    assert expand("f'{{x}}' {{job_dir}} {0} { y }", values) == "f'{x}' {job_dir} {0} { y }"
    with pytest.raises(QueueError):
        expand("{nope}", values)


def test_memory_and_after(sandbox: Sandbox) -> None:
    sandbox.queue.write_text(
        f'[queue]\nprograms = ["{PY}"]\n[defaults]\ntimeout = "1m"\nmemory = "2G"\n'
        f'[[job]]\nid = "a"\nargv = ["{PY}"]\n'
        f'[[job]]\nid = "b"\nargv = ["{PY}"]\nmemory = "512M"\nafter = ["a"]\n'
    )
    a, b = load_queue(sandbox.queue).jobs
    assert (a.memory_bytes, a.after) == (2 * 2**30, ())
    assert (b.memory_bytes, b.after) == (512 * 2**20, ("a",))
    # scheduling keys do not change the job's definition hash
    sandbox.write([py_job("a", "pass"), py_job("b", "pass")])
    plain = load_queue(sandbox.queue).jobs[1].spec_hash
    sandbox.write([py_job("a", "pass"), py_job("b", "pass", memory="1G", after=["a"])])
    assert load_queue(sandbox.queue).jobs[1].spec_hash == plain
    # a dependency on a later job (or a typo) is refused: dependencies follow the file order
    sandbox.write([py_job("a", "pass", after=["b"]), py_job("b", "pass")])
    with pytest.raises(QueueError, match="earlier in the file"):
        load_queue(sandbox.queue)
    assert parse_size("1.5K", "m") == 1536
    for bad in ("0G", "1T", "", "2 g"):
        with pytest.raises(QueueError):
            parse_size(bad, "m")
