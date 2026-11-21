# expqueue

`expq` runs the heavy batch jobs of a project one after another, at low priority, from a
`queue.toml` that is tracked in the project's repository. It can be stopped and resumed, it
survives a crash of the runner or of the machine, and it records what each job ran and
produced. Python standard library only; Linux and macOS.

It is meant for long, unattended work (trainings, solver runs, evaluations, batch model
jobs) on a machine that people or other agents also use: the queue takes at most half of it.

## Install

Pin a commit, not a tag (a tag can be moved):

```bash
uv tool install git+https://github.com/R0drig0V1/expqueue@<sha>
```

`<sha>` is the full commit SHA of the release you want. The releases and their SHAs are in
[CHANGELOG.md](CHANGELOG.md); `git ls-remote https://github.com/R0drig0V1/expqueue 'refs/tags/v*'`
prints them too. Python 3.12 or later. The package has no dependencies.

## Quick start

```bash
export EXPQ_RESULTS_DIR=~/.cache/my-project/results   # required, one per project
cd my-project                                          # holds queue.toml (or a parent does)
expq list          # the jobs, in order, with their estimates
expq run           # run everything left, one job at a time, nice 19
expq status        # progress (--json for machines)
```

## Settings

There are two, and neither has a built-in default.

| Setting | How | Required |
| --- | --- | --- |
| Results directory | `--results-dir DIR`, else `$EXPQ_RESULTS_DIR` | always |
| Data directory | `$EXPQ_DATA_DIR`, the value of the `{data_dir}` placeholder | when a selected job or `provenance` uses `{data_dir}` |

- **The results directory holds the runner lock and the state**, so it decides what "one
  runner" means. Give each project its own. `expq` refuses to guess one: a default shared by
  two projects would make them wait on each other's lock and mix their state.
- It must be **outside the repository** (`[queue].root`); `expq run` refuses a directory
  inside it.
- `expq run` fails with exit code 2, before any job starts, if the queue needs `{data_dir}`
  and `$EXPQ_DATA_DIR` is not set.
- The queue file is `--queue PATH`, else the `queue.toml` of the current directory or of the
  nearest parent that has one.

## The queue file

`queue.toml` is the only source of commands. `expq` has no way to run a command given on the
command line, so what runs is what was reviewed and committed.

```toml
[queue]
root = "."                       # repository root, relative to this file (default ".")
programs = ["python", "uv"]      # allowed argv[0] values (required, not empty)
kill_grace = "30s"               # SIGTERM -> SIGKILL delay on timeout or stop (default 30s)
provenance = ["uv.lock"]         # files whose sha256 every job.json records

[defaults]                       # inherited by every job, overridable per job
timeout = "1h"
memory = "2G"
[defaults.env]
OMP_NUM_THREADS = "1"

[[job]]
id = "train-a-60"                # unique; names the job's results directory
cwd = "."                        # relative to root, must stay inside it (default ".")
argv = ["uv", "run", "python", "scripts/train.py", "--out", "{job_dir}"]
env = { SEED = "0" }             # merged over defaults.env
outputs = ["{job_dir}/summary.csv"]
timeout = "30m"                  # required here or in [defaults]
estimate = "8m"                  # optional, for status / list
memory = "6G"                    # optional peak-memory estimate
after = ["other-job"]            # optional: start only once these earlier jobs are done
```

Rules, all checked when the file is loaded (a failure names the job and the key):

- `argv` is a list of strings, executed without a shell. A single string is rejected. Quotes,
  `;`, `|`, `$()` and globs inside an argument reach the program literally.
- `argv[0]` must be listed in `[queue].programs`.
- Unknown keys are errors, so a typo fails loudly.
- `id` matches `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and is unique.
- `cwd` is relative and stays inside `root` after resolving `..` and symlinks.
- Durations are `"90s"`, `"45m"`, `"2h"` or a positive number of seconds. Sizes are `"800K"`,
  `"512M"`, `"6G"` (binary units).
- `after` names jobs that come earlier in the file, so dependencies follow the file order and
  cannot form a cycle.
- Relative `outputs` are relative to the job's `cwd`; relative `provenance` paths to `root`.

Placeholders, replaced by plain string substitution in `argv`, `env` values, `outputs` and
`provenance`:

| Placeholder | Value |
| --- | --- |
| `{job_dir}` | `<results>/<job id>`, the job's own directory |
| `{repo}` | `[queue].root`, resolved |
| `{results_dir}` | the results directory |
| `{data_dir}` | `$EXPQ_DATA_DIR` |

Any other `{name}` is an error. `{{` and `}}` are literal braces.

## Commands

```bash
expq list                      # the jobs of queue.toml, in order, with estimates
expq run                       # run everything left
expq run --only 'train-*'      # only the jobs whose id matches a glob (repeatable)
expq run --jobs 2              # two at a time (at most half the CPUs)
expq status [--json]           # progress and estimated remaining time
expq retry train-a-60          # failed / timed-out / done -> pending
expq retry --failed            # every failed or timed-out job -> pending
expq skip eval-b               # never run it (until retried)
expq stop                      # drain: finish the running job, start no new one
expq stop --now                # SIGTERM the runner: jobs are interrupted and rerun later
```

`python -m expqueue` is the same as `expq`.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | ok |
| 1 | at least one job failed or timed out in this run |
| 2 | bad arguments, a missing setting, an invalid queue file or an unknown job id |
| 3 | another runner is active, or the job is running |
| 4 | stopped on request (`stop`) with jobs left |
| 5 | nothing failed, but selected jobs were left waiting for their `after` jobs |
| 130 | interrupted by a signal |

## What is in the results directory

```
<results>/_queue/state.json    per-job status (replaced atomically, readable at any time)
<results>/_queue/state.lock    lock held only while the state is rewritten
<results>/_queue/runner.lock   lock held by the one running runner; contains its pid
<results>/_queue/STOP          drain request
<results>/_queue/runner.log    the runner's progress lines
<results>/<job id>/            stdout.log, stderr.log, job.json and the job's own outputs
<results>/<job id>/attempts/   what earlier attempts left, one directory per attempt
```

`job.json` records the exact argv, the working directory, the environment overrides, the git
commit and dirty flag of `root`, the sha256 of the `provenance` files, the host, the times,
the exit code or signal, the peak resident memory and the size and sha256 of every file the
job wrote. Job statuses: `pending`, `running`, `done`, `failed`, `timeout`, `interrupted`,
`skipped`.

## The runner's guarantees

- **One runner per results directory.** The lock is a kernel `flock`, released when its
  holder dies, so a crash or a reboot never leaves a stale lock.
- **No shell, no ad-hoc commands.** Jobs come only from the queue file and run as argv lists
  with programs from the allowlist.
- **Low priority and a capacity cap.** Jobs run at `nice 19`, one at a time by default.
  `--jobs` is refused above half the CPUs. A job does not start while the `memory` estimates
  of the running jobs plus its own exceed half the physical memory, and a job whose estimate
  alone exceeds it fails without starting. The cap counts this runner's jobs only.
- **Strict file order.** Jobs start in the order of the file; a job waiting for memory is not
  overtaken. The file is read again before each start, so jobs can be appended while the
  runner works. If it is invalid at that moment, the runner keeps the last valid version.
- **Resuming.** `done` and `skipped` jobs are never run again. `failed` and `timeout` jobs
  are not retried on their own (`expq retry`). A job that was running when the runner died
  is run again from the start. Its orphaned process group is killed first, only when it is
  provably the same process: same boot, the pid leads its group, the start time matches.
- **Clean attempts.** Before each attempt, whatever an earlier one left in the job directory
  moves to `attempts/<n>/`, so old outputs can never make a new attempt look successful.
- **Timeouts.** SIGTERM to the job's process group, SIGKILL after `kill_grace`. The process
  group is also killed when the job ends, so nothing it left behind competes with the next.
- **`done` means on disk.** A job is `done` only if it exits 0, every declared output was
  written during this attempt, and all its files and their directories were flushed to disk
  (`F_FULLFSYNC` on macOS) before `done` reaches the state file. Exit 0 with missing or stale
  outputs, or a failed flush, is `failed`.
- **Dependencies are checked.** Before a job starts, the files of its `after` jobs are
  compared with the manifest in their `job.json`. A dependency whose files changed or
  vanished becomes `failed`, and its dependents are blocked instead of reading broken inputs.
- **Signals.** On SIGINT, SIGTERM or SIGHUP the runner stops its jobs, marks them
  `interrupted` (the next run reruns them) and exits 130. A second signal kills at once.
- **Clean environment.** A job does not inherit the runner's own virtual environment
  (`VIRTUAL_ENV` and its `bin` on `PATH` are removed). Standard input is `/dev/null`.

What it does not do:

- No checkpoints. An interrupted job loses its partial work.
- The capacity cap is per runner. Two runners with different results directories each take up
  to half of the machine. Machine-wide CPU, memory and GPU ledgers are planned for v0.2.0.
- `peak_rss_bytes` is the largest single process of the job's tree, not the sum.
- Timings measured with `--jobs N > 1` are not comparable with single-job estimates.

## Development

```bash
uv sync
uv run pytest                                   # offline, dummy jobs only, a few seconds
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Licence

MIT, see [LICENSE](LICENSE).
