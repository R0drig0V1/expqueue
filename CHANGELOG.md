# Changelog

Releases are pinned by commit SHA (`uv tool install git+https://github.com/R0drig0V1/expqueue@<sha>`).

## v0.1.0 (2026-11-21)

First release as a repository of its own. The code was extracted from a research project
that used it as an internal package, with a fresh history. The behaviour of the runner is
unchanged; the settings that were specific to that project became generic and required.

### Renamed or changed settings (action needed when switching from the internal package)

| Before | Now |
| --- | --- |
| Results directory from a project-specific environment variable | `$EXPQ_RESULTS_DIR` (or `--results-dir`, unchanged) |
| Default results directory under `~/.cache/<project>/results` when nothing was set | **No default.** `expq` exits 2 unless `--results-dir` or `$EXPQ_RESULTS_DIR` is set |
| `{data_dir}` from a project-specific environment variable | `{data_dir}` from `$EXPQ_DATA_DIR` |
| Default data directory under `~/.cache/<project>` when nothing was set | **No default.** `expq run` exits 2, before any job starts, if a selected job or `provenance` uses `{data_dir}` and `$EXPQ_DATA_DIR` is not set. A queue that never uses `{data_dir}` needs nothing |
| Default queue file: `experiments/queue.toml` of the nearest parent | `queue.toml` of the current directory or of the nearest parent that has one (`--queue` unchanged) |

To keep an existing results directory, its state and its lock, set `EXPQ_RESULTS_DIR` to the
path that was used before. The layout of the results directory, `state.json` (version 1) and
`job.json` did not change, so a queue in progress resumes where it was.

### Renamed in the Python API

| Before | Now |
| --- | --- |
| `expqueue.state.default_results_dir()` (always a path) | `expqueue.state.results_dir_from_env()` (`None` when unset); `expqueue.cli.resolve_results_dir()` raises `QueueError` when nothing is set |
| `expqueue.runner.default_data_dir()` (always a path) | `expqueue.runner.data_dir_from_env()` (`None` when unset) |

### Added

- `expqueue.queuefile.PLACEHOLDER_ENV` and `uses_placeholder()`.
- `expand()` raises `QueueError` for a known placeholder that has no value.
- A job added to the queue file while the runner works, and that needs `{data_dir}` without
  `$EXPQ_DATA_DIR`, is recorded `failed` without starting.

### Unchanged

The queue file format, the commands and their options, the exit codes, the capacity cap, the
resume rules, durable outputs and the dependency check.

### Not included

Machine-wide CPU, memory and GPU ledgers shared by every runner on a machine: planned for
v0.2.0. Until then the capacity cap is per runner.
