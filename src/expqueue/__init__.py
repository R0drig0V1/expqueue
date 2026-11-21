"""Serial, resumable runner for heavy batch jobs declared in a tracked ``queue.toml``.

Jobs are declared in a project's ``queue.toml`` (:mod:`expqueue.queuefile`), run one at a
time at ``nice 19`` by :mod:`expqueue.runner`, with state, locks, logs and results outside
git (:mod:`expqueue.state`). Command line: :mod:`expqueue.cli`; guide: README.md.
"""
