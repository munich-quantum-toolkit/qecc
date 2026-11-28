# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the worker pools.

Worker pools: a job over the per-worker memory limit fails with MemoryError inside its own job (instead of being
killed by the kernel and breaking the pool), and the pool keeps working; the pool replaces a worker after a set number of jobs; workers run their native libraries
single-threaded without changing this process's environment.

"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core import pool as pool_module
from mqt.qecc.circuit_synthesis.ft_search._core.pool import (
    THREAD_VARS,
    JobTimeout,
    job_time_limit,
    kill_pool,
    process_pool,
)


def _alloc(gb: float) -> str:
    try:
        b = bytearray(int(gb * 1024**3))
        return f"ok {len(b)}"
    except MemoryError:
        return "MemoryError"


@pytest.mark.skipif(sys.platform != "linux", reason="address-space limit (RLIMIT_AS) on Linux only")
def test_memory_limit_per_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job over the memory limit fails with MemoryError, and the pool keeps working."""
    # a worker reserves ~2.3 GB of address space at start (numpy, scipy, stim, thread arenas; ~50 MB resident)
    monkeypatch.setattr(pool_module, "MEM_GB", 4.0)
    with process_pool(2) as pool:
        assert pool.submit(_alloc, 6.0).result() == "MemoryError"
        assert pool.submit(_alloc, 0.5).result().startswith("ok")  # same pool still usable


def _threads_after_blas() -> tuple[dict, int]:
    a = np.random.default_rng(0).random((1500, 1500))
    (a @ a).sum()  # OpenBLAS starts its thread pool on a large product

    return {v: os.environ.get(v) for v in THREAD_VARS}, len(list(Path("/proc/self/task").iterdir()))


@pytest.mark.skipif(sys.platform != "linux", reason="thread count from /proc")
def test_workers_single_threaded_parent_env_unchanged() -> None:
    """Workers run the native libraries single-threaded and leave this process's environment as it was."""
    before = {v: os.environ.get(v) for v in THREAD_VARS}
    with process_pool(1) as pool:
        env, n = pool.submit(_threads_after_blas).result()
    assert all(x == "1" for x in env.values())
    assert n <= 4  # main thread (+ a few helper threads), not one per core
    assert {v: os.environ.get(v) for v in THREAD_VARS} == before


def _pid_and_limit() -> tuple[int, int]:
    import resource  # ruff: ignore[import-outside-top-level]  (Unix only)

    return os.getpid(), resource.getrlimit(resource.RLIMIT_AS)[0]


@pytest.mark.skipif(sys.platform != "linux", reason="address-space limit (RLIMIT_AS) on Linux only")
def test_workers_replaced_after_task_limit_keep_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool replaces a worker after the task limit, and the new worker has the memory limit."""
    monkeypatch.setattr(pool_module, "MEM_GB", 4.0)
    monkeypatch.setattr(pool_module, "TASKS", 2)
    with process_pool(1) as pool:
        out = [pool.submit(_pid_and_limit).result() for _ in range(6)]
    assert len({pid for pid, _ in out}) == 3  # a new worker every 2 jobs
    assert all(lim == 4 * 1024**3 for _, lim in out)  # each new worker gets the limit too


def test_task_limit_zero_keeps_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a task limit of 0 the pool keeps its worker."""
    monkeypatch.setattr(pool_module, "TASKS", 0)
    with process_pool(1) as pool:
        pids = {pool.submit(os.getpid).result() for _ in range(4)}
    assert len(pids) == 1


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM")
def test_time_limit_raises_job_timeout_past_except_exception() -> None:
    """A job over its limit ends with JobTimeout, which an ``except Exception`` handler does not catch."""

    def sleep_past_the_limit() -> None:
        try:
            time.sleep(5)
        except Exception:  # ruff: ignore[blind-except]
            pytest.fail("JobTimeout caught as Exception")

    with pytest.raises(JobTimeout), job_time_limit(0.2):
        sleep_past_the_limit()


def test_time_limit_zero_means_no_limit() -> None:
    """A time limit of 0 means no limit."""
    with job_time_limit(0):
        pass


def _sleep(seconds: float, ignore_sigterm: bool) -> None:
    if ignore_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(seconds)


@pytest.mark.parametrize("ignore_sigterm", [False, True])
def test_kill_pool_ends_busy_workers(ignore_sigterm: bool) -> None:
    """``kill_pool`` ends workers in the middle of a job, also a worker that ignores ``SIGTERM``."""
    pool = process_pool(2)
    for _ in range(2):
        pool.submit(_sleep, 60, ignore_sigterm)
    time.sleep(1.0)  # the jobs have started
    procs = list(pool._processes.values())  # ruff: ignore[private-member-access]  (the workers to check)
    start = time.monotonic()
    kill_pool(pool)
    for p in procs:
        p.join(5)
        assert not p.is_alive()
    assert time.monotonic() - start < 10
