# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Worker pools for the search.

Workers start from a fork server (a fork of a process that has run threaded native code such as HiGHS can dead-lock),
else they are spawned. Each worker has an address-space limit, so a runaway job fails with ``MemoryError`` in its own
job, ends after a number of jobs, and runs the native libraries single-threaded (``THREAD_VARS``).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Never

from .kernels import warm

if TYPE_CHECKING:
    from collections.abc import Generator

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MEM_GB = 16.0  # address-space limit per worker in GB (0 = no limit; ignored where the platform has none)
TASKS = 100  # jobs per worker before the pool replaces it (0 = never)
PRELOAD = [
    "numpy",
    "stim",
    "scipy.optimize",
    "mqt.qecc.circuit_synthesis.ft_search._core.evaluation",
    "mqt.qecc.circuit_synthesis.ft_search._core.solvers",
    "mqt.qecc.circuit_synthesis.ft_search._core.complete",
    "mqt.qecc.circuit_synthesis.ft_search._core.kernels",
]


class JobTimeout(BaseException):
    """A job's wall-clock limit (:func:`job_time_limit`); a ``BaseException`` so that ``except Exception`` lets it pass."""


def _limit_memory(mem_gb: float) -> None:
    try:
        import resource  # ruff: ignore[import-outside-top-level]  (Unix only)
    except ImportError:  # Windows: no address-space limit
        return
    if mem_gb and mem_gb > 0:
        lim = int(mem_gb * 1024**3)
        # e.g. macOS: no address-space limit
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def _start_forkserver() -> None:
    """Start the fork server (if not running) with one native thread per worker; keep this environment."""
    from multiprocessing import forkserver  # ruff: ignore[import-outside-top-level]  (fork-server platforms only)

    old = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update(dict.fromkeys(THREAD_VARS, "1"))
    try:
        forkserver.ensure_running()
    finally:
        for v, x in old.items():
            if x is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = x


def process_pool(workers: int) -> ProcessPoolExecutor:
    """Fork-server pool (spawned workers where the fork server is not available) of ``workers`` processes.

    Each worker has the address-space limit ``MEM_GB`` and is replaced after ``TASKS`` jobs.
    """
    warm()  # compile once here; the workers then load the kernels from numba's disk cache
    if "forkserver" in mp.get_all_start_methods():
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(PRELOAD)
        _start_forkserver()
    else:
        ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(
        workers, mp_context=ctx, initializer=_limit_memory, initargs=(MEM_GB,), max_tasks_per_child=TASKS or None
    )


def kill_pool(pool: ProcessPoolExecutor) -> None:
    """Hard stop: end every worker process (running jobs are lost), then shut the executor down.

    Workers get ``SIGTERM`` first, so that exit handlers (e.g. coverage measurement) can run, and ``SIGKILL`` if they
    are still alive after a second.
    """
    procs = list((getattr(pool, "_processes", None) or {}).values())
    for p in procs:
        with contextlib.suppress(Exception):  # already gone
            p.terminate()
    for p in procs:
        with contextlib.suppress(Exception):
            p.join(1.0)
            if p.is_alive():
                p.kill()
    pool.shutdown(wait=False, cancel_futures=True)


@contextlib.contextmanager
def job_time_limit(seconds: float) -> Generator[None, None, None]:
    """Raise :class:`JobTimeout` in this (worker) process after ``seconds`` (0 = no limit).

    Without ``SIGALRM`` (Windows) there is no limit inside the job; the search still stops at its own deadline.
    """
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(_signum: int, _frame: object) -> Never:
        msg = f"job exceeded {seconds:.0f} s"
        raise JobTimeout(msg)

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
