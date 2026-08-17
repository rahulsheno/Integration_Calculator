"""Shared compute helpers for the advanced-calculus engines.

The sympy routines powering ODEs, transforms, and series can run
indefinitely on pathological inputs (dsolve on a nasty nonlinear ODE,
high-order expansions, Fourier coefficients of complicated functions).
This module wraps them so the solver always either finishes or fails fast
with an honest timeout message, instead of hanging the HTTP request.

Windows has no SIGALRM-style signal timers, so timeouts are implemented
with a small worker-thread pool. Python threads can't be force-killed, so
an over-budget worker is simply abandoned: its result is discarded and the
caller reports a clean timeout. Because abandoned workers permanently
occupy pool slots, the submission queue is BOUNDED: when every worker is
busy and the queue already holds `QUEUE_LIMIT` pending tasks, new requests
fail immediately with ComputeTimeout instead of piling up without limit.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

_MAX_WORKERS = 4
# Pending (queued, not yet started) tasks allowed beyond the running workers.
# Keeps worst-case pile-up at _MAX_WORKERS + QUEUE_LIMIT computations.
_QUEUE_LIMIT = 8

_COMPUTE_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="calculus-compute")
_PENDING_SLOTS = threading.BoundedSemaphore(_QUEUE_LIMIT)


class ComputeTimeout(Exception):
    """Raised when a symbolic computation exceeds its time budget, or when
    the compute pool is saturated and the job can't even be queued."""


def run_with_timeout(func, *args, seconds: float = 30, **kwargs):
    """Run func(*args, **kwargs), returning its result or raising
    ComputeTimeout if it doesn't finish within `seconds`. Also raises
    ComputeTimeout immediately if the pool's bounded queue is full, so a
    flood of pathological inputs degrades gracefully (fast honest errors)
    rather than growing an unbounded backlog."""
    if not _PENDING_SLOTS.acquire(blocking=False):
        raise ComputeTimeout(
            "The calculation server is temporarily saturated by other complex "
            "requests. Please try again in a moment."
        )
    future = _COMPUTE_POOL.submit(func, *args, **kwargs)
    future.add_done_callback(lambda f: _PENDING_SLOTS.release())
    try:
        return future.result(timeout=seconds)
    except TimeoutError:
        future.cancel()
        raise ComputeTimeout(
            f"The symbolic computation did not finish within {seconds:g} seconds. "
            "This problem may need a technique this solver doesn't yet support, "
            "or the expression may be too complex for closed-form evaluation."
        )
