"""Micro-batching request queue, so several users are not serialised.

WHY THIS EXISTS
---------------
Measured on the production light model (2051 training rows, RTX 5060):

    1 spectrum   -> 58.8 s
    297 spectra  -> 61.5 s

Predicting 297 costs 3 seconds more than predicting 1. TabICL is an in-context
learner: nearly all of the time is spent re-reading the whole training set, and
the marginal cost of an extra test row is about 9 milliseconds.

Served naively, five users pressing the button at once take 5 x 59s = ~5
minutes, because each request waits for the last. Batched, the same five
requests ride along in ONE model call and finish in ~59 seconds together.
Twenty users would also be ~59 seconds. The queue turns a per-user cost into a
per-batch cost, which is what the hardware actually charges.

This is not a workaround for slow inference; it is the shape the cost function
already has. Making inference faster (see the single-pass decoder work) shrinks
the constant but does not change the argument.

DESIGN
------
Requests are accepted immediately and given a job id. A single worker thread
owns the GPU: it takes the first queued job, waits a short window for others to
arrive, then runs everything compatible as one batch. Only one batch is ever in
flight, so the GPU is never oversubscribed and memory use is bounded.

Jobs are grouped by BACKING MODE, because different modes load different model
artifacts. Mixing them in one call is not possible; each group gets its own.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

# How long to hold the first arrival while waiting for company. Long enough to
# gather a burst of users clicking at once, short enough that a lone user does
# not notice: 1.5s against a ~60s prediction is under 3% added latency.
BATCH_WINDOW_S = 1.5

# Cap on one batch. The marginal cost per row is tiny, but GPU memory is not
# free and a runaway queue should degrade into several batches rather than one
# enormous allocation.
MAX_BATCH = 16

# Finished jobs are kept this long so a client that polls slowly can still
# collect its answer, then dropped so the process does not leak.
RESULT_TTL_S = 900


@dataclass
class Job:
    id: str
    r_light: list | None
    r_dark: list | None
    mode: str
    submitted: float = field(default_factory=time.time)
    status: str = "queued"          # queued | running | done | error
    result: dict | None = None
    error: str | None = None
    finished: float | None = None


class PredictionQueue:
    """Accepts spectra, returns job ids, runs them in shared batches.

    predict_batch(mode, specs) -> list[dict]
        Given a backing mode and a list of (r_light, r_dark) tuples, return one
        result dict per input, in the same order. Raising is allowed; the
        exception is recorded against every job in that batch.
    """

    def __init__(
        self,
        predict_batch: Callable[[str, list], list],
        mode_of: Callable[[list | None, list | None], str],
        batch_window_s: float = BATCH_WINDOW_S,
        max_batch: int = MAX_BATCH,
    ):
        self._predict_batch = predict_batch
        self._mode_of = mode_of
        self._window = batch_window_s
        self._max_batch = max_batch

        self._lock = threading.Lock()
        self._pending: deque[Job] = deque()
        self._jobs: dict[str, Job] = {}
        self._wake = threading.Event()
        self._worker = threading.Thread(target=self._run, name="predict-worker", daemon=True)
        self._worker.start()

    # ---- public API -----------------------------------------------------

    def submit(self, r_light, r_dark) -> str:
        job = Job(id=uuid.uuid4().hex[:12], r_light=r_light, r_dark=r_dark,
                  mode=self._mode_of(r_light, r_dark))
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append(job)
        self._wake.set()
        return job.id

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            out = {"job_id": job.id, "status": job.status,
                   "waited_s": round(time.time() - job.submitted, 1)}
            if job.status == "queued":
                # Position is informational only. Report it so the page can say
                # "3rd in line" rather than showing an unexplained wait -- the
                # single most common reason a user assumes a tool has hung.
                try:
                    out["position"] = list(self._pending).index(job) + 1
                except ValueError:
                    out["position"] = 1
            elif job.status == "done":
                out.update(job.result or {})
            elif job.status == "error":
                out["error"] = job.error
            return out

    def wait(self, job_id: str, timeout: float = 900.0) -> dict:
        """Block until a job finishes, then return its result.

        This is what lets the HTTP API keep its original synchronous shape: the
        caller still gets a recipe back from one request, with no job ids and no
        polling, so the existing front end needs no change. The batching gain is
        unaffected -- several callers blocking at once are exactly what the
        worker coalesces into a single model call.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return {"error": "unknown job"}
                if job.status == "done":
                    return dict(job.result or {})
                if job.status == "error":
                    return {"error": job.error}
            time.sleep(0.1)
        return {"error": f"timed out after {timeout:.0f}s"}

    def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    # ---- worker ---------------------------------------------------------

    def _collect(self) -> list[Job]:
        """Take the first pending job, then hold briefly for others.

        The wait happens OUTSIDE the lock so submissions are never blocked by
        the batching window.
        """
        with self._lock:
            if not self._pending:
                return []
            first = self._pending[0]

        deadline = time.time() + self._window
        while time.time() < deadline:
            with self._lock:
                same_mode = [j for j in self._pending if j.mode == first.mode]
                if len(same_mode) >= self._max_batch:
                    break
            time.sleep(0.05)

        with self._lock:
            batch = [j for j in self._pending if j.mode == first.mode][: self._max_batch]
            for j in batch:
                self._pending.remove(j)
                j.status = "running"
            return batch

    def _reap(self) -> None:
        cutoff = time.time() - RESULT_TTL_S
        with self._lock:
            stale = [k for k, j in self._jobs.items()
                     if j.finished is not None and j.finished < cutoff]
            for k in stale:
                del self._jobs[k]

    def _run(self) -> None:
        while True:
            batch = self._collect()
            if not batch:
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                self._reap()
                continue

            specs = [(j.r_light, j.r_dark) for j in batch]
            try:
                results = self._predict_batch(batch[0].mode, specs)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"batch size mismatch: {len(batch)} in, {len(results)} out"
                    )
                for job, res in zip(batch, results):
                    job.result, job.status = res, "done"
            except Exception as exc:                      # noqa: BLE001
                # One failure fails its whole batch. That is the honest
                # behaviour: the batch shares a single model call, so there is
                # no way to know which input was at fault without re-running
                # them individually, and silently returning partial results
                # would be worse than a clear error.
                for job in batch:
                    job.error, job.status = f"{type(exc).__name__}: {exc}", "error"
            finally:
                now = time.time()
                for job in batch:
                    job.finished = now
