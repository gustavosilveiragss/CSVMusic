"""Qt-free scheduler for per-track download jobs, with an inline mode for a single worker."""
from __future__ import annotations

import queue
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping

from .concurrency import AdaptiveLimit, hard_cap
from .downloader import (
	YOUTUBE_MITIGATION_AGGRESSIVE,
	CancelToken, DownloadCancelled, YouTubeMitigationProfile, YtdlpPacer, lane_pacer_gap, pacer_gap_for_profile,
	pacer_scope, process_scope, scale_profile_for_concurrency,
)
from .log import log
from .ytmusic_match import RATE_LIMIT_S

MATCH_GAP_S = RATE_LIMIT_S  # floor between two searches; read as download_pool.MATCH_GAP_S so tests can zero it

_MAX_REQUESTED = 8


def concurrency_limit(
	requested: int | None, profile: YouTubeMitigationProfile, *, in_process_ytdlp: bool,
	cpu_count: int | None,
) -> int:
	"""Hard ceiling for the adaptive controller; the pacer, not this cap, rate-limits requests."""
	if in_process_ytdlp or profile.label == YOUTUBE_MITIGATION_AGGRESSIVE.label:
		return 1
	cap = hard_cap(cpu_count) if requested is None else max(1, min(requested, _MAX_REQUESTED))
	return cap


def track_pause_s(profile: YouTubeMitigationProfile, *, uniform: Callable[[float, float], float] = random.uniform) -> float:
	if profile.track_sleep_s <= 0:
		return MATCH_GAP_S
	jitter = min(1.0, profile.track_sleep_s * 0.2)
	return max(MATCH_GAP_S, uniform(profile.track_sleep_s - jitter, profile.track_sleep_s + jitter))


@dataclass(frozen=True)
class PoolPolicy:
	requested: int | None
	in_process_ytdlp: bool = False
	cpu_count: int | None = None
	lane_limits: Mapping[str, int] = field(default_factory=dict)  # per-lane ceiling; a lane absent here is uncapped

	def max_limit(self, profile: YouTubeMitigationProfile) -> int:
		return concurrency_limit(self.requested, profile, in_process_ytdlp=self.in_process_ytdlp, cpu_count=self.cpu_count)


@dataclass(frozen=True)
class JobEvent:
	index: int
	kind: str
	args: tuple


@dataclass(frozen=True)
class Completion:
	index: int
	item: Any
	result: Any
	error: BaseException | None
	cancelled: bool


class JobContext:
	def __init__(self, index: int, out: "queue.Queue[JobEvent | Completion]", parallel: bool) -> None:
		self._index = index
		self._out = out
		self.parallel = parallel

	def emit(self, kind: str, *args: Any) -> None:
		self._out.put(JobEvent(self._index, kind, args))


_WAKE = object()  # queue marker that cancel() uses to wake a pump that is waiting


class _InlineExecutor:
	"""Runs the job on the caller's thread, so one code path serves both modes."""

	def submit(self, fn: Callable[..., Any], *args: Any) -> None:
		fn(*args)

	def shutdown(self, wait: bool = True) -> None:
		pass


class DownloadPool:
	"""The pool owns the mitigation profile, the cancel state and every wait of its coordinator."""

	def __init__(
		self, run: Callable[[Any, JobContext], Any], policy: PoolPolicy, profile: YouTubeMitigationProfile,
		*, clock: Callable[[], float] = time.monotonic,
	) -> None:
		self._run = run
		self._policy = policy
		self._profile = profile
		self._lock = threading.Lock()
		self._token = CancelToken()
		self._out: "queue.Queue[JobEvent | Completion]" = queue.Queue()
		self._ready: deque[tuple[int, Any, Hashable, str | None]] = deque()
		self._in_flight: dict[int, tuple[Hashable, str | None]] = {}
		self._closed = False
		self._cancelled = False
		ceiling = policy.max_limit(profile)
		# The adaptive controller owns the limit and is only touched under self._lock (it has no lock of its own).
		self._adaptive = AdaptiveLimit(start=min(3, ceiling), ceiling=ceiling, clock=clock)
		self._parallel = ceiling > 1
		self._pacer = YtdlpPacer(*pacer_gap_for_profile(profile), cancel=self._token) if self._parallel else None
		self._lane_pacers = {
			lane: YtdlpPacer(*gap, cancel=self._token) for lane in policy.lane_limits
			if self._parallel and (gap := lane_pacer_gap(lane)) is not None
		}
		self._executor: Any = ThreadPoolExecutor(max_workers=ceiling) if self._parallel else _InlineExecutor()

	def __enter__(self) -> "DownloadPool":
		return self

	def __exit__(self, *exc: Any) -> None:
		self.cancel()
		self._executor.shutdown(wait=True)

	@property
	def profile(self) -> YouTubeMitigationProfile:
		with self._lock:
			return self._profile

	@property
	def limit(self) -> int:
		with self._lock:
			return self._adaptive.level

	@property
	def parallel(self) -> bool:
		return self._parallel

	@property
	def cancelled(self) -> bool:
		return self._cancelled

	def submit(self, index: int, item: Any, *, key: Hashable, priority: bool = False, lane: str | None = None) -> bool:
		# close() stops new tracks but still takes a priority retry of an accepted one (a 429 in flight
		# when the search aborted must not lose its safe-mode retry); cancel() rejects everything.
		with self._lock:
			if self._cancelled or (self._closed and not priority):
				return False
			entry = (index, item, key, lane)
			if priority:
				self._ready.appendleft(entry)
			else:
				self._ready.append(entry)
			return True

	def has_capacity(self) -> bool:
		with self._lock:
			return not self._closed and len(self._in_flight) + self._dispatchable_ready() < self._adaptive.level

	def _dispatchable_ready(self) -> int:
		"""Invariant: a ready entry held back only by its full lane occupies no pool capacity. Caller holds the lock."""
		limits = self._policy.lane_limits
		room = {lane: limit - sum(1 for _k, l in self._in_flight.values() if l == lane) for lane, limit in limits.items()}
		count = 0
		for entry in self._ready:
			lane = entry[3]
			if lane in room:
				if room[lane] <= 0:
					continue
				room[lane] -= 1
			count += 1
		return count

	def idle(self) -> bool:
		with self._lock:
			return not self._ready and not self._in_flight and self._out.empty()

	def apply_profile(self, profile: YouTubeMitigationProfile) -> None:
		with self._lock:
			self._profile = profile
			self._adaptive.lower_ceiling(self._policy.max_limit(profile))
		if self._pacer is not None:
			self._pacer.set_gap(*pacer_gap_for_profile(profile))

	def job_profile(self) -> YouTubeMitigationProfile:
		with self._lock:
			return scale_profile_for_concurrency(self._profile, self._adaptive.level)

	def close(self) -> None:
		with self._lock:
			self._closed = True

	def cancel(self) -> None:
		"""Never blocks the caller (the GUI thread): taskkill is synchronous on Windows, so the kills run on a daemon thread."""
		with self._lock:
			self._closed = True
			first = not self._cancelled
			self._cancelled = True
			dropped = list(self._ready)
			self._ready.clear()
			for index, item, _key, _lane in dropped:
				self._out.put(Completion(index, item, None, None, True))
			self._out.put(_WAKE)
		if first:
			threading.Thread(target=self._token.cancel, daemon=True).start()

	def pump(self, timeout: float) -> list[JobEvent | Completion]:
		"""Return queued events, waiting up to `timeout` for one even when idle; cancel() wakes the wait."""
		self._dispatch()
		out: list[Any] = []
		self._drain(out)
		if not out and timeout > 0 and not (self._cancelled and not self._has_work()):
			try:
				out.append(self._out.get(timeout=timeout))
			except queue.Empty:
				return []
			self._drain(out)
		return [self._collect(msg) for msg in out if msg is not _WAKE]

	def _has_work(self) -> bool:
		with self._lock:
			return bool(self._in_flight)

	def _drain(self, out: list[JobEvent | Completion]) -> None:
		while True:
			try:
				out.append(self._out.get_nowait())
			except queue.Empty:
				return

	def _collect(self, msg: JobEvent | Completion) -> JobEvent | Completion:
		if isinstance(msg, Completion):
			with self._lock:
				self._in_flight.pop(msg.index, None)
		return msg

	def track_done(self) -> None:
		"""The coordinator calls this once per track downloaded successfully: that is the controller's throughput unit."""
		with self._lock:
			action = self._adaptive.on_completion()
			if action:
				log(f"concurrency: {self._adaptive.describe()} action={action}")

	def _dispatch(self) -> None:
		while True:
			with self._lock:
				if self._cancelled or len(self._in_flight) >= self._adaptive.level:
					return
				busy = {key for key, _lane in self._in_flight.values()}
				lanes = [lane for _key, lane in self._in_flight.values() if lane is not None]
				limits = self._policy.lane_limits
				# A ready entry parked because its lane is full must not block the ones behind it.
				pick = next((e for e in self._ready if e[2] not in busy and (e[3] not in limits or lanes.count(e[3]) < limits[e[3]])), None)
				if pick is None:
					return
				self._ready.remove(pick)
				self._in_flight[pick[0]] = (pick[2], pick[3])
			self._executor.submit(self._job, pick[0], pick[1], pick[3])

	def _job(self, index: int, item: Any, lane: str | None) -> None:
		ctx = JobContext(index, self._out, self._parallel)
		# Lane pacer first: a job waiting on its lane gap occupies a thread and an in-flight slot but not the global pacer's turn.
		lane_pacer = self._lane_pacers.get(lane) if lane is not None else None
		pacer = (lane_pacer, self._pacer) if lane_pacer is not None and self._pacer is not None else self._pacer
		try:
			with process_scope(self._token), pacer_scope(pacer):
				result = self._run(item, ctx)
			done = Completion(index, item, result, None, False)
		except DownloadCancelled:
			done = Completion(index, item, None, None, True)
		except BaseException as exc:
			done = Completion(index, item, None, exc, False)
		self._out.put(done)
