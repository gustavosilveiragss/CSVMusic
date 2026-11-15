"""Adaptive download concurrency. Qt-free; the owner calls it from a single thread."""

from __future__ import annotations

import math
import time
from typing import Callable


def hard_cap(cpu_count: int | None) -> int:
	if cpu_count is None:
		return 2
	return max(2, min(8, math.ceil(1.5 * cpu_count)))


class AdaptiveLimit:
	"""Hill-climbs throughput (finished tracks per second, fed by the owner via on_completion) over the concurrency level."""

	def __init__(
		self,
		start: int,
		ceiling: int,
		*,
		clock: Callable[[], float] = time.monotonic,
		epsilon: float = 0.10,
		min_window_s: float = 30.0,
		reprobe_s: float = 300.0,
	) -> None:
		self._clock = clock
		self._epsilon = epsilon
		self._min_window_s = min_window_s
		self._reprobe_s = reprobe_s
		self._ceiling = max(1, ceiling)
		self._level = max(1, min(start, self._ceiling))
		self._base_x: float | None = None
		self._base_level = self._level
		self._last_x: float | None = None
		self._trial = False
		self._plateau_since: float | None = None
		self._restart()

	@property
	def level(self) -> int:
		return self._level

	@property
	def ceiling(self) -> int:
		return self._ceiling

	def _restart(self, warmup: bool = False) -> None:
		# Jobs completing right after a level change were dispatched at the old level.
		self._plateau_since = None
		self._warmup_left = self._level if warmup else 0
		self._count = 0
		self._window_start = self._clock()

	def _enter_plateau(self) -> None:
		self._plateau_since = self._clock()
		self._trial = False

	def _climb(self) -> str:
		self._base_level = self._level
		self._level += 1
		self._trial = True
		self._restart(warmup=True)
		return "climb"

	def lower_ceiling(self, n: int) -> None:
		self._ceiling = min(self._ceiling, max(1, n))
		if self._level > self._ceiling:
			self._level = self._ceiling
			self._base_x = None
			self._trial = False
			self._restart(warmup=True)

	def on_completion(self) -> str | None:
		if self._ceiling <= 1:
			return None
		now = self._clock()
		if self._plateau_since is not None:
			if self._level < self._ceiling and now - self._plateau_since >= self._reprobe_s:
				# Conditions drift: re-measure from scratch instead of comparing with a stale base.
				self._base_x = None
				self._trial = False
				self._restart()
				return "reprobe"
			return None
		if self._warmup_left > 0:
			self._warmup_left -= 1
			if self._warmup_left == 0:
				self._window_start = now
			return None
		self._count += 1
		elapsed = now - self._window_start
		if self._count < max(4, 2 * self._level) or elapsed < self._min_window_s:
			return None
		x = self._count / elapsed
		self._last_x = x
		if self._trial and self._base_x is not None and x < self._base_x * (1 + self._epsilon):
			self._level = self._base_level
			self._enter_plateau()
			return "revert"
		self._base_x = x
		if self._level < self._ceiling:
			return self._climb()
		self._enter_plateau()
		return "hold"

	def describe(self) -> str:
		x = (self._last_x or 0.0) * 60
		prev = (self._base_x or 0.0) * 60
		return f"level={self._level} x_per_min={x:.1f} prev={prev:.1f}"
