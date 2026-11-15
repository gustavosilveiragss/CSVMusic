import sys
import threading
import time

import pytest

from csvmusic.core import downloader
from csvmusic.core.download_pool import (
	Completion, DownloadPool, JobEvent, PoolPolicy, concurrency_limit, track_pause_s,
)
from csvmusic.core.downloader import (
	YOUTUBE_MITIGATION_AGGRESSIVE, YOUTUBE_MITIGATION_LARGE_BATCH, YOUTUBE_MITIGATION_NONE,
	DownloadCancelled, YouTubeMitigationProfile,
)
from csvmusic.core.ytmusic_match import RATE_LIMIT_S

N, L, A = YOUTUBE_MITIGATION_NONE, YOUTUBE_MITIGATION_LARGE_BATCH, YOUTUBE_MITIGATION_AGGRESSIVE


def collect(pool: DownloadPool, count: int) -> list[Completion]:
	done: list[Completion] = []
	while len(done) < count:
		done += [m for m in pool.pump(0.2) if isinstance(m, Completion)]
	return done


def test_limit_table() -> None:
	for requested in range(1, 5):
		assert concurrency_limit(requested, N, account_cookies=False, in_process_ytdlp=False) == min(requested, 3)
		assert concurrency_limit(requested, L, account_cookies=False, in_process_ytdlp=False) == min(requested, 2)
		assert concurrency_limit(requested, A, account_cookies=False, in_process_ytdlp=False) == 1
		assert concurrency_limit(requested, N, account_cookies=True, in_process_ytdlp=False) == min(requested, 2)
		assert concurrency_limit(requested, N, account_cookies=False, in_process_ytdlp=True) == 1
	assert concurrency_limit(0, N, account_cookies=False, in_process_ytdlp=False) == 1


@pytest.mark.parametrize("profile", [N, L, A])
def test_track_pause_matches_old_formula(profile: YouTubeMitigationProfile) -> None:
	seen = []

	def uniform(lo: float, hi: float) -> float:
		seen.append((lo, hi))
		return hi

	got = track_pause_s(profile, uniform=uniform)
	if profile.track_sleep_s <= 0:
		assert got == RATE_LIMIT_S and not seen
	else:
		jitter = min(1.0, profile.track_sleep_s * 0.2)
		assert got == max(RATE_LIMIT_S, profile.track_sleep_s + jitter)
	assert track_pause_s(profile, uniform=lambda a, b: 0.0) == RATE_LIMIT_S


def test_concurrency_never_exceeds_limit() -> None:
	barrier = threading.Barrier(3, timeout=5)
	lock = threading.Lock()
	state = {"cur": 0, "max": 0}

	def run(item, ctx):
		with lock:
			state["cur"] += 1
			state["max"] = max(state["max"], state["cur"])
		barrier.wait()
		with lock:
			state["cur"] -= 1
		return item

	with DownloadPool(run, PoolPolicy(4), N) as pool:
		assert pool.limit == 3 and pool.parallel
		for i in range(6):
			assert pool.submit(i, i, key=i)
		done = collect(pool, 6)
	assert sorted(c.result for c in done) == list(range(6))
	assert state["max"] == 3


def test_apply_profile_reduces_and_sets_gap() -> None:
	with DownloadPool(lambda item, ctx: item, PoolPolicy(3), N) as pool:
		pool.apply_profile(A)
		assert pool.limit == 1
		assert (pool._pacer._min_gap_s, pool._pacer._jitter_s) == downloader.pacer_gap_for_profile(A)
		pool.apply_profile(N)
		assert pool.limit == 1
		for i in range(3):
			pool.submit(i, i, key=i)
		done = collect(pool, 3)
	assert sorted(c.item for c in done) == [0, 1, 2]


def test_priority_goes_first_and_busy_key_skips() -> None:
	order = []
	with DownloadPool(lambda item, ctx: order.append(item), PoolPolicy(1), N) as pool:
		pool.submit(0, "a", key="k")
		pool.submit(1, "b", key="k")
		pool.submit(2, "retry", key="r", priority=True)
		collect(pool, 3)
	assert order == ["retry", "a", "b"]


def test_busy_key_does_not_dispatch_and_next_free_passes() -> None:
	release = threading.Event()
	started = {"b": threading.Event(), "c": threading.Event()}

	def run(item, ctx):
		if item == "a":
			assert release.wait(5)
		elif item in started:
			started[item].set()
		return item

	with DownloadPool(run, PoolPolicy(3), N) as pool:
		pool.submit(0, "a", key="k")
		pool.submit(1, "b", key="k")
		pool.submit(2, "c", key="z")
		first = [m for m in pool.pump(0) if isinstance(m, Completion)]
		assert started["c"].wait(5)
		assert not started["b"].is_set()
		release.set()
		done = first + collect(pool, 3 - len(first))
	assert started["b"].is_set() and sorted(c.item for c in done) == ["a", "b", "c"]


def test_close_rejects_new_work_but_takes_priority_retry_until_cancel() -> None:
	with DownloadPool(lambda i, c: i, PoolPolicy(1), N) as pool:
		pool.close()
		assert pool.submit(0, 0, key=0) is False
		assert pool.submit(1, 1, key=1, priority=True) is True
		assert [c.result for c in collect(pool, 1)] == [1]
		pool.cancel()
		assert pool.submit(2, 2, key=2, priority=True) is False and pool.cancelled
		assert pool.submit(3, 3, key=3) is False


def test_pool_owns_profile() -> None:
	with DownloadPool(lambda i, c: i, PoolPolicy(3), N) as pool:
		assert pool.profile is N
		pool.apply_profile(A)
		assert pool.profile is A


def test_idle_pump_waits_for_timeout() -> None:
	with DownloadPool(lambda i, c: i, PoolPolicy(1), N) as pool:
		started = time.monotonic()
		assert pool.pump(0.2) == []
		assert 0.15 <= time.monotonic() - started < 0.6


def test_cancel_wakes_idle_pump() -> None:
	with DownloadPool(lambda i, c: i, PoolPolicy(3), N) as pool:
		at: list[float] = []
		threading.Timer(0.1, lambda: (at.append(time.monotonic()), pool.cancel())).start()
		assert pool.pump(5.0) == []
		assert time.monotonic() - at[0] < 0.05
		assert pool.pump(5.0) == [] and pool.idle()


def test_cancel_marks_ready_and_kills_in_flight(sleeper) -> None:
	running = threading.Event()

	def run(item, ctx):
		if item == "sleep":
			running.set()
			downloader._run_capture(sleeper.cmd)
			raise DownloadCancelled("killed")
		return item

	with DownloadPool(run, PoolPolicy(2), N) as pool:
		pool.submit(0, "sleep", key=0)
		assert pool.pump(0) == []
		assert running.wait(5)
		pool.submit(1, "queued", key=0)
		threading.Timer(0.3, pool.cancel).start()
		done = collect(pool, 2)
	by_item = {c.item: c for c in done}
	assert by_item["queued"].cancelled
	assert by_item["sleep"].cancelled
	time.sleep(0.2)
	assert sleeper.leftovers() == ""


def test_cancel_never_blocks_and_kills_once(monkeypatch) -> None:
	release, killed = threading.Event(), []

	def slow_kill(proc) -> None:
		killed.append(proc)
		release.wait(1)

	monkeypatch.setattr(downloader, "_kill", slow_kill)
	with DownloadPool(lambda item, ctx: item, PoolPolicy(1), N) as pool:
		pool._token.register(object())
		start = time.monotonic()
		pool.cancel()
		pool.cancel()
		assert time.monotonic() - start < 0.05
		assert pool.cancelled
		deadline = time.monotonic() + 3
		while not killed and time.monotonic() < deadline:
			time.sleep(0.01)
		release.set()
	assert len(killed) == 1


def test_base_exception_becomes_error() -> None:
	def run(item, ctx):
		raise SystemExit(3)

	for limit in (1, 2):
		with DownloadPool(run, PoolPolicy(limit), N) as pool:
			pool.submit(0, "x", key=0)
			(c,) = collect(pool, 1)
		assert isinstance(c.error, SystemExit) and not c.cancelled


def test_events_precede_completion() -> None:
	def run(item, ctx):
		ctx.emit("a", 1)
		ctx.emit("b")
		return "ok"

	for limit in (1, 3):
		with DownloadPool(run, PoolPolicy(limit), N) as pool:
			pool.submit(7, "x", key=0)
			msgs = []
			while not any(isinstance(m, Completion) for m in msgs):
				msgs += pool.pump(0.2)
			assert pool.idle()
		assert msgs[0] == JobEvent(7, "a", (1,)) and msgs[1] == JobEvent(7, "b", ())
		assert isinstance(msgs[2], Completion) and msgs[2].result == "ok"


def test_exit_under_exception_leaves_no_threads() -> None:
	before = set(threading.enumerate())
	release = threading.Event()

	def run(item, ctx):
		release.wait(0.5)

	with pytest.raises(RuntimeError):
		with DownloadPool(run, PoolPolicy(3), N) as pool:
			pool.submit(0, 0, key=0)
			pool.pump(0)
			raise RuntimeError("boom")
	assert not [t for t in set(threading.enumerate()) - before if t.is_alive()]


def test_inline_uses_caller_thread_without_pacer_and_threaded_has_pacer() -> None:
	seen = []

	def run(item, ctx):
		seen.append((threading.current_thread(), downloader._active_pacer.__dict__.get("pacer"), ctx.parallel))

	with DownloadPool(run, PoolPolicy(1), N) as pool:
		assert not pool.parallel and pool._pacer is None
		pool.submit(0, 0, key=0)
		collect(pool, 1)
	assert seen[0] == (threading.current_thread(), None, False)
	seen.clear()
	with DownloadPool(run, PoolPolicy(2), N) as pool:
		assert pool._pacer is not None
		pool.submit(0, 0, key=0)
		collect(pool, 1)
	assert seen[0][0] is not threading.current_thread() and seen[0][1] is pool._pacer and seen[0][2]
