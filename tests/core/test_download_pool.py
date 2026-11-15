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


def _limit(requested, profile=N, *, in_process=False, cpus=4) -> int:
	return concurrency_limit(requested, profile, in_process_ytdlp=in_process, cpu_count=cpus)


def test_limit_table() -> None:
	for cpus, expected in [(1, 2), (2, 3), (4, 6), (8, 8), (12, 8), (None, 2)]:
		assert _limit(None, N, cpus=cpus) == expected
		assert _limit(None, L, cpus=cpus) == expected
	for requested, expected in [(0, 1), (1, 1), (3, 3), (8, 8), (99, 8)]:
		assert _limit(requested, N) == expected
		assert _limit(requested, L) == expected
	for requested in (None, 1, 5):
		assert _limit(requested, A) == 1
		assert _limit(requested, N, in_process=True) == 1


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


def _adaptive_pool(monkeypatch, logs: list[str], run=lambda item, ctx: item) -> DownloadPool:
	from csvmusic.core import download_pool

	monkeypatch.setattr(download_pool, "concurrency_limit", lambda *a, **k: 6)
	monkeypatch.setattr(download_pool, "log", logs.append)
	now = [0.0]
	holder: list[DownloadPool] = []

	def clock() -> float:
		# Called under the pool lock, so read the controller directly; throughput grows with the level.
		now[0] += 18.0 / (holder[0]._adaptive.level if holder else 3)
		return now[0]

	pool = DownloadPool(run, PoolPolicy(6), N, clock=clock)
	holder.append(pool)
	return pool


def test_adaptive_limit_climbs_and_gates_in_flight(monkeypatch) -> None:
	logs: list[str] = []
	lock = threading.Lock()
	state = {"cur": 0, "max": 0}
	release = threading.Event()

	def run(item, ctx):
		with lock:
			state["cur"] += 1
			state["max"] = max(state["max"], state["cur"])
		if item[1] < item[0]:
			assert release.wait(5)
		with lock:
			state["cur"] -= 1
		return item

	pool = _adaptive_pool(monkeypatch, logs, run)
	with pool:
		assert pool.limit == 3
		seen = {pool.limit}
		for phase in range(40):
			lim = pool.limit
			if lim == 6:
				break
			release.clear()
			state["max"] = 0
			for i in range(lim + 2):
				pool.submit(phase * 100 + i, (lim, i), key=phase * 100 + i)
			done: list[Completion] = []
			while True:
				got = [m for m in pool.pump(0.05) if isinstance(m, Completion)]
				for _ in got:
					pool.track_done()
				done += got
				with lock:
					if state["cur"] == lim:
						release.set()
				if len(done) == lim + 2:
					break
			assert state["max"] == lim
			assert lim <= pool.limit <= 6
			seen.add(pool.limit)
		assert pool.limit == 6
	assert {3, 4, 5, 6} <= seen
	assert any(m.startswith("concurrency: ") and "action=climb" in m for m in logs)


def test_apply_profile_ceiling_one_drops_limit_now(monkeypatch) -> None:
	from csvmusic.core import download_pool

	logs: list[str] = []
	pool = _adaptive_pool(monkeypatch, logs)
	monkeypatch.setattr(download_pool, "concurrency_limit", lambda requested, profile, **k: 1 if profile is A else 6)
	with pool:
		assert pool.limit == 3 and pool.has_capacity()
		for i in range(3):
			pool.submit(i, i, key=i)
		assert not pool.has_capacity()
		pool.apply_profile(A)
		assert pool.limit == 1
		assert not pool.has_capacity()
		collect(pool, 3)
		assert pool.has_capacity()


def test_completions_do_not_feed_controller_only_track_done_does(monkeypatch) -> None:
	logs: list[str] = []
	pool = _adaptive_pool(monkeypatch, logs)
	calls: list[int] = []
	real = pool._adaptive.on_completion
	pool._adaptive.on_completion = lambda: calls.append(1) or real()  # type: ignore[method-assign]
	with pool:
		for i in range(5):
			pool.submit(i, i, key=i)
		collect(pool, 5)
		assert not calls
		pool.track_done()
		assert calls == [1]


def test_inline_has_no_active_controller() -> None:
	with DownloadPool(lambda item, ctx: item, PoolPolicy(1), N) as pool:
		for i in range(10):
			pool.submit(i, i, key=i)
			collect(pool, 1)
		assert pool.limit == 1


def _lane_pool(run, limit=2, requested=6, monkeypatch=None) -> DownloadPool:
	from csvmusic.core import download_pool

	if monkeypatch is not None:
		monkeypatch.setattr(download_pool, "concurrency_limit", lambda *a, **k: requested)
	return DownloadPool(run, PoolPolicy(requested, lane_limits={"account": limit}), N)


def test_lane_limit_never_exceeded_and_anonymous_keep_going(monkeypatch) -> None:
	lock = threading.Lock()
	state = {"cur": 0, "max": 0}
	anon_started = threading.Event()
	release = threading.Event()

	def run(item, ctx):
		if item == "anon":
			anon_started.set()
			release.wait(5)
			return item
		with lock:
			state["cur"] += 1
			state["max"] = max(state["max"], state["cur"])
		release.wait(5)
		with lock:
			state["cur"] -= 1
		return item

	with _lane_pool(run, monkeypatch=monkeypatch) as pool:
		for i in range(6):
			pool.submit(i, "acc", key=i, lane="account")
		pool.submit(10, "anon", key=10)
		pool.pump(0)
		assert anon_started.wait(5)  # started while the lane is full
		assert state["cur"] <= 2 and len(pool._in_flight) == 3
		release.set()
		done = collect(pool, 7)
	assert len(done) == 7 and state["max"] == 2


def test_parked_lane_entries_do_not_consume_pool_capacity(monkeypatch) -> None:
	release = threading.Event()
	anon = threading.Event()

	def run(item, ctx):
		if item == "anon":
			anon.set()
			return item
		release.wait(5)
		return item

	with _lane_pool(run, requested=6, monkeypatch=monkeypatch) as pool:
		pool._adaptive._level = 6
		for i in range(8):
			pool.submit(i, "acc", key=i, lane="account")
		pool._dispatch()
		assert len(pool._in_flight) == 2 and len(pool._ready) == 6
		assert pool.has_capacity()
		for i in range(10, 14):
			assert pool.has_capacity()
			pool.submit(i, "anon", key=i)
			pool._dispatch()
		assert anon.wait(5)
		assert not pool.has_capacity()  # 2 lane jobs + 4 anonymous in flight = level
		release.set()
		collect(pool, 12)


def test_full_lane_does_not_block_following_entries(monkeypatch) -> None:
	release = threading.Event()
	ran: list[int] = []

	def run(item, ctx):
		ran.append(item)
		if item < 2:
			release.wait(5)
		return item

	with _lane_pool(run, monkeypatch=monkeypatch) as pool:
		for i in range(3):
			pool.submit(i, i, key=i, lane="account")
		pool.submit(3, 3, key=3, lane="other")
		pool.submit(4, 4, key=4)
		pool._dispatch()
		assert [e[0] for e in pool._ready] == [2, 4]  # global level is 3; entry 2 is parked, 3 passed it
		assert sorted(pool._in_flight) == [0, 1, 3]
		release.set()
		collect(pool, 5)
	assert sorted(ran) == [0, 1, 2, 3, 4]


def test_priority_respects_lane_limit(monkeypatch) -> None:
	release = threading.Event()
	ran: list[int] = []

	def run(item, ctx):
		ran.append(item)
		release.wait(5)

	with _lane_pool(run, monkeypatch=monkeypatch) as pool:
		pool.submit(0, 0, key=0, lane="account")
		pool.submit(1, 1, key=1, lane="account")
		pool.pump(0)
		pool.submit(2, 2, key=2, lane="account", priority=True)
		pool.submit(3, 3, key=3)
		pool._dispatch()
		assert [e[0] for e in pool._ready] == [2]
		assert 3 in pool._in_flight and 2 not in pool._in_flight
		release.set()
		collect(pool, 4)


def test_cancel_drops_ready_lane_entries(monkeypatch) -> None:
	release = threading.Event()

	def run(item, ctx):
		release.wait(5)

	with _lane_pool(run, monkeypatch=monkeypatch) as pool:
		for i in range(4):
			pool.submit(i, i, key=i, lane="account")
		pool.pump(0)
		assert not pool.idle() and pool.has_capacity()  # parked lane entries take no capacity
		pool.cancel()
		release.set()
		done = collect(pool, 4)
	assert sorted(c.index for c in done if c.cancelled) == [2, 3]


def test_lane_pacer_order_and_no_lane_uses_global_only(monkeypatch) -> None:
	seen = {}

	def run(item, ctx):
		seen[item] = downloader._active_pacer.__dict__.get("pacer")

	with _lane_pool(run, monkeypatch=monkeypatch) as pool:
		pool.submit(0, "lane", key=0, lane="account")
		pool.submit(1, "free", key=1)
		collect(pool, 2)
		assert seen["lane"] == (pool._lane_pacers["account"], pool._pacer)
		assert seen["free"] is pool._pacer


def test_inline_ignores_lane_pacers_and_limits() -> None:
	seen = []

	def run(item, ctx):
		seen.append((threading.current_thread(), downloader._active_pacer.__dict__.get("pacer")))

	with DownloadPool(run, PoolPolicy(1, lane_limits={"account": 2}), N) as pool:
		assert not pool._lane_pacers
		pool.submit(0, 0, key=0, lane="account")
		collect(pool, 1)
	assert seen == [(threading.current_thread(), None)]


def test_controller_does_not_climb_on_completions_alone(monkeypatch) -> None:
	logs: list[str] = []
	pool = _adaptive_pool(monkeypatch, logs)
	with pool:
		for batch in range(10):
			for i in range(6):
				pool.submit(batch * 10 + i, i, key=batch * 10 + i)
			collect(pool, 6)
		assert pool.limit == 3 and not logs


def test_pressure_lowers_level_holds_during_cooldown_and_recovers() -> None:
	from csvmusic.core.resources import Pressure

	now = [0.0]
	state = [Pressure.CRITICAL]
	with DownloadPool(lambda i, c: i, PoolPolicy(4), N, clock=lambda: now[0], sampler=lambda: state[0]) as pool:
		assert pool._adaptive.level == 3
		pool.pump(0)
		assert pool._adaptive.level == 1
		for _ in range(40):  # 200 s of completions while pressure persists: no climb
			now[0] += 5
			pool.pump(0)
			pool.track_done()
		assert pool._adaptive.level == 1
		state[0] = Pressure.OK
		now[0] += 70
		pool.pump(0)
		actions = []
		for _ in range(60):
			now[0] += 5
			actions.append(pool._adaptive.on_completion())
		assert "climb" in actions


def test_inline_pool_never_samples_pressure() -> None:
	calls = []
	with DownloadPool(lambda i, c: i, PoolPolicy(1), N, sampler=lambda: calls.append(1)) as pool:
		pool.pump(0)
	assert calls == []
