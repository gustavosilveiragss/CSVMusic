import sys
import threading
import time

import pytest

from csvmusic.core import audio_quality as audio_quality_module
from csvmusic.core import download_pool, downloader
from csvmusic.core.paths import INTERNAL_YTDLP
from csvmusic.ui import workers
from tests.ui.test_workers_parallel import SIGNALS, _cand, _summary, _track, _worker

LIMIT = RuntimeError("HTTP Error 429: Too Many Requests")


class Rig:
	"""Fake download layer that records how many jobs are in flight."""

	def __init__(self) -> None:
		self.lock = threading.Lock()
		self.cur = 0
		self.max = 0
		self.calls: list[tuple[str, str, int]] = []
		self.behaviors: dict = {}
		self.gates: dict = {}

	def download(self, vid, dest, base, profile):
		title = vid[2:]
		with self.lock:
			self.cur += 1
			self.max = max(self.max, self.cur)
			self.calls.append((title, profile.label, self.cur))
		try:
			gate = self.gates.get(title)
			if gate:
				gate()
			behavior = self.behaviors.get(title)
			if isinstance(behavior, list):
				behavior = behavior.pop(0)
			if isinstance(behavior, BaseException):
				raise behavior
			path = dest / f"{base}.mp3"
			path.write_bytes(b"a")
			return path
		finally:
			with self.lock:
				self.cur -= 1

	def calls_for(self, title: str) -> list[tuple[str, int]]:
		return [(label, cur) for t, label, cur in self.calls if t == title]


def _setup(monkeypatch, tmp_path, titles, *, max_workers=3, matches=None):
	worker, events = _worker(tmp_path, [_track(t) for t in titles], force_download=False)
	worker.max_workers = max_workers
	matches = matches or {t: (_cand(f"v-{t}", 0.95), 0.95, [_cand(f"v-{t}", 0.95)]) for t in titles}
	rig = Rig()

	def find_best(_yt, track):
		result = matches[track["title"]]
		if isinstance(result, Exception):
			raise result
		return result

	monkeypatch.setattr(workers, "YTMusic", lambda: object())
	monkeypatch.setattr(workers, "find_best", find_best)
	monkeypatch.setattr(workers, "_track_cover_bytes", lambda _t, vid: b"c")
	monkeypatch.setattr(workers, "tag_file", lambda *_a, **_k: None)
	monkeypatch.setattr(workers, "_resolve_ytdlp", lambda: "/usr/bin/yt-dlp")
	monkeypatch.setattr(audio_quality_module, "audio_quality", lambda fp: {"low": False, "codec": "aac", "kbps": 256})
	monkeypatch.setattr(worker, "_download_with_profile", rig.download)
	return worker, events, rig


def _results(events):
	return {args[0]: args[1].get("downloaded") for name, args in events if name == "sig_track_result"}


def _done(events):
	return next(args for name, args in events if name == "sig_done")


def _barrier(parties: int):
	barrier = threading.Barrier(parties, timeout=10)
	return barrier.wait


def _pool_threads() -> list[str]:
	return [t.name for t in threading.enumerate() if t.name.startswith("ThreadPoolExecutor")]


def test_three_jobs_run_at_once(monkeypatch, tmp_path):
	titles = [f"t{i}" for i in range(6)]
	worker, events, rig = _setup(monkeypatch, tmp_path, titles)
	wait = _barrier(3)
	rig.gates = {"t0": wait, "t1": wait, "t2": wait}
	started = time.monotonic()
	worker.run()
	assert time.monotonic() - started < 5.0
	assert rig.max == 3
	assert [args for name, args in events if name == "sig_progress"] == [[i, 6] for i in range(1, 7)]
	assert _pool_threads() == []


def test_same_results_as_single_worker(monkeypatch, tmp_path):
	titles = ["a", "b", "c", "d"]
	matches = {t: (_cand(f"v-{t}", 0.95), 0.95, [_cand(f"v-{t}", 0.95)]) for t in titles}
	matches["c"] = (None, 0.1, [])
	runs = []
	for name, n in (("serial", 1), ("pooled", 3)):
		worker, events, _ = _setup(monkeypatch, tmp_path / name, titles, max_workers=n, matches=matches)
		worker.run()
		runs.append(sorted((args[0], str(_summary(args[1]))) for name_, args in events if name_ == "sig_track_result"))
	assert runs[0] == runs[1] and len(runs[0]) == 4


def test_done_and_playlist_follow_csv_order(monkeypatch, tmp_path):
	titles = ["slow", "fast1", "fast2"]
	worker, events, rig = _setup(monkeypatch, tmp_path, titles)
	worker.write_m3u8 = True
	for t in worker.tracks_override:
		t["album"] = "Album"
	fast_done = threading.Event()
	seen = []

	def on_result(index, _payload):
		seen.append(index)
		if {1, 2} <= set(seen):
			fast_done.set()

	worker.sig_track_result.connect(on_result)
	rig.gates = {"slow": lambda: fast_done.wait(10)}
	worker.run()
	assert seen[-1] == 0
	assert [t["title"] for t in _done(events)[1]] == titles, _done(events)[0]
	m3u = next((tmp_path / "out").rglob("*.m3u8")).read_text(encoding="utf-8")
	assert [line for line in m3u.splitlines() if line and not line.startswith("#")] == [f"Artist - {t}.mp3" for t in titles]


def test_same_output_key_never_overlaps(monkeypatch, tmp_path):
	titles = ["Other", "Song", "Song.feat X", "song"]
	worker, events, rig = _setup(monkeypatch, tmp_path, titles)
	active = {"song": 0, "bad": 0}
	lock = threading.Lock()
	inner = worker._download_with_profile

	def guarded(vid, dest, base, profile):
		shared = vid[2:] != "Other"
		if shared:
			with lock:
				active["song"] += 1
				active["bad"] += active["song"] > 1
		try:
			time.sleep(0.02)
			return inner(vid, dest, base, profile)
		finally:
			if shared:
				with lock:
					active["song"] -= 1

	monkeypatch.setattr(worker, "_download_with_profile", guarded)
	worker.run()
	assert active["bad"] == 0
	assert all(_results(events).values()) and len(_results(events)) == 4


def test_search_abort_waits_for_inflight(monkeypatch, tmp_path):
	titles = ["ok", "e1", "e2", "e3", "never"]
	err = RuntimeError("search down")
	matches = {"ok": (_cand("v-ok", 0.95), 0.95, [_cand("v-ok", 0.95)]), "e1": err, "e2": err, "e3": err}
	worker, events, rig = _setup(monkeypatch, tmp_path, titles, matches=matches)
	aborted = threading.Event()
	worker.sig_warning.connect(lambda _msg: aborted.set())
	rig.gates = {"ok": lambda: aborted.wait(10)}
	worker.run()
	done = _done(events)
	assert done[0].startswith("Stopped because search is unavailable")
	assert [t["title"] for t in done[1]] == ["ok"]
	assert [s["track"]["title"] for s in done[2]] == ["e1", "e2", "e3"]
	assert 4 not in _results(events)


def test_rate_limit_one_mitigation_one_safe_mode_retry(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"])
	rig.behaviors = {"a": [LIMIT, None], "b": [LIMIT, LIMIT]}
	b_started = threading.Event()
	rig.gates = {"a": lambda: b_started.wait(10), "b": b_started.set}
	worker.run()
	assert [label for label, _ in rig.calls_for("a")] == ["normal", "aggressive"]
	assert [label for label, _ in rig.calls_for("b")][-1] == "aggressive"
	assert len(rig.calls_for("b")) == 2
	assert _results(events) == {0: True, 1: False}
	assert sum(1 for name, _ in events if name == "sig_warning") == 1
	statuses = [args[1] for name, args in events if name == "sig_row_status" and args[0] == 0]
	assert statuses.count("Retrying with YouTube safe mode…") == 1


def test_escalation_caps_inflight_and_tightens_pacer(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, [f"t{i}" for i in range(6)])
	rig.behaviors = {"t0": [LIMIT, None]}
	handled = threading.Event()
	rig.gates = {"t1": lambda: handled.wait(10), "t2": lambda: handled.wait(10)}
	inner = worker._apply_mitigation

	def spy_mitigation(*args):
		inner(*args)
		handled.set()

	monkeypatch.setattr(worker, "_apply_mitigation", spy_mitigation)
	gaps = []

	class SpyPacer(download_pool.YtdlpPacer):
		def set_gap(self, min_gap_s, jitter_s):
			gaps.append((min_gap_s, jitter_s))
			super().set_gap(min_gap_s, jitter_s)

	monkeypatch.setattr(download_pool, "YtdlpPacer", SpyPacer)
	worker.run()
	assert gaps == [download_pool.pacer_gap_for_profile(workers.YOUTUBE_MITIGATION_AGGRESSIVE)]
	assert [cur for _label, cur in rig.calls_for("t0")][-1] == 1
	assert [cur for t in ("t3", "t4", "t5") for _l, cur in rig.calls_for(t)] == [1, 1, 1]
	assert all(_results(events).values())


def test_retries_respect_limit_after_escalation(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"])
	rig.behaviors = {t: [LIMIT, None] for t in "abc"}
	wait = _barrier(3)
	first = {t: True for t in "abc"}

	def gate_once(title):
		def gate():
			if first[title]:
				first[title] = False
				wait()
		return gate

	rig.gates = {t: gate_once(t) for t in "abc"}
	worker.run()
	retries = [cur for t, label, cur in rig.calls if label == "aggressive"]
	assert len(retries) == 3 and set(retries) == {1}
	assert all(_results(events).values())
	assert sum(1 for name, _ in events if name == "sig_warning") == 1


def test_stop_kills_inflight_jobs_quickly(monkeypatch, tmp_path, sleeper):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"])
	started = threading.Semaphore(0)

	def blocking(vid, dest, base, profile):
		started.release()
		downloader._run_capture(sleeper.cmd)
		raise AssertionError("not cancelled")

	monkeypatch.setattr(worker, "_download_with_profile", blocking)
	timing = {}

	def stopper():
		for _ in range(3):
			started.acquire(timeout=10)
		time.sleep(0.2)
		t0 = time.monotonic()
		worker.stop()
		timing["stop"] = time.monotonic() - t0

	thread = threading.Thread(target=stopper)
	thread.start()
	t0 = time.monotonic()
	worker.run()
	total = time.monotonic() - t0
	thread.join()
	assert timing["stop"] < 0.05
	assert total < 2.5
	done = _done(events)
	assert done[0] == "Stopped (partial results saved)."
	assert done[1] == [] and done[3] == []
	assert {args[0]: args[1] for name, args in events if name == "sig_row_status"} == {0: "Stopped", 1: "Stopped", 2: "Stopped"}
	assert "sig_track_result" not in {name for name, _ in events}
	time.sleep(0.2)
	assert sleeper.leftovers() == ""
	assert _pool_threads() == []


def test_fatal_error_leaves_no_pool_thread_before_sig_done(monkeypatch, tmp_path, sleeper):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["fast", "slow", "slower"])

	def blocking(vid, dest, base, profile):
		if vid == "v-fast":
			return rig.download(vid, dest, base, profile)
		downloader._run_capture(sleeper.cmd)
		raise AssertionError("not cancelled")

	def boom(_outcome, _title):
		raise RuntimeError("quality boom")

	snapshot = {}
	worker.sig_done.connect(lambda *_a: snapshot.update(threads=_pool_threads()))
	monkeypatch.setattr(worker, "_download_with_profile", blocking)
	monkeypatch.setattr(worker, "_report_quality", boom)
	t0 = time.monotonic()
	worker.run()
	assert time.monotonic() - t0 < 5.0
	assert _done(events)[0].startswith("Fatal error")
	assert snapshot["threads"] == []
	time.sleep(0.2)
	assert sleeper.leftovers() == ""


class _Boom(BaseException):
	pass


def test_base_exception_fails_track_and_batch_finishes(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"])
	rig.behaviors = {"b": _Boom("bye")}
	worker.run()
	assert _results(events) == {0: True, 1: False, 2: True}
	done = _done(events)
	assert done[0] == "All tasks finished."
	assert [f["track"]["title"] for f in done[3]] == ["b"]
	assert [args for name, args in events if name == "sig_progress"][-1] == [3, 3]


def _stop_inside_download(worker, monkeypatch):
	inner = worker._download_with_profile

	def stopping(vid, dest, base, profile):
		worker.stop()
		return inner(vid, dest, base, profile)

	monkeypatch.setattr(worker, "_download_with_profile", stopping)


@pytest.mark.parametrize("n", [1, 3])
def test_no_retry_warning_or_retrying_status_after_stop(monkeypatch, tmp_path, n):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=n)
	rig.behaviors = {"a": [LIMIT, None]}
	_stop_inside_download(worker, monkeypatch)
	worker.run()
	assert len(rig.calls_for("a")) == 1
	statuses = [args[1] for name, args in events if name == "sig_row_status" and args[0] == 0]
	assert statuses[-1] == "Stopped" and "Retrying with YouTube safe mode…" not in statuses
	assert not [name for name, _ in events if name == "sig_warning"]
	done = _done(events)
	assert done[0] == "Stopped (partial results saved)." and done[1] == [] and done[3] == []


def test_pause_after_rate_limit_uses_the_escalated_profile(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=1)
	rig.behaviors = {"a": [LIMIT, None]}
	seen = []
	monkeypatch.setattr(workers, "track_pause_s", lambda profile: seen.append(profile.label) or 0.0)
	worker.run()
	assert seen == ["aggressive"]


def test_stop_during_pause_returns_quickly(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"], max_workers=1)
	monkeypatch.setattr(workers, "track_pause_s", lambda _profile: 6.0)
	stopped = {}

	def stop():
		stopped["at"] = time.monotonic()
		worker.stop()

	threading.Timer(0.3, stop).start()
	started = time.monotonic()
	worker.run()
	end = time.monotonic()
	assert end - stopped["at"] < 0.5 and end - started < 2.0
	assert _done(events)[0] == "Stopped (partial results saved)."


def test_completions_are_served_during_the_pause(monkeypatch, tmp_path):
	titles = ["a", "skip", "b"]
	matches = {t: (_cand(f"v-{t}", 0.95), 0.95, [_cand(f"v-{t}", 0.95)]) for t in titles}
	matches["skip"] = (None, 0.4, [_cand("v-skip", 0.4)])
	worker, events, rig = _setup(monkeypatch, tmp_path, titles, max_workers=2, matches=matches)
	monkeypatch.setattr(workers, "track_pause_s", lambda _profile: 1.5)
	rig.gates = {"a": lambda: time.sleep(0.2)}
	stamps = {}
	started = time.monotonic()
	worker.sig_track_result.connect(lambda row, _p: stamps.setdefault(row, time.monotonic() - started))
	worker.run()
	assert stamps[0] < 1.0
	assert time.monotonic() - started >= 1.4


def _match_times(monkeypatch):
	times: dict[str, float] = {}
	inner = workers.find_best

	def find(yt, track):
		times[track["title"]] = time.monotonic()
		return inner(yt, track)

	monkeypatch.setattr(workers, "find_best", find)
	return times


def test_skip_waits_the_full_track_pause(monkeypatch, tmp_path):
	titles = ["skip", "b"]
	matches = {t: (_cand(f"v-{t}", 0.95), 0.95, [_cand(f"v-{t}", 0.95)]) for t in titles}
	matches["skip"] = (None, 0.4, [_cand("v-skip", 0.4)])
	worker, events, rig = _setup(monkeypatch, tmp_path, titles, max_workers=2, matches=matches)
	monkeypatch.setattr(workers, "track_pause_s", lambda _profile: 0.8)
	times = _match_times(monkeypatch)
	worker.run()
	assert times["b"] - times["skip"] >= 0.75


def test_searches_keep_the_minimum_gap(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=2)
	monkeypatch.setattr(download_pool, "MATCH_GAP_S", 0.5)
	times = _match_times(monkeypatch)
	worker.run()
	assert times["b"] - times["a"] >= 0.45


def test_retry_survives_search_abort(monkeypatch, tmp_path):
	titles = ["ok", "e1", "e2", "e3", "never"]
	err = RuntimeError("search down")
	matches = {"ok": (_cand("v-ok", 0.95), 0.95, [_cand("v-ok", 0.95)]), "e1": err, "e2": err, "e3": err}
	worker, events, rig = _setup(monkeypatch, tmp_path, titles, matches=matches)
	aborted = threading.Event()
	worker.sig_warning.connect(lambda msg: aborted.set() if "search" in msg.lower() else None)
	rig.behaviors = {"ok": [LIMIT, None]}
	rig.gates = {"ok": lambda: aborted.wait(10)}
	worker.run()
	assert [label for label, _ in rig.calls_for("ok")] == ["normal", "aggressive"]
	assert _results(events)[0] is True
	assert _done(events)[0].startswith("Stopped because search is unavailable")


def test_stop_during_match_marks_row_stopped_and_not_matched(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=3)
	inner = workers.find_best

	def find(yt, track):
		result = inner(yt, track)
		if track["title"] == "b":
			worker.stop()
		return result

	monkeypatch.setattr(workers, "find_best", find)
	worker.run()
	statuses = [args[1] for name, args in events if name == "sig_row_status" and args[0] == 1]
	assert statuses == ["Stopped"]
	assert [args for name, args in events if name == "sig_match_stats"][-1] == [1, 0]
	assert 1 not in _results(events)


def test_inline_stop_kills_running_download(monkeypatch, tmp_path, sleeper):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=1)
	started = threading.Event()

	def blocking(vid, dest, base, profile):
		started.set()
		downloader._run_capture(sleeper.cmd)
		raise AssertionError("not cancelled")

	monkeypatch.setattr(worker, "_download_with_profile", blocking)
	threading.Thread(target=lambda: (started.wait(10), time.sleep(0.2), worker.stop())).start()
	t0 = time.monotonic()
	worker.run()
	assert time.monotonic() - t0 < 2.5
	time.sleep(0.2)
	assert sleeper.leftovers() == ""


def test_signals_and_notices_come_from_the_run_thread(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"])
	monkeypatch.setattr(worker, "_download_with_profile", type(worker)._download_with_profile.__get__(worker))

	def impl(vid, dest, base, profile):
		downloader._download_notice("slow down")
		path = dest / f"{base}.mp3"
		path.write_bytes(b"a")
		return path

	monkeypatch.setattr(worker, "_download_with_profile_impl", impl)
	seen = []
	logs = []
	for name in SIGNALS:
		getattr(worker, name).connect(lambda *_args, _n=name: seen.append(threading.current_thread()))
	worker.sig_log.connect(logs.append)
	worker.run()
	assert sum("[recovery]" in m for m in logs) == 3
	assert set(seen) == {threading.current_thread()}


def test_in_process_ytdlp_downloads_one_at_a_time(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"])
	monkeypatch.setattr(workers, "_resolve_ytdlp", lambda: INTERNAL_YTDLP)
	worker.run()
	assert rig.max == 1
	logs = [args[0] for name, args in events if name == "sig_log"]
	assert logs.count("Parallel downloads need a standalone yt-dlp; downloading one at a time.") == 1


def test_missing_ytdlp_is_not_the_in_process_case(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"])

	def missing():
		raise RuntimeError("no yt-dlp")

	monkeypatch.setattr(workers, "_resolve_ytdlp", missing)
	worker.run()
	logs = [args[0] for name, args in events if name == "sig_log"]
	assert "Parallel downloads need a standalone yt-dlp; downloading one at a time." not in logs


def test_account_cookies_cap_at_two(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c", "d"])
	worker.cookies_file = str(tmp_path / "cookies.txt")
	wait = _barrier(2)
	rig.gates = {"a": wait, "b": wait}
	worker.run()
	assert rig.max == 2
	assert all(_results(events).values())


def test_single_worker_uses_inline_executor_and_no_threads(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=1)

	def boom(*_a, **_k):
		raise AssertionError("no thread pool for one worker")

	monkeypatch.setattr(download_pool, "ThreadPoolExecutor", boom)
	before = threading.active_count()
	worker.run()
	assert _results(events) == {0: True, 1: True}
	assert rig.max == 1
	assert threading.active_count() == before


@pytest.mark.parametrize("n", [1, 3])
def test_no_thread_survives_run(monkeypatch, tmp_path, n):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c"], max_workers=n)
	before = threading.active_count()
	worker.run()
	assert threading.active_count() == before
