import sys
import threading
import time

import pytest

from csvmusic.core import audio_quality as audio_quality_module
from csvmusic.core import download_pool, downloader
from csvmusic.core.downloader import AgeRestrictedError, age_restriction_message
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
		self.auths: list[tuple[str, str]] = []
		self.cur_by_auth = {workers.AUTH_ANON: 0, workers.AUTH_ACCOUNT: 0}
		self.max_by_auth = {workers.AUTH_ANON: 0, workers.AUTH_ACCOUNT: 0}
		self.behaviors: dict = {}
		self.gates: dict = {}

	def download(self, vid, dest, base, profile):
		title = vid[2:]
		auth = workers._JOB_CONTEXT.get().auth
		with self.lock:
			self.cur += 1
			self.max = max(self.max, self.cur)
			self.calls.append((title, profile.label, self.cur))
			self.auths.append((title, auth))
			self.cur_by_auth[auth] += 1
			self.max_by_auth[auth] = max(self.max_by_auth[auth], self.cur_by_auth[auth])
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
				self.cur_by_auth[auth] -= 1

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


AGE = AgeRestrictedError("age restricted")


def _login(worker, tmp_path):
	worker.cookies_file = str(tmp_path / "cookies.txt")


@pytest.fixture(autouse=True)
def no_lane_pacing(monkeypatch):
	monkeypatch.setattr(download_pool, "lane_pacer_gap", lambda _lane: None)


def _statuses(events, row):
	return [args[1] for name, args in events if name == "sig_row_status" and args[0] == row]


def test_login_does_not_touch_the_account_for_plain_tracks(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c", "d"])
	_login(worker, tmp_path)
	worker.run()
	assert {auth for _t, auth in rig.auths} == {workers.AUTH_ANON} and len(rig.auths) == 4
	assert all(_results(events).values())


def test_cookie_args_only_for_the_account(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=1)
	_login(worker, tmp_path)
	seen = []
	monkeypatch.setattr(workers, "download_mp3", lambda *a, extra_yt_dlp_args=None, **k: seen.append(extra_yt_dlp_args) or (_ for _ in ()).throw(RuntimeError("stop")))
	monkeypatch.setattr(worker, "_download_with_profile", type(worker)._download_with_profile.__get__(worker))
	worker.run()
	assert seen and "--cookies" not in (seen[0] or [])
	seen.clear()
	token = workers._JOB_CONTEXT.set(workers._JobContext(lambda *_a: None, {}, workers.AUTH_ACCOUNT))
	try:
		with pytest.raises(RuntimeError):
			worker._download_with_profile_impl("v", tmp_path, "b", downloader.YOUTUBE_MITIGATION_NONE)
	finally:
		workers._JOB_CONTEXT.reset(token)
	assert seen[0][:2] == ["--cookies", worker.cookies_file]


@pytest.mark.parametrize("n", [1, 3])
def test_age_restriction_goes_anonymous_then_account_from_the_failing_candidate(monkeypatch, tmp_path, n):
	opts = [_cand("v-k1", 0.5), _cand("v-k2", 0.45, "video")]
	worker, events, rig = _setup(monkeypatch, tmp_path, ["k"], max_workers=n, matches={"k": (None, 0.5, opts)})
	worker.force_download = True
	_login(worker, tmp_path)
	rig.behaviors = {"k1": RuntimeError("bad"), "k2": [AGE, None]}
	lanes = []
	inner = download_pool.DownloadPool.submit
	monkeypatch.setattr(download_pool.DownloadPool, "submit", lambda self, *a, **k: lanes.append(k.get("lane")) or inner(self, *a, **k))
	worker.run()
	assert rig.auths == [("k1", workers.AUTH_ANON), ("k2", workers.AUTH_ANON), ("k2", workers.AUTH_ACCOUNT)]
	assert lanes == [None, workers.AUTH_ACCOUNT]
	assert _results(events) == {0: True}
	statuses = _statuses(events, 0)
	assert statuses.count("Needs sign-in: retrying with your YouTube account…") == 1
	assert not any(s.startswith("Fail") for s in statuses)


def test_account_refused_is_final_with_the_cookie_message(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"])
	_login(worker, tmp_path)
	refused = AgeRestrictedError(age_restriction_message(using_cookies=True))
	rig.behaviors = {"a": [AGE, refused, None]}
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON), ("a", workers.AUTH_ACCOUNT)]
	assert _results(events) == {0: False}
	assert _statuses(events, 0)[-1] == f"Fail: {refused}"[:126]


@pytest.mark.parametrize("n", [1, 3])
def test_age_without_login_fails_at_once(monkeypatch, tmp_path, n):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=n)
	err = AgeRestrictedError(age_restriction_message(using_cookies=False))
	rig.behaviors = {"a": [err, None]}
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON)]
	assert _statuses(events, 0)[-1] == f"Fail: {err}"[:126]
	assert not any("account" in s for s in _statuses(events, 0))


def test_account_lane_holds_two_while_anonymous_run_wider(monkeypatch, tmp_path):
	titles = [f"a{i}" for i in range(4)] + [f"x{i}" for i in range(8)]
	worker, events, rig = _setup(monkeypatch, tmp_path, titles, max_workers=8)
	real = download_pool.AdaptiveLimit
	monkeypatch.setattr(download_pool, "AdaptiveLimit", lambda start, ceiling, **k: real(start=ceiling, ceiling=ceiling, **k))
	_login(worker, tmp_path)
	rig.behaviors = {f"a{i}": [AGE, None] for i in range(4)}
	go = threading.Event()
	anon_wait = _barrier(3)
	account_wait = _barrier(2)

	def anon_gate():
		anon_wait()
		go.set()

	rig.gates = {"x0": anon_gate, "x1": anon_gate, "x2": anon_gate}
	account_gate_done = {"a0": False, "a1": False}

	def account_gate(title):
		def gate():
			if workers._JOB_CONTEXT.get().auth == workers.AUTH_ACCOUNT and not account_gate_done[title]:
				account_gate_done[title] = True
				account_wait()
				assert go.wait(10)
		return gate

	rig.gates.update({"a0": account_gate("a0"), "a1": account_gate("a1")})
	worker.run()
	assert rig.max_by_auth[workers.AUTH_ACCOUNT] == 2
	assert rig.max_by_auth[workers.AUTH_ANON] >= 3
	assert all(_results(events).values()) and len(_results(events)) == 12
	assert _pool_threads() == []


def test_age_escalation_after_stop_is_stopped(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=3)
	_login(worker, tmp_path)
	rig.behaviors = {"a": [AGE, None]}
	_stop_inside_download(worker, monkeypatch)
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON)]
	statuses = _statuses(events, 0)
	assert statuses[-1] == "Stopped" and "Needs sign-in: retrying with your YouTube account…" not in statuses
	assert _pool_threads() == []


def test_rate_limit_on_an_account_job_keeps_the_lane_and_safe_mode_is_authenticated(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=3)
	_login(worker, tmp_path)
	rig.behaviors = {"a": [AGE, LIMIT, None]}
	lanes = []
	inner = download_pool.DownloadPool.submit
	monkeypatch.setattr(download_pool.DownloadPool, "submit", lambda self, *a, **k: lanes.append(k.get("lane")) or inner(self, *a, **k))
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON), ("a", workers.AUTH_ACCOUNT), ("a", workers.AUTH_ACCOUNT)]
	assert [label for label, _ in rig.calls_for("a")] == ["normal", "normal", "aggressive"]
	assert lanes == [None, workers.AUTH_ACCOUNT, workers.AUTH_ACCOUNT]
	assert _results(events) == {0: True}


BOT = RuntimeError("Sign in to confirm you\u2019re not a bot")


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("err", [BOT, LIMIT])
@pytest.mark.parametrize("login", [True, False])
def test_bot_check_and_rate_limit_are_risk_retried_once_in_safe_mode(monkeypatch, tmp_path, login, err, n):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=n)
	if login:
		_login(worker, tmp_path)
	rig.behaviors = {"a": [err, None]}
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON), ("a", workers.AUTH_ACCOUNT if login else workers.AUTH_ANON)]
	assert [label for label, _ in rig.calls_for("a")] == ["normal", "aggressive"]
	assert _results(events) == {0: True}


@pytest.mark.parametrize("n", [1, 3])
def test_private_video_goes_to_the_account_and_is_final_there(monkeypatch, tmp_path, n):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=n)
	_login(worker, tmp_path)
	private = downloader.AuthRequiredError(downloader.auth_required_message(using_cookies=False))
	again = downloader.AuthRequiredError(downloader.auth_required_message(using_cookies=True))
	rig.behaviors = {"a": [private, again, None]}
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON), ("a", workers.AUTH_ACCOUNT)]
	assert _results(events) == {0: False}
	assert _statuses(events, 0)[-1] == f"Fail: {again}"[:126]


def test_anonymous_attempt_runs_in_auth_escalation_only_with_login(monkeypatch, tmp_path):
	seen = []
	for login in (True, False):
		worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=1)
		if login:
			_login(worker, tmp_path)
		rig.gates = {"a": lambda: seen.append(getattr(downloader._auth_scope, "active", False))}
		worker.run()
	assert seen == [True, False]


def test_anonymous_args_refuse_config_cookies_only_with_login(monkeypatch, tmp_path):
	worker, _events, _rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=1)
	seen = []
	monkeypatch.setattr(workers, "download_mp3", lambda *a, extra_yt_dlp_args=None, **k: seen.append(extra_yt_dlp_args or []) or tmp_path)
	token = workers._JOB_CONTEXT.set(workers._JobContext(lambda *_a: None, {}, workers.AUTH_ANON))
	try:
		worker._download_with_profile_impl("v", tmp_path, "b", downloader.YOUTUBE_MITIGATION_NONE)
		_login(worker, tmp_path)
		worker._download_with_profile_impl("v", tmp_path, "b", downloader.YOUTUBE_MITIGATION_NONE)
	finally:
		workers._JOB_CONTEXT.reset(token)
	assert "--no-cookies" not in seen[0]
	assert "--no-cookies" in seen[1] and "--no-cookies-from-browser" in seen[1] and "--cookies" not in seen[1]


def _count_track_done(monkeypatch) -> list[int]:
	counted: list[int] = []
	inner = download_pool.DownloadPool.track_done
	monkeypatch.setattr(download_pool.DownloadPool, "track_done", lambda self: counted.append(1) or inner(self))
	return counted


@pytest.mark.parametrize("n", [1, 3])
def test_throughput_counts_a_track_once_when_it_succeeds(monkeypatch, tmp_path, n):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b", "c", "d"], max_workers=n)
	_login(worker, tmp_path)
	rig.behaviors = {"a": [AGE, None], "b": [LIMIT, None], "c": RuntimeError("bad")}
	counted = _count_track_done(monkeypatch)
	worker.run()
	assert _results(events) == {0: True, 1: True, 2: False, 3: True}
	assert len(counted) == 3  # escalation and 429 retry are not samples, the final failure is not a track done


def test_lane_pacer_is_used_by_account_jobs(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=3)
	_login(worker, tmp_path)
	rig.behaviors = {"a": [AGE, None], "b": [AGE, None]}
	rig.gates = {"a": downloader._pace_ytdlp, "b": downloader._pace_ytdlp}  # what every yt-dlp launch does
	acquired: list[float] = []

	class Spy(download_pool.YtdlpPacer):
		def acquire(self):
			acquired.append(self._min_gap_s)
			return super().acquire()

	monkeypatch.setattr(download_pool, "lane_pacer_gap", lambda lane: (0.01, 0.0) if lane == downloader.ACCOUNT_LANE else None)
	monkeypatch.setattr(download_pool, "YtdlpPacer", Spy)
	worker.run()
	assert _results(events) == {0: True, 1: True}
	assert acquired.count(0.01) == 2  # one per account attempt; anonymous attempts only meet the global pacer


def test_timing_logs_carry_the_auth(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a"], max_workers=3)
	_login(worker, tmp_path)
	rig.behaviors = {"a": [AGE, None]}
	lines = []
	monkeypatch.setattr(workers, "log", lines.append)
	timing = []
	monkeypatch.setattr("csvmusic.core.timing.log", timing.append)
	worker.run()
	tracks = [m for m in lines if m.startswith("timing: stage=track")]
	assert [m.rsplit(" ", 1)[1] for m in tracks] == ["auth=anon", "auth=account"]
	assert all("auth=" in m for m in timing if "stage=download" in m) and len(timing) >= 2


def test_inline_executor_escalates_without_lanes_or_pacers(monkeypatch, tmp_path):
	worker, events, rig = _setup(monkeypatch, tmp_path, ["a", "b"], max_workers=1)
	_login(worker, tmp_path)
	rig.behaviors = {"a": [AGE, None]}

	def boom(*_a, **_k):
		raise AssertionError("no thread pool for one worker")

	monkeypatch.setattr(download_pool, "ThreadPoolExecutor", boom)
	monkeypatch.setattr(download_pool, "YtdlpPacer", boom)
	worker.run()
	assert rig.auths == [("a", workers.AUTH_ANON), ("a", workers.AUTH_ACCOUNT), ("b", workers.AUTH_ANON)]
	assert _results(events) == {0: True, 1: True}
	assert _statuses(events, 0)[:3] == ["Downloading (mp3)…", "Needs sign-in: retrying with your YouTube account…", "Tagging…"]


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
