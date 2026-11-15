import subprocess
import threading

import pytest

from csvmusic.core import downloader
from csvmusic.core.downloader import (
	CancelToken, DownloadCancelled, YouTubeMitigationProfile, YtdlpPacer, pacer_gap_for_profile,
	pacer_scope, scale_profile_for_concurrency, detect_concurrency_risk,
)


class FakeClock:
	def __init__(self) -> None:
		self.now = 100.0
		self.lock = threading.Lock()

	def __call__(self) -> float:
		return self.now

	def sleep(self, seconds: float) -> None:
		with self.lock:
			self.now += seconds


def test_threads_respect_min_gap():
	clock = FakeClock()
	pacer = YtdlpPacer(5.0, 0.0, clock=clock, sleep=clock.sleep)
	starts: list[float] = []
	def work() -> None:
		for _ in range(3):
			pacer.acquire()
			starts.append(clock())
	threads = [threading.Thread(target=work) for _ in range(4)]
	for t in threads: t.start()
	for t in threads: t.join()
	starts.sort()
	assert len(starts) == 12
	assert all(b - a >= 5.0 - 1e-9 for a, b in zip(starts, starts[1:]))


def test_set_gap_changes_next_spacing():
	clock = FakeClock()
	pacer = YtdlpPacer(2.0, 0.0, clock=clock, sleep=clock.sleep)
	starts = []
	for gap in (2.0, 2.0, 10.0, 10.0):
		pacer.set_gap(gap, 0.0)
		pacer.acquire()
		starts.append(clock())
	assert starts[1] - starts[0] == pytest.approx(2.0)
	assert starts[3] - starts[2] == pytest.approx(10.0)


def test_cancel_during_wait_raises():
	token = CancelToken()
	clock = FakeClock()
	def sleep(seconds: float) -> None:
		token.cancel()
		clock.sleep(seconds)
	pacer = YtdlpPacer(60.0, 0.0, token, clock=clock, sleep=sleep)
	pacer.acquire()
	with pytest.raises(DownloadCancelled):
		pacer.acquire()
	assert clock.now - 100.0 < 1.0


class CountingPacer:
	def __init__(self) -> None:
		self.calls = 0

	def acquire(self) -> None:
		self.calls += 1


def _setup_403(monkeypatch):
	calls = []
	def run(args):
		calls.append(args)
		return (1, "", "ERROR: unable to download video data: HTTP Error 403: Forbidden") if len(calls) == 1 else (0, "ok", "")
	monkeypatch.setattr(downloader, "_run_ytdlp_module", run)
	monkeypatch.setattr(downloader, "_run_capture", lambda args: subprocess.CompletedProcess(args, *run(args)))
	monkeypatch.setattr(downloader, "capture_source", lambda text: None)
	monkeypatch.setattr(downloader, "log", lambda message: None)
	downloader.take_download_notices()
	return calls


@pytest.mark.parametrize("internal", [False, True])
def test_ytdlp_detail_paces_only_with_active_pacer(monkeypatch, internal):
	binary = downloader.INTERNAL_YTDLP if internal else "yt-dlp"
	cmd = [binary, "--cookies", "c.txt", "https://www.youtube.com/watch?v=x"]
	calls = _setup_403(monkeypatch)
	assert downloader._run_ytdlp_detail(cmd) == (0, "")
	assert len(calls) == 2
	calls.clear()
	pacer = CountingPacer()
	with pacer_scope(pacer):
		assert downloader._run_ytdlp_detail(cmd) == (0, "")
	assert len(calls) == 2
	assert pacer.calls == 2
	with pacer_scope(pacer):
		downloader._run_ytdlp_detail([binary, "url"])
	assert pacer.calls == 3


def test_scale_profile_for_concurrency():
	p = YouTubeMitigationProfile(label="x", limit_rate="1.5M")
	assert scale_profile_for_concurrency(p, 1) is p
	assert scale_profile_for_concurrency(p, 2).limit_rate == "750K"
	assert scale_profile_for_concurrency(YouTubeMitigationProfile(label="x", limit_rate="900K"), 3).limit_rate == "300K"
	assert scale_profile_for_concurrency(YouTubeMitigationProfile(label="x", limit_rate="100K"), 4).limit_rate == "50K"
	none = YouTubeMitigationProfile(label="x")
	assert scale_profile_for_concurrency(none, 3) is none
	assert p.limit_rate == "1.5M"


def test_pacer_gap_for_profile():
	assert pacer_gap_for_profile(downloader.YOUTUBE_MITIGATION_NONE) == (3.0, 0.5)
	assert pacer_gap_for_profile(downloader.YOUTUBE_MITIGATION_LARGE_BATCH) == (8.0, 2.0)
	assert pacer_gap_for_profile(downloader.YOUTUBE_MITIGATION_AGGRESSIVE) == (12.0, 2.0)


@pytest.mark.parametrize("text", [
	"Sign in to confirm you're not a bot", "Sign in to confirm you’re not a bot", "Bot verification required",
	"HTTP Error 429: Too Many Requests",
])
def test_detect_concurrency_risk_positive(text):
	assert detect_concurrency_risk(text)


@pytest.mark.parametrize("text", ["", "private video", "network unreachable"])
def test_detect_concurrency_risk_negative(text):
	assert detect_concurrency_risk(text) is None


def test_pacer_scope_tuple_acquires_in_order_and_lane_gap():
	order: list[str] = []

	class Named:
		def __init__(self, name: str) -> None:
			self.name = name

		def acquire(self) -> None:
			order.append(self.name)

	with pacer_scope((Named("lane"), Named("global"))):
		downloader._pace_ytdlp()
	assert order == ["lane", "global"]
	assert downloader.lane_pacer_gap("account") == downloader.ACCOUNT_LANE_GAP
	assert downloader.lane_pacer_gap("other") is None
