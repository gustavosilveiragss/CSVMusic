import os
import sys

import pytest

from csvmusic.core import downloader
from csvmusic.core.timing import timed


@pytest.fixture
def logged(monkeypatch):
	lines: list[str] = []
	monkeypatch.setattr("csvmusic.core.timing.log", lines.append)
	return lines


def test_timed_logs_stage_seconds_and_fields(logged):
	with timed("match", track="x") as info:
		info["rc"] = 0
	assert len(logged) == 1
	assert logged[0].startswith("timing: stage=match seconds=")
	assert logged[0].endswith("track=x rc=0")


def test_timed_logs_even_when_the_stage_raises(logged):
	with pytest.raises(RuntimeError):
		with timed("download"):
			raise RuntimeError("boom")
	assert logged[0].startswith("timing: stage=download seconds=")


def test_timing_fields_describe_ytdlp_client_and_url():
	cmd = ["/usr/bin/yt-dlp", "--extractor-args", "youtube:player_client=web_embedded", "-o", "x", "https://y/watch?v=1"]
	assert downloader._timing_fields(cmd) == {"tool": "yt-dlp", "client": "web_embedded", "url": "https://y/watch?v=1"}
	assert downloader._timing_fields(["/usr/bin/yt-dlp", "https://y"])["client"] == "default"
	assert downloader._timing_fields([downloader.INTERNAL_YTDLP, "https://y"])["tool"] == "yt-dlp"
	assert downloader._timing_fields(["/opt/ffmpeg", "-i", "in", "out.mp3"]) == {"tool": "ffmpeg"}


def test_run_capture_logs_tool_and_return_code(logged):
	downloader._run_capture([sys.executable, "-c", "pass"])
	tool = os.path.basename(sys.executable)
	assert any("stage=process" in line and f"tool={tool}" in line and "rc=0" in line for line in logged)
