import pathlib, subprocess

import pytest

from csvmusic.core import downloader
from csvmusic.core.downloader import DownloadCancelled, DownloadError
from csvmusic.core.track_output import expected_track_path, plan_track_outputs

FORMATS = {
	"mp3": downloader.download_mp3,
	"m4a": downloader.download_m4a,
	"opus": downloader.download_opus,
}
BASE = "Artist - Song"


def _fake_ytdlp(cmd: list[str]) -> tuple[int, str]:
	tpl = cmd[cmd.index("-o") + 1]
	pathlib.Path(tpl.replace("%(ext)s", "webm")).write_bytes(b"source")
	return 0, ""


def _fake_ffmpeg(mode: str):
	def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
		pathlib.Path(cmd[-1]).write_bytes(b"partial")
		if mode == "cancel":
			raise DownloadCancelled("cancelled")
		if mode == "timeout":
			raise DownloadError("timed out")
		return subprocess.CompletedProcess(cmd, 1 if mode == "rc" else 0, "", "boom")
	return run


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
	monkeypatch.setattr(downloader, "_run_ytdlp_detail", _fake_ytdlp)
	monkeypatch.setattr(downloader, "store_source", lambda path: None)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("mode,exc", [("cancel", DownloadCancelled), ("timeout", DownloadError), ("rc", DownloadError)])
def test_failed_encode_leaves_nothing(monkeypatch, tmp_path, fmt, mode, exc):
	monkeypatch.setattr(downloader, "_run_capture", _fake_ffmpeg(mode))
	with pytest.raises(exc):
		FORMATS[fmt](
			"vid", tmp_path, BASE, yt_dlp_bin="yt-dlp", ffmpeg_bin="ffmpeg", extra_yt_dlp_args=["--cookies", "c"]
		)
	assert list(tmp_path.iterdir()) == []

	track = {"playlist": "P", "title": "Song", "artists": "Artist"}
	out = expected_track_path(track, tmp_path, fmt)
	assert not out.exists()
	assert plan_track_outputs([track], tmp_path, fmt).queued_rows == (0,)


@pytest.mark.parametrize("fmt", FORMATS)
def test_successful_encode_leaves_only_final(monkeypatch, tmp_path, fmt):
	monkeypatch.setattr(downloader, "_run_capture", _fake_ffmpeg("ok"))
	result = FORMATS[fmt](
		"vid", tmp_path, BASE, yt_dlp_bin="yt-dlp", ffmpeg_bin="ffmpeg", extra_yt_dlp_args=["--cookies", "c"]
	)
	assert result == tmp_path / f"{BASE}.{fmt}"
	assert [p.name for p in tmp_path.iterdir()] == [f"{BASE}.{fmt}"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_stale_part_from_earlier_run_is_removed(monkeypatch, tmp_path, fmt):
	(tmp_path / f"{BASE}.part.{fmt}").write_bytes(b"old")
	monkeypatch.setattr(downloader, "_run_capture", _fake_ffmpeg("ok"))
	FORMATS[fmt]("vid", tmp_path, BASE, yt_dlp_bin="yt-dlp", ffmpeg_bin="ffmpeg", extra_yt_dlp_args=["--cookies", "c"])
	assert [p.name for p in tmp_path.iterdir()] == [f"{BASE}.{fmt}"]
