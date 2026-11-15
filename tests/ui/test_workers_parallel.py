import pathlib
from typing import Any

from csvmusic.core import audio_quality as audio_quality_module
from csvmusic.ui import workers

SIGNALS = ("sig_log", "sig_warning", "sig_total", "sig_match_stats", "sig_row_status", "sig_progress", "sig_done", "sig_track_result")


def _norm(value: Any, tmp: pathlib.Path) -> Any:
	if isinstance(value, str):
		return value.replace(str(tmp), "<TMP>")
	if isinstance(value, bytes):
		return value.decode()
	if isinstance(value, dict):
		return {key: _norm(val, tmp) for key, val in value.items()}
	if isinstance(value, (list, tuple)):
		return [_norm(val, tmp) for val in value]
	return value


def _worker(tmp: pathlib.Path, tracks: list[dict], *, force_download: bool) -> tuple[workers.PipelineWorker, list]:
	worker = workers.PipelineWorker(
		csv_path="", out_dir=str(tmp / "out"), playlist="PL", fmt="mp3",
		write_m3u8=False, write_m3u_plain=False, embed_art=False,
		yt_dlp_path=None, ffmpeg_path_override=None, cookies_browser=None, cookies_file=None,
		force_download=force_download, tracks_override=tracks, row_indices=list(range(len(tracks))),
	)
	events: list = []
	for name in SIGNALS:
		getattr(worker, name).connect(lambda *args, _n=name: events.append((_n, _norm(list(args), tmp))))
	return worker, events


def _track(title: str) -> dict:
	return {"title": title, "artists": "Artist", "playlist": "PL", "duration_ms": 200000}


def _cand(vid: str, score: float, source: str = "music") -> dict:
	return {"videoId": vid, "title": "x", "author": "Artist", "source": source, "score": score}


def _patch_common(monkeypatch, worker, matches: dict, behaviors: dict, calls: list) -> None:
	monkeypatch.setattr(workers, "YTMusic", lambda: object())
	monkeypatch.setattr(workers, "find_best", lambda _yt, track: matches[track["title"]])
	monkeypatch.setattr(workers, "_track_cover_bytes", lambda _t, vid: b"cover-" + vid.encode())
	monkeypatch.setattr(workers, "tag_file", lambda *_a, **_k: None)
	monkeypatch.setattr(workers.time, "sleep", lambda _s: None)
	monkeypatch.setattr(audio_quality_module, "audio_quality", lambda fp: {"low": fp.name.startswith("Artist - lowq"), "codec": "aac", "kbps": 96})

	def fake_download(vid, dest, base, profile):
		calls.append((vid, profile.label))
		path = dest / f"{base}.mp3"
		behavior = behaviors[vid].pop(0) if isinstance(behaviors[vid], list) else behaviors[vid]
		if isinstance(behavior, Exception):
			raise behavior
		path.write_bytes(b"audio")
		return path

	monkeypatch.setattr(worker, "_download_with_profile", fake_download)


def _by_track(events: list) -> dict:
	groups: dict = {}
	for name, args in events:
		if name in ("sig_row_status", "sig_track_result"):
			groups.setdefault(args[0], []).append((name, args[1] if name == "sig_row_status" else _summary(args[1])))
	return groups


def _summary(payload: dict) -> dict:
	keys = ("skipped", "downloaded", "error", "forced_match", "low_confidence", "file_path", "confidence", "audio_quality")
	out = {key: payload.get(key) for key in keys}
	out["match"] = (payload.get("match") or {}).get("videoId")
	out["cover"] = payload.get("cover_bytes")
	return out


def test_sequential_signal_stream_without_force(monkeypatch, tmp_path):
	from csvmusic.core.downloader import AgeRestrictedError
	titles = ["ok1", "lowq", "skip", "fail", "rate", "age"]
	tracks = [_track(t) for t in titles]
	worker, events = _worker(tmp_path, tracks, force_download=False)
	matches = {
		"ok1": (_cand("v-ok1", 0.95), 0.95, [_cand("v-ok1", 0.95)]),
		"lowq": (_cand("v-lowq", 0.9), 0.9, [_cand("v-lowq", 0.9)]),
		"skip": (None, 0.4, [_cand("v-skip", 0.4)]),
		"fail": (_cand("v-fail", 0.9), 0.9, [_cand("v-fail", 0.9)]),
		"rate": (_cand("v-rate", 0.9), 0.9, [_cand("v-rate", 0.9)]),
		"age": (_cand("v-age", 0.9), 0.9, [_cand("v-age", 0.9)]),
	}
	behaviors = {
		"v-ok1": None, "v-lowq": None, "v-skip": None,
		"v-fail": RuntimeError("boom"),
		"v-rate": [RuntimeError("HTTP Error 429: Too Many Requests"), None],
		"v-age": AgeRestrictedError("age restricted"),
	}
	calls: list = []
	_patch_common(monkeypatch, worker, matches, behaviors, calls)
	worker.run()
	assert (events, calls) == (EXPECTED_STREAM, EXPECTED_CALLS)
	assert calls == EXPECTED_CALLS


def test_sequential_signal_stream_with_force_fallback(monkeypatch, tmp_path):
	tracks = [_track("forced"), _track("after")]
	worker, events = _worker(tmp_path, tracks, force_download=True)
	forced_opts = [_cand("v-f1", 0.5), _cand("v-f2", 0.45, "video")]
	matches = {
		"forced": (None, 0.5, forced_opts),
		"after": (_cand("v-after", 0.95), 0.95, [_cand("v-after", 0.95)]),
	}
	behaviors = {"v-f1": RuntimeError("bad first"), "v-f2": None, "v-after": None}
	calls: list = []
	_patch_common(monkeypatch, worker, matches, behaviors, calls)
	worker.run()
	assert (events, calls) == (EXPECTED_FORCE_STREAM, EXPECTED_FORCE_CALLS)
	assert calls == EXPECTED_FORCE_CALLS


EXPECTED_STREAM: list = [("sig_log", ["[csv] loading…"]),
	("sig_total", [6]),
	("sig_log", ["[match] searching on YouTube Music…"]),
	("sig_match_stats", [0, 0]),
	("sig_match_stats", [1, 0]),
	("sig_row_status", [0, "Downloading (mp3)…"]),
	("sig_row_status", [0, "Tagging…"]),
	("sig_row_status", [0, "Done → Artist - ok1.mp3"]),
	("sig_progress", [1, 6]),
	("sig_track_result",
	[0,
	{"audio_quality": {"codec": "aac", "kbps": 96, "low": False},
	"confidence": 0.95,
	"cover_bytes": "cover-v-ok1",
	"downloaded": True,
	"error": None,
	"file_path": "<TMP>/out/PL/Artist - ok1.mp3",
	"forced_match": False,
	"low_confidence": False,
	"match": {"author": "Artist", "score": 0.95, "source": "music", "title": "x", "videoId": "v-ok1"},
	"options": [{"author": "Artist", "score": 0.95, "source": "music", "title": "x", "videoId": "v-ok1"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "ok1"}}]),
	("sig_match_stats", [2, 0]),
	("sig_row_status", [1, "Downloading (mp3)…"]),
	("sig_row_status", [1, "Tagging…"]),
	("sig_row_status", [1, "Done → Artist - lowq.mp3"]),
	("sig_progress", [2, 6]),
	("sig_log", ["[warn] Low YouTube source bitrate: lowq — aac ~96 kbps"]),
	("sig_row_status", [1, "Downloaded — low quality"]),
	("sig_track_result",
	[1,
	{"audio_quality": {"codec": "aac", "kbps": 96, "low": True},
	"confidence": 0.9,
	"cover_bytes": "cover-v-lowq",
	"downloaded": True,
	"error": None,
	"file_path": "<TMP>/out/PL/Artist - lowq.mp3",
	"forced_match": False,
	"low_confidence": False,
	"match": {"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-lowq"},
	"options": [{"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-lowq"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "lowq"}}]),
	("sig_row_status", [2, "Skipped (low confidence)"]),
	("sig_progress", [3, 6]),
	("sig_track_result",
	[2,
	{"confidence": 0.4,
	"cover_bytes": None,
	"downloaded": False,
	"error": None,
	"file_path": None,
	"forced_match": False,
	"match": None,
	"options": [{"author": "Artist", "score": 0.4, "source": "music", "title": "x", "videoId": "v-skip"}],
	"playlist_name": "PL",
	"skipped": True,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "skip"}}]),
	("sig_match_stats", [2, 1]),
	("sig_match_stats", [3, 1]),
	("sig_row_status", [3, "Downloading (mp3)…"]),
	("sig_row_status", [3, "Fail: boom"]),
	("sig_progress", [4, 6]),
	("sig_track_result",
	[3,
	{"confidence": 0.9,
	"cover_bytes": None,
	"downloaded": False,
	"error": "boom",
	"file_path": None,
	"forced_match": False,
	"low_confidence": False,
	"match": {"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-fail"},
	"options": [{"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-fail"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "fail"}}]),
	("sig_match_stats", [4, 1]),
	("sig_row_status", [4, "Downloading (mp3)…"]),
	("sig_log",
	["[warn] YouTube reported a request limit. Applying slower download pacing automatically; waiting before retrying may still be "
	'necessary.\n'
	'\n'
	"Detected: YouTube returned HTTP 429"]),
	("sig_warning",
	['YouTube reported a request limit. Applying slower download pacing automatically; waiting before retrying may still be necessary.\n'
	'\n'
	"Detected: YouTube returned HTTP 429"]),
	("sig_row_status", [4, "Retrying with YouTube safe mode…"]),
	("sig_row_status", [4, "Tagging…"]),
	("sig_row_status", [4, "Done → Artist - rate.mp3"]),
	("sig_progress", [5, 6]),
	("sig_track_result",
	[4,
	{"audio_quality": {"codec": "aac", "kbps": 96, "low": False},
	"confidence": 0.9,
	"cover_bytes": "cover-v-rate",
	"downloaded": True,
	"error": None,
	"file_path": "<TMP>/out/PL/Artist - rate.mp3",
	"forced_match": False,
	"low_confidence": False,
	"match": {"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-rate"},
	"options": [{"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-rate"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "rate"}}]),
	("sig_match_stats", [5, 1]),
	("sig_row_status", [5, "Downloading (mp3)…"]),
	("sig_row_status", [5, "Fail: age restricted"]),
	("sig_progress", [6, 6]),
	("sig_track_result",
	[5,
	{"confidence": 0.9,
	"cover_bytes": None,
	"downloaded": False,
	"error": "age restricted",
	"file_path": None,
	"forced_match": False,
	"low_confidence": False,
	"match": {"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-age"},
	"options": [{"author": "Artist", "score": 0.9, "source": "music", "title": "x", "videoId": "v-age"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "age"}}]),
	("sig_done",
	["All tasks finished.",
	[{"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "ok1"},
	{"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "lowq"},
	{"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "rate"}],
	[{"options": [{"author": "Artist", "score": 0.4, "source": "music", "title": "x", "videoId": "v-skip"}],
		"reason": "No confident match",
		"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "skip"}}],
	[{"error": "boom", "track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "fail"}},
	{"error": "age restricted", "track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "age"}}]])]
EXPECTED_CALLS: list = [("v-ok1", "normal"), ("v-lowq", "normal"), ("v-fail", "normal"), ("v-rate", "normal"), ("v-rate", "aggressive"), ("v-age", "aggressive")]
EXPECTED_FORCE_STREAM: list = [("sig_log", ["[csv] loading…"]),
	("sig_total", [2]),
	("sig_log", ["[match] searching on YouTube Music…"]),
	("sig_match_stats", [0, 0]),
	("sig_match_stats", [1, 0]),
	("sig_row_status", [0, "Downloading low-confidence match (mp3)…"]),
	("sig_row_status", [0, "Trying official result 1/2 (mp3)…"]),
	("sig_row_status", [0, "Trying fallback YouTube result 2/2 (mp3)…"]),
	("sig_row_status", [0, "Tagging…"]),
	("sig_row_status", [0, "Low confidence → Artist - forced.mp3"]),
	("sig_progress", [1, 2]),
	("sig_track_result",
	[0,
	{"audio_quality": {"codec": "aac", "kbps": 96, "low": False},
	"confidence": 0.5,
	"cover_bytes": "cover-v-f2",
	"downloaded": True,
	"error": None,
	"file_path": "<TMP>/out/PL/Artist - forced.mp3",
	"forced_match": True,
	"low_confidence": True,
	"match": {"author": "Artist", "score": 0.45, "source": "video", "title": "x", "videoId": "v-f2"},
	"options": [{"author": "Artist", "score": 0.5, "source": "music", "title": "x", "videoId": "v-f1"},
				{"author": "Artist", "score": 0.45, "source": "video", "title": "x", "videoId": "v-f2"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "forced"}}]),
	("sig_match_stats", [2, 0]),
	("sig_row_status", [1, "Downloading (mp3)…"]),
	("sig_row_status", [1, "Tagging…"]),
	("sig_row_status", [1, "Done → Artist - after.mp3"]),
	("sig_progress", [2, 2]),
	("sig_track_result",
	[1,
	{"audio_quality": {"codec": "aac", "kbps": 96, "low": False},
	"confidence": 0.95,
	"cover_bytes": "cover-v-after",
	"downloaded": True,
	"error": None,
	"file_path": "<TMP>/out/PL/Artist - after.mp3",
	"forced_match": False,
	"low_confidence": False,
	"match": {"author": "Artist", "score": 0.95, "source": "music", "title": "x", "videoId": "v-after"},
	"options": [{"author": "Artist", "score": 0.95, "source": "music", "title": "x", "videoId": "v-after"}],
	"playlist_name": "PL",
	"skipped": False,
	"track": {"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "after"}}]),
	("sig_done",
	["All tasks finished.",
	[{"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "forced"},
	{"artists": "Artist", "duration_ms": 200000, "playlist": "PL", "title": "after"}],
	[],
	[]])]
EXPECTED_FORCE_CALLS: list = [("v-f1", "normal"), ("v-f2", "normal"), ("v-after", "normal")]
