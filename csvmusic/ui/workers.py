# tabs only
from PySide6.QtCore import QObject, Signal, QThread
import os
import pathlib, traceback, time
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import subprocess
import json
import sqlite3
import re, unicodedata
from typing import Callable, List, Dict

from ytmusicapi import YTMusic

from csvmusic.core.csv_import import load_csv, tracks_from_csv
from csvmusic.core.url_import import fetch_music_url
from csvmusic.core.log import log
from csvmusic.core.timing import timed
from csvmusic.core.ytmusic_match import find_best, more_candidates, CONFIDENCE_MIN, SearchHealth
from csvmusic.core import download_pool
from csvmusic.core.download_pool import Completion, DownloadPool, JobContext, JobEvent, PoolPolicy, track_pause_s
from csvmusic.core.downloader import (
	download_m4a, download_mp3, download_opus, tag_file, yt_thumbnail_bytes, cover_art_url_bytes, write_m3u, sanitize_name,
	youtube_batch_mitigation, build_ytdlp_mitigation_args, detect_youtube_risk, output_collision_key,
	YOUTUBE_MITIGATION_AGGRESSIVE, YouTubeMitigationProfile, AuthRequiredError, DownloadCancelled,
	ACCOUNT_LANE, ACCOUNT_LANE_LIMIT, auth_escalation
)
from csvmusic.core.paths import ytdlp_path as _resolve_ytdlp, INTERNAL_YTDLP
from csvmusic.core.subprocess_env import subprocess_kwargs
from csvmusic.core.update_check import UpdateInfo, fetch_available_update
from csvmusic.core.track_output import expected_track_path, track_filename_base

_FORCE_FALLBACK_MIN_SCORE = 0.45


class UpdateCheckWorker(QThread):
	sig_done = Signal(object)

	def __init__(self, current_version: str, parent: QObject | None = None):
		super().__init__(parent)
		self.current_version = current_version

	def run(self):
		update: UpdateInfo | None = None
		try:
			update = fetch_available_update(self.current_version)
		except Exception as exc:
			log(f"update check unavailable: {exc}")
		self.sig_done.emit(update)

class MusicURLImportWorker(QThread):
	sig_done = Signal(bool, dict, str)

	def __init__(self, playlist_url: str, parent: QObject | None = None):
		super().__init__(parent)
		self.playlist_url = playlist_url

	def run(self):
		try:
			playlist = fetch_music_url(self.playlist_url)
			self.sig_done.emit(True, {
				"id": playlist.id,
				"name": playlist.name,
				"tracks": playlist.tracks,
				"total_count": playlist.total_count,
				"source_type": playlist.source_type,
				"platform": playlist.platform,
				"warning": playlist.warning,
			}, "")
		except Exception as exc:
			log(f"url import failure: url='{self.playlist_url}' error={exc}")
			self.sig_done.emit(False, {}, str(exc))

def _legacy_cover_size(legacy_options: Dict | None, *, embed_art: bool) -> int:
	if not embed_art:
		return 0
	mode = str((legacy_options or {}).get("cover_art_mode") or "standard").lower()
	if mode == "off":
		return 0
	if mode == "small":
		return 300
	if mode == "medium":
		return 450
	return 600


def _track_cover_bytes(track: Dict, video_id: str) -> bytes | None:
	"""Prefer source artwork, but never suppress the YouTube fallback."""
	cover = cover_art_url_bytes(track.get("cover_url"))
	if cover:
		return cover
	return yt_thumbnail_bytes(video_id)

def _legacy_cbr_bitrate(legacy_options: Dict | None) -> int | None:
	if not legacy_options or not legacy_options.get("enabled"):
		return None
	mode = str(legacy_options.get("mp3_mode") or "").lower()
	if mode == "cbr_192":
		return 192
	if mode == "cbr_256":
		return 256
	if mode == "cbr_320":
		return 320
	return None

def _norm_text(text: str) -> str:
	return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", (text or "").casefold())).strip()

def _tokens(text: str) -> set[str]:
	return {tok for tok in re.findall(r"\w+", _norm_text(text), flags=re.UNICODE) if any(ch.isalnum() for ch in tok)}

def _run_yt_dlp_command(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
	if cmd and cmd[0] == INTERNAL_YTDLP:
		stdout_buf: list[str] = []
		stderr_buf: list[str] = []
		try:
			from yt_dlp import main as yt_dlp_main
		except Exception as exc:
			return subprocess.CompletedProcess(cmd, 1, "", f"failed to import yt_dlp module: {exc}")
		import io
		import contextlib
		out = io.StringIO()
		err = io.StringIO()
		with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
			try:
				rc = yt_dlp_main(cmd[1:])
			except SystemExit as exc:
				code = exc.code
				if isinstance(code, int):
					rc = code
				elif code is None:
					rc = 0
				else:
					rc = 1
			except Exception as exc:
				rc = 1
				err.write(str(exc))
		return subprocess.CompletedProcess(cmd, rc, out.getvalue(), err.getvalue())
	return subprocess.run(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		encoding="utf-8",
		errors="replace",
		timeout=timeout,
		**subprocess_kwargs()
	)

def _new_payload(track: Dict, options: List[Dict], match: Dict | None, confidence: float, playlist_name: str, forced_match: bool) -> Dict:
	return {
		"track": track,
		"options": options,
		"match": match,
		"confidence": confidence,
		"skipped": False,
		"error": None,
		"playlist_name": playlist_name,
		"file_path": None,
		"downloaded": False,
		"cover_bytes": None,
		"forced_match": forced_match
	}


AUTH_ANON = "anon"
AUTH_ACCOUNT = ACCOUNT_LANE  # the account is also the pool lane that paces it


@dataclass(frozen=True)
class TrackJob:
	row_idx: int
	track: Dict
	match: Dict
	options: List[Dict]
	confidence: float
	low_confidence: bool
	forced_match: bool
	candidate_sequence: List[Dict]
	dest_dir: pathlib.Path
	base: str
	profile: YouTubeMitigationProfile
	audio_processing: Dict
	safe_mode: bool
	playlist_name: str
	track_playlist_name: str
	auth: str = AUTH_ANON  # AUTH_ANON | AUTH_ACCOUNT: only the coordinator moves a job to the account


@dataclass(frozen=True)
class _JobContext:
	emit: Callable[..., None]
	audio_processing: Dict
	auth: str


class _NeedsAccount(Exception):
	"""An anonymous attempt needs the account (age, private, members-only); index is the candidate to resume from."""

	def __init__(self, index: int) -> None:
		super().__init__("account required")
		self.index = index


# Per-job state read by the download helpers, so jobs never share mutable worker attributes.
_JOB_CONTEXT: ContextVar[_JobContext | None] = ContextVar("csvmusic_job_context", default=None)


@dataclass
class _MatchSkip:
	row_idx: int
	track: Dict
	payload: Dict
	status: str
	reason: str
	options: List[Dict]
	abort_reason: str | None


@dataclass
class _BatchState:
	total: int
	playlist_name: str
	health: SearchHealth = field(default_factory=SearchHealth)
	matched: int = 0
	skipped_count: int = 0
	processed: int = 0
	abort_reason: str | None = None
	next_match_at: float = 0.0
	# Rows are (csv index, entry): tracks finish out of order, results are reported in CSV order.
	done_tracks: List[tuple[int, Dict]] = field(default_factory=list)
	failed_tracks: List[tuple[int, Dict]] = field(default_factory=list)
	skipped_tracks: List[tuple[int, Dict]] = field(default_factory=list)

	def ordered(self, rows: List[tuple[int, Dict]]) -> List[Dict]:
		return [entry for _idx, entry in sorted(rows, key=lambda row: row[0])]


@dataclass
class TrackOutcome:
	row_idx: int
	payload: Dict
	fp: pathlib.Path | None = None
	cover: bytes | None = None
	error: str | None = None
	risk_reason: str | None = None
	account_from: int | None = None  # candidate index to retry with the account (age restriction)


class PipelineWorker(QThread):
	sig_log = Signal(str)                       # log strings
	sig_warning = Signal(str)                   # warning dialog text
	sig_total = Signal(int)                     # total tracks queued
	sig_match_stats = Signal(int, int)          # matched, skipped
	sig_row_status = Signal(int, str)           # row_index, status text
	sig_progress = Signal(int, int)             # processed, total
	sig_done = Signal(str, list, list, list)    # final message, matched, skipped, failed
	sig_track_result = Signal(int, dict)        # per-track summary

	def __init__(self, csv_path: str, out_dir: str, playlist: str | None,
	             fmt: str,
	             write_m3u8: bool, write_m3u_plain: bool,
	             embed_art: bool,
	             yt_dlp_path: str | None,
	             ffmpeg_path_override: str | None,
	             cookies_browser: str | None,
	             cookies_file: str | None,
	             audio_processing: Dict | None = None,
	             mp3_quality: int = 0,
	             legacy_options: Dict | None = None,
	             force_download: bool = False,
	             tracks_override: List[Dict] | None = None,
	             m3u_tracks_override: List[Dict] | None = None,
	             m3u_output_dir: str | None = None,
	             row_indices: List[int] | None = None,
	             max_workers: int | None = 1,
	             parent: QObject | None = None):
		super().__init__(parent)
		self.max_workers = max_workers
		self.csv_path = csv_path
		self.out_dir = pathlib.Path(out_dir)
		self.playlist = playlist
		self.fmt = fmt
		self.write_m3u8 = write_m3u8
		self.write_m3u_plain = write_m3u_plain
		self.embed_art = embed_art
		self.yt_dlp_path = yt_dlp_path
		self.ffmpeg_path_override = ffmpeg_path_override
		self.cookies_browser = cookies_browser
		self.cookies_file = cookies_file
		self.audio_processing = audio_processing or {}
		self.mp3_quality = max(0, min(10, int(mp3_quality)))
		self.legacy_options = legacy_options or {}
		self.force_download = bool(force_download)
		self.tracks_override = tracks_override
		self.m3u_tracks_override = m3u_tracks_override
		self.m3u_output_dir = pathlib.Path(m3u_output_dir) if m3u_output_dir else None
		self.row_indices = row_indices or []
		self._stop = False
		self._pool: DownloadPool | None = None

	def stop(self):
		self._stop = True
		pool = self._pool
		if pool is not None:
			pool.cancel()

	def _download_with_profile(self, vid: str, dest_dir: pathlib.Path, base: str, profile: YouTubeMitigationProfile):
		from csvmusic.core.downloader import take_download_notices, download_notice_handler
		take_download_notices()
		try:
			with download_notice_handler(lambda notice: self._job_context().emit("log", f"[recovery] {notice}")):
				return self._download_with_profile_impl(vid, dest_dir, base, profile)
		finally:
			take_download_notices()

	def _download_with_profile_impl(self, vid: str, dest_dir: pathlib.Path, base: str, profile: YouTubeMitigationProfile):
		audio_processing = self._job_context().audio_processing
		extra_args: list[str] = []
		if self._job_context().auth == AUTH_ACCOUNT:
			if self.cookies_file:
				extra_args += ["--cookies", self.cookies_file]
			elif self.cookies_browser:
				extra_args += ["--cookies-from-browser", self.cookies_browser]
		elif self._account_available:
			# A yt-dlp config file with cookies must not authenticate the anonymous attempt.
			extra_args += ["--no-cookies", "--no-cookies-from-browser"]
		extra_args += build_ytdlp_mitigation_args(profile)
		if self.fmt == "m4a":
			return download_m4a(vid, dest_dir, base, yt_dlp_bin=self.yt_dlp_path, ffmpeg_bin=self.ffmpeg_path_override, extra_yt_dlp_args=extra_args or None, audio_processing=audio_processing)
		if self.fmt == "opus":
			return download_opus(vid, dest_dir, base, yt_dlp_bin=self.yt_dlp_path, ffmpeg_bin=self.ffmpeg_path_override, extra_yt_dlp_args=extra_args or None)
		return download_mp3(vid, dest_dir, base, yt_dlp_bin=self.yt_dlp_path, ffmpeg_bin=self.ffmpeg_path_override, extra_yt_dlp_args=extra_args or None, audio_processing=audio_processing, mp3_quality=self.mp3_quality, cbr_bitrate_kbps=_legacy_cbr_bitrate(self.legacy_options))

	@property
	def _account_available(self) -> bool:
		return bool(self.cookies_file or self.cookies_browser)

	def _job_context(self) -> _JobContext:
		ctx = _JOB_CONTEXT.get()
		if ctx is None:
			raise RuntimeError("no active download job")
		return ctx

	def _track_audio_processing(self, track: Dict) -> Dict:
		processing = dict(self.audio_processing)
		track_gain = max(-12, min(12, int(track.get("audio_volume_gain", 0) or 0)))
		if track_gain:
			processing["enabled"] = True
			processing["volume_gain"] = int(processing.get("volume_gain", 0) or 0) + track_gain
		return processing

	def _apply_mitigation(self, pool: DownloadPool, profile: YouTubeMitigationProfile, reason: str | None = None) -> None:
		if profile.label == pool.profile.label:
			return
		pool.apply_profile(profile)
		if profile.warning:
			msg = profile.warning
			if reason:
				msg = f"{msg}\n\nDetected: {reason}"
			self.sig_log.emit(f"[warn] {msg}")
			self.sig_warning.emit(msg)

	def _is_official_candidate(self, cand: Dict) -> bool:
		source = str(cand.get("source") or "").lower()
		author = str(cand.get("author") or "").lower()
		title = str(cand.get("title") or "").lower()
		if source == "music":
			return True
		return any(term in author for term in ("topic", "official", "vevo")) or "official" in title

	def _candidate_relevant_to_track(self, track: Dict, cand: Dict) -> bool:
		track_artist_tokens = _tokens(track.get("artists", ""))
		track_title_tokens = _tokens(track.get("title", ""))
		cand_author_tokens = _tokens(cand.get("author", ""))
		cand_title_tokens = _tokens(cand.get("title", ""))
		artist_ok = bool(track_artist_tokens & (cand_author_tokens | cand_title_tokens))
		title_overlap = len(track_title_tokens & cand_title_tokens)
		return artist_ok and title_overlap >= max(1, min(2, len(track_title_tokens)))

	def _force_download_candidates(self, track: Dict, options: List[Dict]) -> List[Dict]:
		if not options:
			return []
		relevant = [opt for opt in options if self._candidate_relevant_to_track(track, opt)]
		pool = relevant or options
		official = [opt for opt in pool if self._is_official_candidate(opt)]
		standard = [opt for opt in pool if opt not in official]

		sequence: list[Dict] = []

		def _append_group(group: list[Dict]) -> None:
			if not group:
				return
			sequence.append(group[0])
			if len(group) > 1 and float(group[1].get("score", 0.0) or 0.0) >= _FORCE_FALLBACK_MIN_SCORE:
				sequence.append(group[1])

		_append_group(official)
		_append_group(standard)

		seen: set[str] = set()
		unique: list[Dict] = []
		for opt in sequence:
			vid = opt.get("videoId")
			if not vid or vid in seen:
				continue
			seen.add(vid)
			unique.append(opt)
		return unique

	def _ordered_force_candidates(self, track: Dict, match: Dict | None, options: List[Dict]) -> List[Dict]:
		if not self.force_download:
			return [match] if match else []
		ordered = self._force_download_candidates(track, options)
		if match and match.get("videoId"):
			match_vid = match.get("videoId")
			if not ordered:
				return [match]
			if ordered[0].get("videoId") != match_vid:
				ordered = [match] + [opt for opt in ordered if opt.get("videoId") != match_vid]
		return ordered or ([match] if match else [])

	def _attempt_status_text(self, candidate: Dict, attempt_idx: int, total_attempts: int, *, safe_mode: bool = False) -> str:
		source_label = "official result" if self._is_official_candidate(candidate) else "YouTube result"
		if total_attempts <= 1:
			base = f"Trying {source_label} ({self.fmt})…"
		elif attempt_idx == 1:
			base = f"Trying {source_label} 1/{total_attempts} ({self.fmt})…"
		else:
			base = f"Trying fallback {source_label} {attempt_idx}/{total_attempts} ({self.fmt})…"
		if safe_mode:
			return f"Safe mode: {base[0].lower()}{base[1:]}"
		return base

	def _emit(self, kind: str, *args) -> None:
		{"status": self.sig_row_status, "warning": self.sig_warning, "log": self.sig_log}[kind].emit(*args)

	def _report_quality(self, outcome: "TrackOutcome", title: str) -> None:
		if outcome.error is not None:
			return
		from csvmusic.core.audio_quality import audio_quality
		quality = audio_quality(outcome.fp)
		outcome.payload["audio_quality"] = quality
		if quality["low"]:
			self._emit("log", f"[warn] Low YouTube source bitrate: {title} — {quality['codec']} ~{quality['kbps']} kbps")
			self._emit("status", outcome.row_idx, "Downloaded — low quality")

	def _try_candidates(self, job: "TrackJob", emit):
		stage = (lambda _name, **_fields: nullcontext()) if job.safe_mode else timed
		sequence = job.candidate_sequence
		fallback_attempts = self.force_download and len(sequence) > 1
		last_err = None
		fp = None
		cover = None
		chosen = None
		for attempt_idx, candidate in enumerate(sequence, start=1):
			vid = candidate["videoId"]
			if fallback_attempts:
				emit("status", job.row_idx, self._attempt_status_text(candidate, attempt_idx, len(sequence), safe_mode=job.safe_mode))
			try:
				with stage("download", auth=job.auth):
					fp = self._download_with_profile(vid, job.dest_dir, job.base, job.profile)
				emit("status", job.row_idx, "Tagging…")
				with stage("cover"):
					cover = _track_cover_bytes(job.track, vid)
				tag_file(fp, job.track, cover if self.embed_art else None, cover_size=_legacy_cover_size(self.legacy_options, embed_art=self.embed_art))
				chosen = candidate
				break
			except AuthRequiredError:
				if job.auth == AUTH_ANON and self._account_available:
					raise _NeedsAccount(attempt_idx - 1)
				raise
			except DownloadCancelled:
				raise
			except Exception as candidate_exc:
				last_err = str(candidate_exc)
		if fp is None:
			raise RuntimeError(last_err or "Download failed.")
		return fp, cover, chosen

	def _process_track(self, job: "TrackJob", ctx: JobContext) -> "TrackOutcome":
		token = _JOB_CONTEXT.set(_JobContext(ctx.emit, job.audio_processing, job.auth))
		# The anonymous attempt of a job that has an account to fall back on: auth failures end it at once.
		escalate = auth_escalation() if job.auth == AUTH_ANON and self._account_available else nullcontext()
		try:
			started = time.monotonic()
			ok = False
			try:
				with escalate:
					outcome = self._process_track_in_context(job, ctx)
				ok = outcome.error is None
				return outcome
			finally:
				log(f"timing: stage=track seconds={time.monotonic() - started:.1f} row={job.row_idx} ok={ok} auth={job.auth}")
		finally:
			_JOB_CONTEXT.reset(token)

	def _process_track_in_context(self, job: "TrackJob", ctx: JobContext) -> "TrackOutcome":
		emit = ctx.emit
		payload = _new_payload(job.track, job.options, job.match, job.confidence, job.track_playlist_name, job.forced_match)
		payload["low_confidence"] = job.low_confidence
		outcome = TrackOutcome(job.row_idx, payload)
		try:
			fp, cover, chosen = self._try_candidates(job, emit)
			if chosen is not None:
				payload["match"] = chosen
			emit("status", job.row_idx, f"{'Low confidence' if job.low_confidence else 'Done'} → {fp.name}")
			outcome.fp = fp
			outcome.cover = cover
			payload.update(error=None, downloaded=True, file_path=str(fp), cover_bytes=cover)
			return outcome
		except DownloadCancelled:
			raise
		except _NeedsAccount as need:
			outcome.account_from = need.index
			return outcome
		except Exception as exc:
			err = str(exc)
			risk_reason = detect_youtube_risk(err)
			if risk_reason and not job.safe_mode and job.profile.label != YOUTUBE_MITIGATION_AGGRESSIVE.label:
				outcome.risk_reason = risk_reason
				return outcome
			log(f"download failure: playlist='{job.playlist_name}' track='{job.track['artists']} — {job.track['title']}' fmt={self.fmt} error={err}")
			emit("status", job.row_idx, f"Fail: {err[:120]}")
			outcome.error = err
			payload["error"] = err
			return outcome

	def _match_track(self, yt, state: "_BatchState", row_idx: int, t: Dict, profile: YouTubeMitigationProfile) -> "TrackJob | _MatchSkip | TrackOutcome | None":
		playlist_name = state.playlist_name
		track_playlist_name = self.playlist or t.get("playlist") or playlist_name
		dest_dir = self.out_dir / (sanitize_name(track_playlist_name) or "Playlist")
		dest_dir.mkdir(parents=True, exist_ok=True)
		title = t["title"]
		artists = t["artists"]
		audio_processing = self._track_audio_processing(t)
		search_error = None
		options: List[Dict] = []
		match = None
		confidence = 0.0
		try:
			preferred_video_id = str(t.get("preferred_video_id") or "").strip()
			if preferred_video_id:
				preferred_title = str(t.get("youtube_video_title") or t.get("preferred_video_label") or "").strip()
				if preferred_title.startswith(("http://", "https://")):
					preferred_title = title
				match = {
					"videoId": preferred_video_id,
					"title": preferred_title or title,
					"author": t.get("youtube_video_author") or artists,
					"score": 1.0,
				}
				options = [match]
				confidence = 1.0
			else:
				with timed("match"):
					match, confidence, options = find_best(yt, t)
		except Exception as exc:
			search_error = str(exc)
		abort_reason = state.health.record(search_error, bool(options))
		forced = False
		if match is None and self.force_download and options:
			forced_candidates = self._force_download_candidates(t, options)
			match = forced_candidates[0] if forced_candidates else options[0]
			confidence = float(match.get("score", confidence or 0.0) or 0.0)
			forced = True
		payload = _new_payload(t, options, match, confidence, track_playlist_name, forced)

		if match is None:
			if search_error:
				log(f"match skip: query='{t['title']} {t['artists']}' error={search_error}")
				status = f"Search failed: {search_error[:100]}"
			elif not options:
				status = "Skipped (no search results)"
			else:
				log(f"match skip: query='{t['title']} {t['artists']}' no candidate >= threshold (confidence={confidence:.2f})")
				status = "Skipped (low confidence)"
			payload["skipped"] = True
			payload["error"] = search_error
			reason = search_error or ("No search results" if not options else "No confident match")
			return _MatchSkip(row_idx, t, payload, status, reason, options, abort_reason)

		if self._stop:
			return None
		state.matched += 1
		self.sig_match_stats.emit(state.matched, state.skipped_count)
		low_confidence = bool(forced or confidence < CONFIDENCE_MIN)
		self.sig_row_status.emit(row_idx, f"Downloading low-confidence match ({self.fmt})…" if low_confidence else f"Downloading ({self.fmt})…")
		try:
			return TrackJob(
				row_idx=row_idx, track=t, match=match, options=options, confidence=confidence,
				low_confidence=low_confidence, forced_match=forced,
				candidate_sequence=self._ordered_force_candidates(t, match, options),
				dest_dir=dest_dir, base=track_filename_base(t), profile=profile,
				audio_processing=audio_processing, safe_mode=False,
				playlist_name=playlist_name, track_playlist_name=track_playlist_name,
			)
		except Exception as exc:
			return self._fail_outcome(row_idx, t, payload, match, low_confidence, exc, playlist_name)

	def _fail_outcome(self, row_idx: int, t: Dict, payload: Dict, match: Dict, low_confidence: bool, exc: Exception, playlist_name: str) -> "TrackOutcome":
		err = str(exc)
		log(f"download failure: playlist='{playlist_name}' track='{t['artists']} — {t['title']}' fmt={self.fmt} error={err}")
		self.sig_row_status.emit(row_idx, f"Fail: {err[:120]}")
		payload.update(match=match, low_confidence=low_confidence, error=err)
		return TrackOutcome(row_idx, payload, error=err)

	def _job_failure(self, job: "TrackJob", exc: BaseException) -> "TrackOutcome":
		payload = _new_payload(job.track, job.options, job.match, job.confidence, job.track_playlist_name, job.forced_match)
		return self._fail_outcome(job.row_idx, job.track, payload, job.match, job.low_confidence, exc, job.playlist_name)

	def _record_skip(self, state: "_BatchState", idx: int, skip: "_MatchSkip") -> None:
		state.skipped_tracks.append((idx, {"track": skip.track, "reason": skip.reason, "options": skip.options}))
		self.sig_row_status.emit(skip.row_idx, skip.status)
		state.processed += 1
		self.sig_progress.emit(state.processed, state.total)
		self.sig_track_result.emit(skip.row_idx, skip.payload)
		state.skipped_count += 1
		self.sig_match_stats.emit(state.matched, state.skipped_count)

	def _finish_track(self, pool: DownloadPool, state: "_BatchState", idx: int, t: Dict, outcome: "TrackOutcome") -> None:
		if outcome.error is None:
			state.done_tracks.append((idx, t))
		else:
			state.failed_tracks.append((idx, {"track": t, "error": outcome.error}))
		state.processed += 1
		self.sig_progress.emit(state.processed, state.total)
		self._report_quality(outcome, t["title"])
		self.sig_track_result.emit(outcome.row_idx, outcome.payload)
		self._pace(pool, state)

	def _pace(self, pool: DownloadPool, state: "_BatchState", gap: float | None = None) -> None:
		"""Set the earliest start of the next match; the coordinator waits for it in pool.pump, never in sleep."""
		if self._stop or state.processed >= state.total:
			return
		if gap is None:
			gap = track_pause_s(pool.profile)
		state.next_match_at = max(state.next_match_at, time.monotonic() + gap)

	@staticmethod
	def _lane(job: TrackJob) -> str | None:
		return ACCOUNT_LANE if job.auth == AUTH_ACCOUNT else None

	def _resubmit(self, pool: DownloadPool, done: Completion, *, auth: str, safe_mode: bool, status: str, candidate_sequence: List[Dict] | None = None) -> None:
		"""The only way a finished job runs again: same row, new auth/mode, ahead of the queue."""
		job: TrackJob = done.item
		if not self._stop:
			self.sig_row_status.emit(job.row_idx, status)
			retry = replace(
				job, auth=auth, safe_mode=safe_mode,
				profile=YOUTUBE_MITIGATION_AGGRESSIVE if safe_mode else job.profile,
				candidate_sequence=job.candidate_sequence if candidate_sequence is None else candidate_sequence,
			)
			if pool.submit(done.index, retry, key=output_collision_key(job.dest_dir, job.base), priority=True, lane=self._lane(retry)):
				return
		self.sig_row_status.emit(job.row_idx, "Stopped")

	def _retry_in_safe_mode(self, pool: DownloadPool, done: Completion) -> None:
		if not self._stop:
			self._apply_mitigation(pool, YOUTUBE_MITIGATION_AGGRESSIVE, done.result.risk_reason)
		# Safe mode keeps the login's benefit against bot checks, at the account lane's minimal pace.
		auth = AUTH_ACCOUNT if self._account_available else AUTH_ANON
		self._resubmit(pool, done, auth=auth, safe_mode=True, status="Retrying with YouTube safe mode…")

	def _retry_with_account(self, pool: DownloadPool, done: Completion) -> None:
		job: TrackJob = done.item
		self._resubmit(
			pool, done, auth=AUTH_ACCOUNT, safe_mode=job.safe_mode, status="Needs sign-in: retrying with your YouTube account…",
			candidate_sequence=job.candidate_sequence[done.result.account_from:],
		)

	def _on_completion(self, pool: DownloadPool, state: "_BatchState", done: Completion) -> None:
		job: TrackJob = done.item
		if done.cancelled:
			self.sig_row_status.emit(job.row_idx, "Stopped")
			return
		outcome = self._job_failure(job, done.error) if done.error is not None else done.result
		if outcome.risk_reason:
			self._retry_in_safe_mode(pool, done)
			return
		if outcome.account_from is not None:
			self._retry_with_account(pool, done)
			return
		self._finish_track(pool, state, done.index, job.track, outcome)
		if outcome.error is None:
			pool.track_done()  # throughput = tracks downloaded; retries, escalations and failures are not samples

	def _consume(self, pool: DownloadPool, state: "_BatchState", timeout: float = 0.0) -> None:
		for msg in pool.pump(timeout):
			if isinstance(msg, JobEvent):
				self._emit(msg.kind, *msg.args)
			else:
				self._on_completion(pool, state, msg)

	def _pool_policy(self) -> PoolPolicy:
		in_process = False
		if self.max_workers != 1:
			try:
				in_process = (self.yt_dlp_path or _resolve_ytdlp()) == INTERNAL_YTDLP
			except RuntimeError:
				pass  # no yt-dlp at all: downloads fail with their own error, it is not the in-process case
		lanes = {ACCOUNT_LANE: ACCOUNT_LANE_LIMIT} if self._account_available else {}
		return PoolPolicy(self.max_workers, in_process, os.cpu_count(), lanes)

	def _wait_for_slot(self, pool: DownloadPool, state: "_BatchState") -> None:
		"""Serve the pool until a slot is free and the pacing deadline has passed (stop ends the wait at once)."""
		self._consume(pool, state)
		while not self._stop:
			remaining = state.next_match_at - time.monotonic()
			if pool.has_capacity() and remaining <= 0:
				return
			self._consume(pool, state, min(remaining, 0.2) if remaining > 0 else 0.2)

	def _feed_pool(self, pool: DownloadPool, yt, state: "_BatchState", tracks: List[Dict]) -> None:
		for idx, track in enumerate(tracks):
			self._wait_for_slot(pool, state)
			if self._stop:
				break
			row_idx = self.row_indices[idx] if idx < len(self.row_indices) else idx
			step = self._match_track(yt, state, row_idx, track, pool.job_profile())
			self._pace(pool, state, download_pool.MATCH_GAP_S)
			if step is None:
				self.sig_row_status.emit(row_idx, "Stopped")
				break
			if isinstance(step, _MatchSkip):
				self._record_skip(state, idx, step)
				if step.abort_reason:
					state.abort_reason = step.abort_reason
					self.sig_warning.emit(step.abort_reason)
					pool.close()
					break
				self._pace(pool, state)
			elif isinstance(step, TrackOutcome):
				self._finish_track(pool, state, idx, track, step)
			elif not pool.submit(idx, step, key=output_collision_key(step.dest_dir, step.base)):
				self.sig_row_status.emit(row_idx, "Stopped")
				break
		while not pool.idle():
			self._consume(pool, state, 0.1)

	def _write_playlists(self, state: "_BatchState") -> None:
		done_tracks = state.ordered(state.done_tracks)
		if done_tracks:
			ext = self.fmt if self.fmt in ("m4a", "mp3", "opus") else "mp3"
			by_playlist: dict[str, list[dict]] = {}
			manifest_tracks = self.m3u_tracks_override if self.m3u_tracks_override is not None else done_tracks
			for completed_track in manifest_tracks:
				completed_playlist = self.playlist or completed_track.get("playlist") or state.playlist_name
				expected_file = expected_track_path(completed_track, self.out_dir, ext)
				if expected_file.exists():
					by_playlist.setdefault(completed_playlist, []).append(completed_track)
			for completed_playlist, playlist_tracks in by_playlist.items():
				if self.write_m3u8:
					m3u = write_m3u(self.out_dir, completed_playlist, playlist_tracks, ext, suffix=".m3u8", encoding="utf-8", playlist_output_dir=self.m3u_output_dir)
					self.sig_log.emit(f"[m3u] wrote: {m3u}")
				if self.write_m3u_plain:
					m3u_plain = write_m3u(self.out_dir, completed_playlist, playlist_tracks, ext, suffix=".m3u", encoding="utf-8-sig", playlist_output_dir=self.m3u_output_dir)
					self.sig_log.emit(f"[m3u] wrote: {m3u_plain}")
		msg = "All tasks finished."
		if self._stop:
			msg = "Stopped (partial results saved)."
		elif state.abort_reason:
			msg = f"Stopped because search is unavailable. {state.abort_reason}"
		self.sig_done.emit(msg, done_tracks, state.ordered(state.skipped_tracks), state.ordered(state.failed_tracks))

	def run(self):
		try:
			self.sig_log.emit("[csv] loading…")
			if self.tracks_override is not None:
				tracks = list(self.tracks_override)
			else:
				df = load_csv(self.csv_path)
				tracks = tracks_from_csv(df, self.playlist)
			if not tracks:
				self.sig_done.emit("No tracks selected.", [], [], [])
				return
			total = len(tracks)
			self.sig_total.emit(total)
			mitigation = youtube_batch_mitigation(total, using_cookies=bool(self.cookies_file or self.cookies_browser))
			if mitigation.warning:
				self.sig_log.emit(f"[warn] {mitigation.warning}")
			self.sig_log.emit("[match] searching on YouTube Music…")
			self.sig_match_stats.emit(0, 0)
			try:
				yt = YTMusic()
			except Exception as exc:
				raise RuntimeError(f"Failed to initialize YTMusic client: {exc}")
			playlist_name = self.playlist or (tracks[0]["playlist"] if tracks else "Playlist")
			if not playlist_name:
				playlist_name = "Playlist"
			state = _BatchState(total=total, playlist_name=playlist_name)
			(self.out_dir / (sanitize_name(playlist_name) or "Playlist")).mkdir(parents=True, exist_ok=True)
			policy = self._pool_policy()
			with DownloadPool(self._process_track, policy, mitigation) as pool:
				self._pool = pool
				try:
					if self._stop:  # stop() may have run before the pool existed
						pool.cancel()
					if policy.in_process_ytdlp:
						self.sig_log.emit("Parallel downloads need a standalone yt-dlp; downloading one at a time.")
					self._feed_pool(pool, yt, state, tracks)
				finally:
					self._pool = None
			self._write_playlists(state)
		except Exception:
			self.sig_done.emit("Fatal error:\n" + traceback.format_exc(), [], [], [])



class SingleDownloadWorker(QThread):
	sig_status = Signal(int, str)
	sig_finished = Signal(int, dict)

	def __init__(self, row_idx: int, track: Dict, match: Dict, out_dir: str,
	             fmt: str, embed_art: bool,
	             yt_dlp_path: str | None,
	             ffmpeg_path_override: str | None,
	             cookies_browser: str | None,
	             cookies_file: str | None,
	             audio_processing: Dict | None = None,
	             mp3_quality: int = 0,
	             legacy_options: Dict | None = None,
	             force_download: bool = False,
	             parent: QObject | None = None):
		super().__init__(parent)
		self.row_idx = row_idx
		self.track = track
		self.match = match
		self.out_dir = pathlib.Path(out_dir)
		self.fmt = fmt
		self.embed_art = embed_art
		self.playlist_name = track.get("playlist") or "Playlist"
		self.yt_dlp_path = yt_dlp_path
		self.ffmpeg_path_override = ffmpeg_path_override
		self.cookies_browser = cookies_browser
		self.cookies_file = cookies_file
		self.audio_processing = audio_processing or {}
		self.mp3_quality = max(0, min(10, int(mp3_quality)))
		self.legacy_options = legacy_options or {}
		self.force_download = bool(force_download)

	def run(self):
		from csvmusic.core.downloader import download_notice_handler, take_download_notices
		take_download_notices()
		try:
			with download_notice_handler(lambda notice: self.sig_status.emit(self.row_idx, notice)):
				self._run_download()
		finally:
			take_download_notices()

	def _run_download(self):
		try:
			safe_playlist = sanitize_name(self.playlist_name) or "Playlist"
			dest_dir = self.out_dir / safe_playlist
			dest_dir.mkdir(parents=True, exist_ok=True)

			base = track_filename_base(self.track)
			vid = self.match.get("videoId")
			self.sig_status.emit(self.row_idx, f"Downloading ({self.fmt})…")
			if self.cookies_file:
				cookies_args = ["--cookies", self.cookies_file]
			elif self.cookies_browser:
				cookies_args = ["--cookies-from-browser", self.cookies_browser]
			else:
				cookies_args = None
			if self.fmt == "m4a":
				fp = download_m4a(vid, dest_dir, base, yt_dlp_bin=self.yt_dlp_path, ffmpeg_bin=self.ffmpeg_path_override, extra_yt_dlp_args=cookies_args, audio_processing=self.audio_processing)
			elif self.fmt == "opus":
				fp = download_opus(vid, dest_dir, base, yt_dlp_bin=self.yt_dlp_path, ffmpeg_bin=self.ffmpeg_path_override, extra_yt_dlp_args=cookies_args)
			else:
				fp = download_mp3(vid, dest_dir, base, yt_dlp_bin=self.yt_dlp_path, ffmpeg_bin=self.ffmpeg_path_override, extra_yt_dlp_args=cookies_args, audio_processing=self.audio_processing, mp3_quality=self.mp3_quality, cbr_bitrate_kbps=_legacy_cbr_bitrate(self.legacy_options))
			self.sig_status.emit(self.row_idx, "Tagging…")
			cover = _track_cover_bytes(self.track, vid)
			tag_file(fp, self.track, cover if self.embed_art else None, cover_size=_legacy_cover_size(self.legacy_options, embed_art=self.embed_art))
			self.sig_status.emit(self.row_idx, f"Done → {fp.name}")
			from csvmusic.core.audio_quality import audio_quality
			quality = audio_quality(fp)
			if quality["low"]:
				self.sig_status.emit(self.row_idx, f"Downloaded — low quality ({quality['kbps']} kbps)")
			payload = {
				"audio_quality": quality,
				"track": self.track,
				"match": self.match,
				"file_path": str(fp),
				"downloaded": True,
				"error": None,
				"playlist_name": self.playlist_name,
				"cover_bytes": cover
			}
			self.sig_finished.emit(self.row_idx, payload)
		except Exception as e:
			err = str(e)
			log(f"manual download failure: playlist='{self.playlist_name}' track='{self.track.get('artists','')} — {self.track.get('title','')}' fmt={self.fmt} error={err}")
			self.sig_status.emit(self.row_idx, f"Fail: {err[:120]}")
			payload = {
				"track": self.track,
				"match": self.match,
				"file_path": None,
				"downloaded": False,
				"error": err,
				"playlist_name": self.playlist_name
			}
			self.sig_finished.emit(self.row_idx, payload)


class AlternativesFetchWorker(QThread):
	sig_done = Signal(int, list, str)

	def __init__(self, row_idx: int, track: Dict, exclude_ids: set[str] | None = None, parent: QObject | None = None):
		super().__init__(parent)
		self.row_idx = row_idx
		self.track = track
		self.exclude_ids = set(exclude_ids or set())

	def run(self):
		try:
			options = more_candidates(self.track, exclude_ids=self.exclude_ids, source_mode="all")
			self.sig_done.emit(self.row_idx, options, "")
		except Exception as exc:
			log(f"alternatives fetch failure: track='{self.track.get('artists','')} — {self.track.get('title','')}' error={exc}")
			self.sig_done.emit(self.row_idx, [], str(exc))


class YouTubeSessionTestWorker(QThread):
	"""Bounded network check; never returns cookie contents or private account data."""
	sig_done = Signal(bool, str)

	def __init__(self, path: str, parent=None):
		super().__init__(parent)
		self.path = path

	def run(self) -> None:
		from csvmusic.core.youtube_session import validate_cookie_file, verify_youtube_session
		try:
			ok, message = validate_cookie_file(self.path)
			if not ok:
				self.sig_done.emit(False, message)
				return
			if verify_youtube_session(self.path):
				self.sig_done.emit(True, "PASS — YouTube recognizes this signed-in session. Save settings to use it. This checks login only, not age eligibility, download formats, or Premium quality.")
			else:
				self.sig_done.emit(False, "NOT VERIFIED — YouTube did not confirm the login. Reconnect your account; the session may have expired or been revoked.")
		except Exception:
			self.sig_done.emit(False, "CHECK FAILED — Could not verify with YouTube. Check your connection and try again; this does not necessarily mean the cookies are invalid.")


class CookiesCheckWorker(QThread):
	# Emits (ok, message)
	sig_done = Signal(bool, str)

	def __init__(self, cookies_browser: str | None, cookies_file: str | None, yt_dlp_path: str | None, parent: QObject | None = None):
		super().__init__(parent)
		self.cookies_browser = cookies_browser
		self.cookies_file = cookies_file
		self.yt_dlp_path = yt_dlp_path

	def run(self):
		try:
			if self.cookies_file:
				from csvmusic.core.youtube_session import validate_cookie_file
				self.sig_done.emit(*validate_cookie_file(self.cookies_file))
				return
			yt = self.yt_dlp_path or _resolve_ytdlp()
			# Firefox profile pre-check: if a concrete profile path is provided, verify cookies DB exists
			ff_signed_in_hint = None
			if self.cookies_browser:
				parts = str(self.cookies_browser).split(":", 1)
				bid = parts[0].strip().lower()
				prof = parts[1].strip() if len(parts) == 2 else None
				if bid == "firefox" and prof:
					try:
						db = pathlib.Path(prof) / "cookies.sqlite"
						if not db.exists():
							self.sig_done.emit(False, "Firefox cookies DB not found for selected profile.")
							return
						conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
						cur = conn.cursor()
						cur.execute(
							"SELECT name FROM moz_cookies WHERE (host LIKE '%youtube.com' OR host LIKE '%google.com') AND name IN (?,?,?,?,?,?,?) LIMIT 1",
							("__Secure-3PSID","__Secure-1PSID","SAPISID","APISID","SID","SSID","HSID")
						)
						ff_signed_in_hint = cur.fetchone() is not None
						conn.close()
					except Exception:
						# Ignore DB probing errors; continue with yt-dlp probing
						pass
			cmd = [yt]
			if self.cookies_file:
				cmd += ["--cookies", self.cookies_file]
			elif self.cookies_browser:
				cmd += ["--cookies-from-browser", self.cookies_browser]
			# Use YouTube homepage extraction to trigger cookie loading without requiring media formats.
			cmd += ["--skip-download", "--flat-playlist", "--playlist-items", "0", "https://www.youtube.com/"]
			proc = _run_yt_dlp_command(cmd, timeout=12)
			if proc.returncode == 0:
				# Even on success, detect cookie DB issues from logs
				stderr = (proc.stderr or ""); stdout = (proc.stdout or "")
				low_all = (stderr + " \n" + stdout).lower()
				if ("cookie" in low_all) and ("could not" in low_all or "not find" in low_all or "no such file" in low_all):
					self.sig_done.emit(False, "Could not find cookies in database. Check profile selection.")
					return
				# Determine signed-in state
				signed_in = False
				account_hint = None
				# No account name probing; keep it lightweight
				if self.cookies_file:
					try:
						with open(self.cookies_file, "r", encoding="utf-8", errors="ignore") as f:
							for line in f:
								line = line.strip()
								if not line or line.startswith("#"):
									continue
								parts = line.split("\t")
								if len(parts) < 7:
									continue
								domain = parts[0].lower()
								name = parts[5]
								if ("youtube.com" in domain or "google.com" in domain) and name in {"__Secure-3PSID","__Secure-1PSID","SAPISID","APISID","SID","SSID","HSID"}:
									signed_in = True
									break
					except Exception:
						pass

				else:
					# For Firefox, prefer the DB hint result; otherwise do a lightweight probe
					if ff_signed_in_hint is not None:
						signed_in = bool(ff_signed_in_hint)
					else:
						signed_in = "found youtube account cookies" in low_all
				msg = "Signed-in cookies detected" if signed_in else "Guest session (no account cookies)"
				self.sig_done.emit(signed_in, msg)
				return
			stderr = (proc.stderr or "")
			low = stderr.lower()
			if self.cookies_browser and ff_signed_in_hint is True and "signature solving failed" in low:
				self.sig_done.emit(True, "Signed-in cookies detected. YouTube signature warnings may still affect some videos.")
				return
			if ("could not copy" in low and "cookie" in low) or ("locked" in low and "cookie" in low):
				self.sig_done.emit(False, "Cookie DB locked. Close browser and retry.")
				return
			if "dpapi" in low or "cryptprotectdata" in low:
				self.sig_done.emit(False, "DPAPI decryption error. Use same Windows user.")
				return
			self.sig_done.emit(False, stderr.strip()[:160] or "Cookie test failed.")
		except subprocess.TimeoutExpired:
			self.sig_done.emit(False, "Cookie test timeout.")
		except Exception as e:
			self.sig_done.emit(False, str(e)[:160])
