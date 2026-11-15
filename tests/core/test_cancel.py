# tabs only
import os, sys, threading, time
import pytest
from csvmusic.core.downloader import (
	CancelToken, DownloadCancelled, DownloadError, _run_capture, process_scope,
)

SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_cancel_from_other_thread_kills_quickly() -> None:
	token = CancelToken()
	threading.Timer(0.5, token.cancel).start()
	start = time.monotonic()
	with process_scope(token), pytest.raises(DownloadCancelled):
		_run_capture(SLEEP)
	assert time.monotonic() - start < 2.5


def test_timeout_raises_download_error() -> None:
	start = time.monotonic()
	with pytest.raises(DownloadError, match="timed out"):
		_run_capture(SLEEP, timeout=0.5)
	assert time.monotonic() - start < 3


def test_plain_command_unchanged() -> None:
	proc = _run_capture([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"])
	assert proc.stdout.strip() == "hi"
	assert proc.returncode == 3


def test_already_cancelled_token_leaves_no_process() -> None:
	token = CancelToken()
	token.cancel()
	start = time.monotonic()
	with process_scope(token), pytest.raises(DownloadCancelled):
		_run_capture(SLEEP)
	assert time.monotonic() - start < 2.5
	assert not token._procs


def test_process_unregistered_after_finish() -> None:
	token = CancelToken()
	with process_scope(token):
		_run_capture([sys.executable, "-c", "pass"])
	assert not token._procs
	assert not token.cancelled


def test_cancel_kills_grandchild() -> None:
	import shutil, subprocess
	if shutil.which("pgrep") is None:
		pytest.skip("pgrep indisponivel")
	marker = f"csvmusic_gc_{os.getpid()}_{time.monotonic_ns()}"
	grandchild = f"import time; time.sleep(30)  # {marker}"
	code = (
		"import subprocess,sys,time; "
		f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); time.sleep(30)"
	)
	token = CancelToken()
	threading.Timer(0.5, token.cancel).start()
	start = time.monotonic()
	try:
		with process_scope(token), pytest.raises(DownloadCancelled):
			_run_capture([sys.executable, "-c", code])
		elapsed = time.monotonic() - start
		time.sleep(0.3)
		left = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()
	finally:
		subprocess.run(["pkill", "-9", "-f", marker], capture_output=True)
	assert elapsed < 2
	assert left == []
