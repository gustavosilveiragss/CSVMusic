from csvmusic.ui import workers
from tests.ui.test_workers_parallel import _by_track, _cand, _patch_common, _track, _worker


def test_job_build_error_fails_track_and_continues(monkeypatch, tmp_path):
	worker, events = _worker(tmp_path, [_track("bad"), _track("good")], force_download=False)
	matches = {t: (_cand(f"v-{t}", 0.95), 0.95, [_cand(f"v-{t}", 0.95)]) for t in ("bad", "good")}
	calls: list = []
	_patch_common(monkeypatch, worker, matches, {"v-bad": None, "v-good": None}, calls)
	real_base = workers.track_filename_base

	def flaky_base(track):
		if track["title"] == "bad":
			raise ValueError("name boom")
		return real_base(track)

	monkeypatch.setattr(workers, "track_filename_base", flaky_base)
	worker.run()
	groups = _by_track(events)
	assert ("sig_row_status", "Fail: name boom") in groups[0]
	assert groups[0][-1][1]["error"] == "name boom"
	assert groups[1][-1][1]["downloaded"] is True
	assert [c[0] for c in calls] == ["v-good"]
	assert [a for n, a in events if n == "sig_progress"] == [[1, 2], [2, 2]]
	done = [a for n, a in events if n == "sig_done"][0]
	assert len(done[3]) == 1 and done[3][0]["error"] == "name boom"
