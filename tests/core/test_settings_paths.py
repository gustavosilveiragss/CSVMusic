# tabs only
import pathlib

from csvmusic.core import settings


def test_macos_uses_application_support(monkeypatch, tmp_path: pathlib.Path) -> None:
	monkeypatch.setattr(settings.sys, "platform", "darwin")
	monkeypatch.setattr(settings.pathlib.Path, "home", classmethod(lambda cls: tmp_path))

	assert settings._settings_dir() == tmp_path / "Library" / "Application Support" / "CSVMusic"


def test_linux_honors_xdg_data_home(monkeypatch, tmp_path: pathlib.Path) -> None:
	monkeypatch.setattr(settings.sys, "platform", "linux")
	monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
	monkeypatch.setattr(settings.pathlib.Path, "home", classmethod(lambda cls: tmp_path))

	assert settings._settings_dir() == tmp_path / "xdg-data" / "csvmusic"


def test_macos_migrates_previous_cross_platform_location(monkeypatch, tmp_path: pathlib.Path) -> None:
	legacy = tmp_path / ".local" / "share" / "csvmusic"
	legacy.mkdir(parents=True)
	(legacy / "library.json").write_text("{}", encoding="utf-8")
	monkeypatch.setattr(settings.sys, "platform", "darwin")
	monkeypatch.setattr(settings.pathlib.Path, "home", classmethod(lambda cls: tmp_path))

	target = settings._settings_dir()

	assert target == tmp_path / "Library" / "Application Support" / "CSVMusic"
	assert (target / "library.json").exists()


def test_download_workers_clamps_and_defaults() -> None:
	assert settings.download_workers({}) == 3
	assert settings.download_workers({"download_workers": 0}) == 1
	assert settings.download_workers({"download_workers": -1}) == 1
	assert settings.download_workers({"download_workers": 7}) == 4
	assert settings.download_workers({"download_workers": 2}) == 2
	assert settings.download_workers({"download_workers": "2"}) == 2
	assert settings.download_workers({"download_workers": "abc"}) == 3
	assert settings.download_workers({"download_workers": None}) == 3
	assert settings.download_workers({"download_workers": True}) == 3


def test_pipeline_worker_stores_max_workers() -> None:
	from csvmusic.ui.workers import PipelineWorker

	default = PipelineWorker("", "out", None, "mp3", False, False, False, None, None, None, None)
	assert default.max_workers == 1
	worker = PipelineWorker("", "out", None, "mp3", False, False, False, None, None, None, None, max_workers=4)
	assert worker.max_workers == 4
