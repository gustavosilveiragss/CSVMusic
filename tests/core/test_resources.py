from pathlib import Path

import pytest

from csvmusic.core.resources import Pressure, classify, parse_meminfo, parse_psi, read_pressure

GIB = 1024 ** 3


def proc(tmp_path: Path, avail_kib: int | None, total_kib: int = 10 * 1024 * 1024, mem_psi: str | None = None, cpu_psi: str | None = None) -> Path:
	tmp_path.mkdir(exist_ok=True)
	if avail_kib is not None:
		(tmp_path / "meminfo").write_text(f"MemTotal: {total_kib} kB\nMemFree: 1 kB\nMemAvailable: {avail_kib} kB\n")
	(tmp_path / "pressure").mkdir()
	for name, body in (("memory", mem_psi), ("cpu", cpu_psi)):
		if body is not None:
			(tmp_path / "pressure" / name).write_text(body)
	return tmp_path


def psi(some: float, full: float | None = None) -> str:
	text = f"some avg10={some} avg60=0.00 avg300=0.00 total=1\n"
	if full is not None:
		text += f"full avg10={full} avg60=0.00 avg300=0.00 total=1\n"
	return text


def test_parsers() -> None:
	assert parse_meminfo("MemTotal: 2048 kB\nMemAvailable: 1024 kB\n") == (1024 * 1024, 2048 * 1024)
	assert parse_meminfo("MemTotal: 1 kB\n") is None
	assert parse_psi(psi(12.5, 3.0)) == {"some": 12.5, "full": 3.0}


@pytest.mark.parametrize(
	"avail_gib,expected",
	[(4.0, Pressure.OK), (0.9, Pressure.HIGH), (0.3, Pressure.CRITICAL)],
)
def test_memory_thresholds(tmp_path: Path, avail_gib: float, expected: Pressure) -> None:
	root = proc(tmp_path, int(avail_gib * 1024 * 1024), mem_psi=psi(0, 0), cpu_psi=psi(0))
	assert read_pressure(root) == expected


def test_psi_signals(tmp_path: Path) -> None:
	assert read_pressure(proc(tmp_path / "a", 4 * 1024 * 1024, mem_psi=psi(25, 0), cpu_psi=psi(0))) == Pressure.HIGH
	assert read_pressure(proc(tmp_path / "b", 4 * 1024 * 1024, mem_psi=psi(0, 6), cpu_psi=psi(0))) == Pressure.CRITICAL
	assert read_pressure(proc(tmp_path / "c", 4 * 1024 * 1024, mem_psi=psi(0, 0), cpu_psi=psi(85))) == Pressure.HIGH


def test_load_fallback_only_without_cpu_psi() -> None:
	mem = (4 * GIB, 10 * GIB)
	assert classify(mem, None, None, 3.0) == Pressure.HIGH
	assert classify(mem, None, None, 1.0) == Pressure.OK
	assert classify(mem, None, {"some": 0.0}, 9.0) == Pressure.OK


def test_unreadable_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("os.getloadavg", lambda: (0.0, 0.0, 0.0))
	assert read_pressure(tmp_path / "missing") == Pressure.OK
	(tmp_path / "meminfo").write_text("garbage")
	assert read_pressure(tmp_path) == Pressure.OK


def test_empty_meminfo_value_is_not_an_error() -> None:
	assert parse_meminfo("MemTotal: 1 kB\nMemAvailable:\n") is None


def test_bad_meminfo_does_not_hide_psi(tmp_path: Path) -> None:
	root = proc(tmp_path, None, mem_psi=psi(0, 9), cpu_psi=psi(0))
	(root / "meminfo").write_text("MemTotal: 1 kB\nMemAvailable:\n")
	assert read_pressure(root) == Pressure.CRITICAL


def test_memory_threshold_is_capped_on_big_machines() -> None:
	total = 64 * GIB
	assert classify((3 * GIB, total), None, None, None) == Pressure.OK  # 12% would be 7.7 GiB
	assert classify((1.5 * GIB, total), None, None, None) == Pressure.HIGH
	assert classify((0.8 * GIB, total), None, None, None) == Pressure.CRITICAL
