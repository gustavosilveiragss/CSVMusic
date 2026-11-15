"""System resource pressure for the download pool. Qt-free; never raises."""

from __future__ import annotations

import enum
import os
from pathlib import Path
from typing import Mapping

HIGH_MEM_FLOOR = 1024 * 1024 * 1024
HIGH_MEM_FRACTION = 0.12
HIGH_MEM_CAP = 2 * 1024 * 1024 * 1024
CRITICAL_MEM_FLOOR = 512 * 1024 * 1024
CRITICAL_MEM_FRACTION = 0.06
CRITICAL_MEM_CAP = 1024 * 1024 * 1024
HIGH_PSI_MEM_SOME = 20.0
HIGH_PSI_CPU_SOME = 80.0
CRITICAL_PSI_MEM_FULL = 5.0
# Without PSI, run-queue length per core is the only CPU signal.
HIGH_LOAD_PER_CORE = 2.5


class Pressure(enum.IntEnum):
	OK = 0
	HIGH = 1
	CRITICAL = 2


def parse_meminfo(text: str) -> tuple[int, int] | None:
	"""Return (MemAvailable, MemTotal) in bytes."""
	values: dict[str, int] = {}
	for line in text.splitlines():
		key, _, rest = line.partition(":")
		if key in ("MemAvailable", "MemTotal"):
			try:
				values[key] = int(rest.split()[0]) * 1024
			except (IndexError, ValueError):
				return None
	if "MemAvailable" not in values or "MemTotal" not in values:
		return None
	return values["MemAvailable"], values["MemTotal"]


def parse_psi(text: str) -> dict[str, float]:
	"""Return {"some": avg10, "full": avg10} from a /proc/pressure file."""
	out: dict[str, float] = {}
	for line in text.splitlines():
		kind, _, rest = line.partition(" ")
		for field in rest.split():
			name, _, value = field.partition("=")
			if name == "avg10":
				out[kind] = float(value)
	return out


def classify(
	mem: tuple[int, int] | None,
	psi_mem: Mapping[str, float] | None,
	psi_cpu: Mapping[str, float] | None,
	load_per_core: float | None,
) -> Pressure:
	if mem is not None:
		available, total = mem
		if available < max(CRITICAL_MEM_FLOOR, min(total * CRITICAL_MEM_FRACTION, CRITICAL_MEM_CAP)):
			return Pressure.CRITICAL
	if psi_mem and psi_mem.get("full", 0.0) >= CRITICAL_PSI_MEM_FULL:
		return Pressure.CRITICAL
	if mem is not None:
		available, total = mem
		if available < max(HIGH_MEM_FLOOR, min(total * HIGH_MEM_FRACTION, HIGH_MEM_CAP)):
			return Pressure.HIGH
	if psi_mem and psi_mem.get("some", 0.0) >= HIGH_PSI_MEM_SOME:
		return Pressure.HIGH
	if psi_cpu is not None:
		if psi_cpu.get("some", 0.0) >= HIGH_PSI_CPU_SOME:
			return Pressure.HIGH
	elif load_per_core is not None and load_per_core >= HIGH_LOAD_PER_CORE:
		return Pressure.HIGH
	return Pressure.OK


def read_pressure(proc: Path = Path("/proc")) -> Pressure:
	"""Sample the system; anything unreadable counts as OK for that signal.

	Linux only: Windows and macOS always give OK. Inside a container /proc usually shows the host, not the container's limits.
	"""
	try:
		mem = parse_meminfo((proc / "meminfo").read_text())
	except (OSError, ValueError):
		mem = None
	psi: dict[str, dict[str, float] | None] = {}
	for name in ("memory", "cpu"):
		try:
			psi[name] = parse_psi((proc / "pressure" / name).read_text())
		except (OSError, ValueError):
			psi[name] = None
	load: float | None = None
	if psi["cpu"] is None:
		try:
			load = os.getloadavg()[0] / (os.cpu_count() or 1)
		except (OSError, AttributeError):
			load = None
	return classify(mem, psi["memory"], psi["cpu"], load)
