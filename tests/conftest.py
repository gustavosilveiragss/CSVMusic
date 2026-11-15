# tabs only
import shutil
import subprocess
import sys
import uuid

import pytest

try:
	from PySide6.QtCore import QCoreApplication, QEvent
	from PySide6.QtWidgets import QApplication
except ImportError:
	# Headless installs (e.g. the Docker image) ship without PySide6; only the
	# Qt-free tests can run there.
	QApplication = None
	collect_ignore = ["ui", "test_app_environment.py"]


@pytest.fixture(scope="session", autouse=True)
def qt_application():
	"""Keep one Qt application alive until all test widgets are disposed."""
	if QApplication is None:
		yield None
		return
	app = QApplication.instance() or QApplication([])
	yield app
	for widget in app.topLevelWidgets():
		widget.close()
		widget.deleteLater()
	QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
	app.processEvents()
	app.quit()


class Sleeper:
	"""A 30 s sleeper command tagged with a per-test marker, so leftovers are never confused with other runs."""

	def __init__(self) -> None:
		self.marker = f"csvmusic_sleep_{uuid.uuid4().hex}"
		self.cmd = [sys.executable, "-c", f"import time; time.sleep(30)  # {self.marker}"]

	def leftovers(self) -> str:
		return subprocess.run(["pgrep", "-f", self.marker], capture_output=True, text=True).stdout.strip()

	def kill(self) -> None:
		subprocess.run(["pkill", "-9", "-f", self.marker], capture_output=True)


@pytest.fixture
def sleeper():
	if shutil.which("pgrep") is None:
		pytest.skip("pgrep indisponivel")
	s = Sleeper()
	yield s
	s.kill()
