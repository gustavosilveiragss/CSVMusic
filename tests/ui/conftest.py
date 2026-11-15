# tabs only
import pytest

from csvmusic.core import download_pool
from csvmusic.ui import workers


@pytest.fixture(autouse=True)
def no_pacing_waits(monkeypatch):
	"""The coordinator waits for pacing deadlines in the pool; tests that care patch workers.track_pause_s themselves."""
	monkeypatch.setattr(workers, "track_pause_s", lambda _profile: 0.0)
	monkeypatch.setattr(download_pool, "MATCH_GAP_S", 0)
