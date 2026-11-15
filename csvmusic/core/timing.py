# tabs only
import time
from collections.abc import Iterator
from contextlib import contextmanager

from csvmusic.core.log import log


@contextmanager
def timed(stage: str, **fields: object) -> Iterator[dict[str, object]]:
	"""Log how long a stage took as one `timing:` line; the caller may add fields (e.g. rc) to the yielded dict."""
	extra: dict[str, object] = {}
	started = time.monotonic()
	try:
		yield extra
	finally:
		details = " ".join(f"{key}={value}" for key, value in {**fields, **extra}.items())
		log(f"timing: stage={stage} seconds={time.monotonic() - started:.1f} {details}".rstrip())
