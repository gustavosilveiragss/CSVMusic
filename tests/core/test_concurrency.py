import pytest

from csvmusic.core.concurrency import AdaptiveLimit, hard_cap


class FakeClock:
	def __init__(self) -> None:
		self.now = 0.0

	def __call__(self) -> float:
		return self.now


@pytest.mark.parametrize(
	"cpus,expected",
	[(None, 2), (1, 2), (2, 3), (4, 6), (8, 8), (12, 8), (16, 8)],
)
def test_hard_cap_table(cpus: int | None, expected: int) -> None:
	assert hard_cap(cpus) == expected


def make(start: int = 1, ceiling: int = 4, **kw) -> tuple[AdaptiveLimit, FakeClock]:
	clock = FakeClock()
	return AdaptiveLimit(start, ceiling, clock=clock, **kw), clock


def feed(limit: AdaptiveLimit, clock: FakeClock, n: int, dt: float) -> list[str | None]:
	out = []
	for _ in range(n):
		clock.now += dt
		out.append(limit.on_completion())
	return out


def window(limit: AdaptiveLimit, clock: FakeClock, rate: float) -> str | None:
	"""Feeds completions at `rate`/s until the window closes; returns the action."""
	for _ in range(10000):
		clock.now += 1.0 / rate
		res = limit.on_completion()
		if res is not None:
			return res
	raise AssertionError("window never closed")


def warm(limit: AdaptiveLimit, clock: FakeClock) -> None:
	res = feed(limit, clock, limit.level, 0.1)
	assert all(r is None for r in res)


def test_climbs_while_throughput_improves_until_ceiling() -> None:
	limit, clock = make(1, 3)
	assert window(limit, clock, 1.0) == "climb"
	assert limit.level == 2
	warm(limit, clock)
	assert window(limit, clock, 1.5) == "climb"
	assert limit.level == 3
	warm(limit, clock)
	assert window(limit, clock, 2.0) == "hold"
	assert limit.level == 3


def test_reverts_when_not_improving_and_plateaus() -> None:
	limit, clock = make(1, 4)
	window(limit, clock, 1.0)
	warm(limit, clock)
	assert window(limit, clock, 1.05) == "revert"
	assert limit.level == 1
	assert feed(limit, clock, 20, 1.0) == [None] * 20


def test_reprobe_remeasures_base_before_climbing_and_reverts_if_worse() -> None:
	limit, clock = make(1, 4, reprobe_s=300.0)
	window(limit, clock, 1.0)
	warm(limit, clock)
	assert window(limit, clock, 1.0) == "revert"
	clock.now += 299.0
	assert limit.on_completion() is None
	clock.now += 1.0
	assert limit.on_completion() == "reprobe"
	assert limit.level == 1
	assert window(limit, clock, 1.0) == "climb"  # fresh base first, then up
	assert limit.level == 2
	warm(limit, clock)
	assert window(limit, clock, 1.0) == "revert"
	assert limit.level == 1
	clock.now += 10.0
	assert limit.on_completion() is None


def test_reprobe_compares_with_fresh_base_not_the_old_plateau() -> None:
	limit, clock = make(1, 4, reprobe_s=100.0)
	window(limit, clock, 1.0)
	warm(limit, clock)
	assert window(limit, clock, 1.0) == "revert"  # stale base: 1/s
	clock.now += 101.0
	assert limit.on_completion() == "reprobe"
	assert window(limit, clock, 0.2) == "climb"  # throughput collapsed to 0.2/s: that is the new base
	warm(limit, clock)
	assert window(limit, clock, 0.25) == "climb"  # +25% over the fresh base; the stale 1/s would revert
	assert limit.level == 3


def test_warmup_discards_level_completions() -> None:
	limit, clock = make(1, 4)
	window(limit, clock, 1.0)
	assert limit.level == 2
	# Slow warmup completions must not pollute the window.
	feed(limit, clock, 2, 100.0)
	assert window(limit, clock, 1.5) == "climb"


def test_window_needs_count_and_time() -> None:
	limit, clock = make(1, 4)
	assert feed(limit, clock, 10, 1.0)[-1] is None  # 10s < 30s
	limit2, clock2 = make(1, 4)
	assert feed(limit2, clock2, 3, 100.0) == [None] * 3  # < 4 completions
	clock2.now += 1.0
	assert limit2.on_completion() == "climb"


def test_lower_ceiling_monotonic_and_cuts_level() -> None:
	limit, clock = make(3, 6)
	limit.lower_ceiling(5)
	assert limit.ceiling == 5
	limit.lower_ceiling(8)
	assert limit.ceiling == 5
	feed(limit, clock, 3, 1.0)
	limit.lower_ceiling(2)
	assert (limit.level, limit.ceiling) == (2, 2)
	# Measurement restarted: warmup of 2 completions, no early window close.
	assert feed(limit, clock, 2, 100.0) == [None, None]


def test_floor_is_one() -> None:
	limit, _ = make(3, 6)
	limit.lower_ceiling(0)
	assert (limit.level, limit.ceiling) == (1, 1)


def test_ceiling_one_never_climbs() -> None:
	limit, clock = make(1, 1)
	assert feed(limit, clock, 50, 10.0) == [None] * 50
	assert limit.level == 1


def test_start_clamped_to_ceiling() -> None:
	limit, _ = make(9, 3)
	assert limit.level == 3


def test_start_at_ceiling_holds() -> None:
	limit, clock = make(3, 3)
	assert window(limit, clock, 1.0) == "hold"
	assert limit.level == 3


def test_describe() -> None:
	limit, clock = make(1, 4)
	window(limit, clock, 0.5)
	text = limit.describe()
	assert "level=2" in text and "x_per_min=30.0" in text and "prev=30.0" in text
