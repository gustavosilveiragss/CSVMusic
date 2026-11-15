from csvmusic.core.ytmusic_match import SearchHealth


def test_aborts_after_three_consecutive_errors():
	health = SearchHealth()
	assert health.record("boom", False) is None
	assert health.record("boom", False) is None
	assert health.record("boom", False) == "YouTube Music search failed repeatedly: boom"


def test_success_resets_error_streak():
	health = SearchHealth()
	health.record("boom", False)
	health.record("boom", False)
	assert health.record(None, True) is None
	assert health.record("boom", False) is None


def test_aborts_after_five_consecutive_empty_searches():
	health = SearchHealth()
	for _ in range(4):
		assert health.record(None, False) is None
	reason = health.record(None, False)
	assert reason is not None and "five tracks in a row" in reason


def test_error_resets_empty_streak():
	health = SearchHealth()
	for _ in range(4):
		health.record(None, False)
	assert health.record("boom", False) is None
	assert health.record(None, False) is None
