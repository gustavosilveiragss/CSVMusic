import pathlib

from csvmusic.core.downloader import output_collision_key


def key(base: str, folder: str = "/music/album") -> tuple[str, str]:
	return output_collision_key(pathlib.Path(folder), base)


def test_names_sharing_the_cleanup_prefix_collide():
	assert key("Song") == key("Song.feat X")


def test_case_and_unicode_form_are_folded():
	assert key("Café") == key("CAFÉ")
	assert key("SONG") == key("song")
	assert key("é") == key("é")


def test_dotted_names_serialize_as_a_known_cost():
	assert key("St. Vincent - A") == key("St. Elsewhere")


def test_different_folders_and_names_do_not_collide():
	assert key("Song", "/a") != key("Song", "/b")
	assert key("Song A") != key("Song B")
