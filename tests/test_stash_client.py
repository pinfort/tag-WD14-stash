import pytest

from conftest import FakeStash
from stash_client import Stash, TagManager, chunks, tag_filter


class FakeResponse:
    def __init__(self, payload=None, content=b"", status=200):
        self.payload, self.content, self.status = payload, content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        return self.responses.pop(0)

    def get(self, url, timeout=None):
        return self.responses.pop(0)


def make_stash(*responses, api_key=""):
    s = Stash("http://stash/graphql", api_key)
    s.session = FakeSession(responses)
    if api_key:
        s.session.headers["ApiKey"] = api_key
    return s


# --- helpers -------------------------------------------------------------------

def test_chunks():
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunks([], 3)) == []


def test_tag_filter():
    assert tag_filter("7") == {"tags": {"value": ["7"], "modifier": "INCLUDES", "depth": 0}}
    assert tag_filter("7", "EXCLUDES")["tags"]["modifier"] == "EXCLUDES"


# --- Stash ----------------------------------------------------------------------

def test_api_key_header_only_when_set():
    assert "ApiKey" in Stash("u", "secret").session.headers
    assert "ApiKey" not in Stash("u", "").session.headers


def test_gql_returns_data_and_sends_variables():
    s = make_stash(FakeResponse({"data": {"x": 1}}))
    assert s.gql("query{x}", {"a": 1}) == {"x": 1}
    assert s.session.posts == [("http://stash/graphql",
                                {"query": "query{x}", "variables": {"a": 1}})]


def test_gql_raises_on_graphql_errors():
    s = make_stash(FakeResponse({"errors": [{"message": "boom"}], "data": None}))
    with pytest.raises(RuntimeError, match="boom"):
        s.gql("query{x}")


def test_gql_raises_on_http_error():
    s = make_stash(FakeResponse({}, status=500))
    with pytest.raises(RuntimeError, match="500"):
        s.gql("query{x}")


def test_find_ids_unwraps_result():
    s = make_stash(FakeResponse({"data": {"findScenes": {"scenes": [{"id": "1"}, {"id": "2"}]}}}))
    assert s.find_ids("scene", tag_filter("9")) == ["1", "2"]
    variables = s.session.posts[0][1]["variables"]
    assert variables == {"f": {"per_page": -1}, "x": tag_filter("9")}


def test_find_ids_without_filter_sends_empty_filter():
    s = make_stash(FakeResponse({"data": {"findImages": {"images": []}}}))
    s.find_ids("image")
    assert s.session.posts[0][1]["variables"]["x"] == {}


def test_update_tags_payload():
    s = make_stash(FakeResponse({"data": {"bulkImageUpdate": []}}))
    s.update_tags("image", ["1", "2"], ["5"], "REMOVE")
    inp = s.session.posts[0][1]["variables"]["i"]
    assert inp == {"ids": ["1", "2"], "tag_ids": {"ids": ["5"], "mode": "REMOVE"}}


def test_create_tag_sets_ignore_auto_tag_and_optional_fields():
    s = make_stash(FakeResponse({"data": {"tagCreate": {"id": "42"}}}),
                   FakeResponse({"data": {"tagCreate": {"id": "43"}}}))
    assert s.create_tag("A") == "42"
    assert s.session.posts[0][1]["variables"]["i"] == {"name": "A", "ignore_auto_tag": True}
    s.create_tag("B", aliases=["b"], parent_ids=["1"], description="d")
    assert s.session.posts[1][1]["variables"]["i"] == {
        "name": "B", "ignore_auto_tag": True, "aliases": ["b"], "parent_ids": ["1"],
        "description": "d"}


def test_get_bytes():
    s = make_stash(FakeResponse(content=b"abc"))
    assert s.get_bytes("http://x") == b"abc"


# --- TagManager -----------------------------------------------------------------

def test_find_by_name_or_alias_case_insensitive():
    tm = TagManager(FakeStash([{"id": "1", "name": "Long Hair", "aliases": ["long_hair"]}]))
    assert tm.find("long hair") == "1"
    assert tm.find("LONG_HAIR") == "1"
    assert tm.find("short hair") is None


def test_name_takes_precedence_over_other_tags_alias():
    tm = TagManager(FakeStash([
        {"id": "1", "name": "Other", "aliases": ["Saber"]},
        {"id": "2", "name": "Saber", "aliases": None},
    ]))
    assert tm.find("saber") == "2"


def test_get_reuses_existing_tag_by_alias():
    stash = FakeStash([{"id": "1", "name": "Saber", "aliases": []}])
    tm = TagManager(stash)
    assert tm.get("Saber (Fate)", aliases=["saber"]) == "1"
    assert stash.created == []


def test_get_creates_once_and_caches():
    stash = FakeStash()
    tm = TagManager(stash)
    tid = tm.get("Long Hair", aliases=["long_hair", "Long Hair"], parents=["9"])
    assert tm.get("long hair") == tid
    assert tm.get("long_hair") == tid
    assert len(stash.created) == 1
    # alias equal to the name is dropped
    assert stash.created[0]["aliases"] == ["long_hair"]
    assert stash.created[0]["parent_ids"] == ["9"]


def test_get_retries_without_aliases_on_clash():
    stash = FakeStash()
    stash.reject_aliases = True
    tm = TagManager(stash)
    tm.get("X", aliases=["x_alias"])
    assert stash.created[0]["aliases"] == []


def test_get_reraises_when_no_aliases():
    class Broken(FakeStash):
        def create_tag(self, *a, **k):
            raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        TagManager(Broken()).get("X")


def test_dry_run_creates_nothing_and_drops_fake_parents(capsys):
    stash = FakeStash()
    tm = TagManager(stash, dry_run=True)
    a = tm.get("A")
    b = tm.get("B", parents=[a])
    assert a.startswith("dry-") and b.startswith("dry-") and a != b
    assert tm.get("a") == a
    assert stash.created == []
    assert "would create tag 'A'" in capsys.readouterr().out
