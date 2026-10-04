from types import SimpleNamespace

import numpy as np
import pytest

import characters
import config as cfg
from conftest import FakeStash
from db import EmbDB
from stash_client import TagManager

SEED = "s"


def args(**kw):
    base = dict(name="Frieren", threshold=0.8, mode="centroid", min_seeds=2, max=0,
                kind="images", dry_run=False, series=None)
    base.update(kw)
    return SimpleNamespace(**base)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def setup():
    """Seeds 1,2 point along x; 3 is close, 4 is far, 5 is close but already suggested."""
    stash = FakeStash(
        tags=[{"id": SEED, "name": "Frieren", "aliases": []}],
        items={("image", "1"): {SEED}, ("image", "2"): {SEED}, ("image", "3"): set(),
               ("image", "4"): set(), ("image", "5"): set()})
    db = EmbDB(":memory:")
    db.put("image", "1", cfg.MODEL_REPO, unit([1, 0.1]))
    db.put("image", "2", cfg.MODEL_REPO, unit([1, -0.1]))
    db.put("image", "3", cfg.MODEL_REPO, unit([1, 0.3]))
    db.put("image", "4", cfg.MODEL_REPO, unit([0, 1]))
    db.put("image", "5", cfg.MODEL_REPO, unit([1, 0.2]))
    db.mark_suggested("Frieren", "image", [("5", 0.99)])
    return stash, TagManager(stash), db


def review_tag(stash):
    return next(t["id"] for t in stash.tags if t["name"] == "Character?/Frieren")


def test_suggest_tags_new_matches(setup):
    stash, tm, db = setup
    characters.cmd_suggest(args(), stash, tm, db)
    review = review_tag(stash)
    assert stash.updates == [("image", ["3"], [review], "ADD")]
    assert db.suggested_ids("Frieren", "image") == {"3", "5"}
    parent = next(t for t in stash.created if t["name"] == "WD14: Character Review")
    assert next(t for t in stash.created if t["id"] == review)["parent_ids"] == [parent["id"]]


def test_suggest_max_mode(setup):
    stash, tm, db = setup
    characters.cmd_suggest(args(mode="max", threshold=0.5), stash, tm, db)
    assert stash.updates[0][1] == ["3"]


def test_suggest_threshold_and_cap(setup):
    stash, tm, db = setup
    db.con.execute("DELETE FROM suggested")
    characters.cmd_suggest(args(threshold=0.0, max=2), stash, tm, db)
    assert stash.updates[0][1] == ["5", "3"]  # best first, item 4 cut by cap


def test_suggest_dry_run_writes_nothing(setup, capsys):
    stash, _, db = setup
    tm = TagManager(stash, dry_run=True)
    characters.cmd_suggest(args(dry_run=True), stash, tm, db)
    assert stash.updates == [] and stash.created == []
    assert db.suggested_ids("Frieren", "image") == {"5"}
    assert "1 matches" in capsys.readouterr().out


def test_suggest_not_enough_seeds(setup, capsys):
    stash, tm, db = setup
    characters.cmd_suggest(args(min_seeds=3), stash, tm, db)
    assert stash.updates == []
    assert "only 2 seeds" in capsys.readouterr().out


def test_suggest_no_embeddings(capsys):
    stash = FakeStash(tags=[{"id": SEED, "name": "Frieren", "aliases": []}])
    characters.cmd_suggest(args(kind="scenes"), stash, TagManager(stash), EmbDB(":memory:"))
    assert "[scene] no embeddings yet" in capsys.readouterr().out


def test_suggest_missing_seed_tag_exits():
    stash = FakeStash()
    with pytest.raises(SystemExit):
        characters.cmd_suggest(args(), stash, TagManager(stash), EmbDB(":memory:"))


def test_promote_moves_review_to_real_tag(setup):
    stash, tm, db = setup
    characters.cmd_suggest(args(), stash, tm, db)
    review = review_tag(stash)
    stash.updates.clear()

    characters.cmd_promote(args(series="Sousou no Frieren"), stash, tm, db)

    # the seed tag "Frieren" is reused as the real tag
    assert stash.updates == [("image", ["3"], [SEED], "ADD"),
                             ("image", ["3"], [review], "REMOVE")]
    assert stash.items[("image", "3")] == {SEED}
    assert any(t["name"] == "[Series] Sousou no Frieren" for t in stash.created)


def test_promote_creates_real_tag_under_series():
    stash = FakeStash(tags=[{"id": "r", "name": "Character?/Fern", "aliases": []}],
                      items={("image", "1"): {"r"}})
    tm = TagManager(stash)
    characters.cmd_promote(args(name="Fern", series="Frieren"), stash, tm, None)
    by_name = {t["name"]: t for t in stash.created}
    assert by_name["Fern"]["parent_ids"] == [by_name["[Series] Frieren"]["id"]]
    assert stash.items[("image", "1")] == {by_name["Fern"]["id"]}


def test_promote_dry_run():
    stash = FakeStash(tags=[{"id": "r", "name": "Character?/Fern", "aliases": []}],
                      items={("image", "1"): {"r"}})
    characters.cmd_promote(args(name="Fern", dry_run=True), stash,
                           TagManager(stash, dry_run=True), None)
    assert stash.updates == [] and stash.created == []


def test_promote_without_review_tag_exits():
    stash = FakeStash()
    with pytest.raises(SystemExit):
        characters.cmd_promote(args(), stash, TagManager(stash), None)
