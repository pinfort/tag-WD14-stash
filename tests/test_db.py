import numpy as np

from db import EmbDB


def test_load_empty():
    db = EmbDB(":memory:")
    ids, M = db.load("image", "m")
    assert ids == [] and M.shape == (0, 0)


def test_put_load_roundtrip_filters_by_kind_and_model():
    db = EmbDB(":memory:")
    db.put("image", 1, "m", [1.0, 0.0])
    db.put("image", "2", "m", np.array([0.0, 1.0], np.float64))
    db.put("scene", "3", "m", [1.0, 1.0])
    db.put("image", "4", "other", [1.0, 1.0])
    ids, M = db.load("image", "m")
    assert sorted(ids) == ["1", "2"]
    assert M.dtype == np.float32 and M.shape == (2, 2)
    assert dict(zip(ids, M.tolist())) == {"1": [1.0, 0.0], "2": [0.0, 1.0]}


def test_put_replaces_existing():
    db = EmbDB(":memory:")
    db.put("image", "1", "m", [1.0])
    db.put("image", "1", "m", [2.0])
    ids, M = db.load("image", "m")
    assert ids == ["1"] and M.tolist() == [[2.0]]


def test_suggested_tracking():
    db = EmbDB(":memory:")
    assert db.suggested_ids("Frieren", "image") == set()
    db.mark_suggested("Frieren", "image", [(1, 0.9), ("2", 0.85)])
    db.mark_suggested("Fern", "image", [("3", 0.9)])
    assert db.suggested_ids("Frieren", "image") == {"1", "2"}
    assert db.suggested_ids("Frieren", "scene") == set()


def test_commit_persists(tmp_path):
    path = str(tmp_path / "e.sqlite")
    db = EmbDB(path)
    db.put("image", "1", "m", [0.5])
    db.commit()
    db.con.close()
    ids, _ = EmbDB(path).load("image", "m")
    assert ids == ["1"]
