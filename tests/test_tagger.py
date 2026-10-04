import io

import numpy as np
import pytest
from PIL import Image

import config as cfg
import tagger
from conftest import FakeStash
from db import EmbDB
from stash_client import TagManager


@pytest.mark.parametrize("raw, expected", [
    ("saber (fate)", "Saber (Fate)"),
    ("hatsune miku", "Hatsune Miku"),
    ("black/white", "Black/White"),
    ("x-ray", "X-Ray"),
    ("1girl", "1girl"),
])
def test_pretty(raw, expected):
    assert tagger.pretty(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("saber_(fate)", ("Saber (Fate)", "Fate")),
    ("hatsune_miku", ("Hatsune Miku", None)),
    ("artoria_pendragon_(lancer)_(fate)", ("Artoria Pendragon (Lancer) (Fate)", "Fate")),
    ("tokai_teio_(umamusume)", ("Tokai Teio (Umamusume)", "Umamusume")),
    ("ganyu_(genshin_impact)", ("Ganyu (Genshin Impact)", "Genshin Impact")),
])
def test_char_parts(raw, expected):
    assert tagger.char_parts(raw) == expected


@pytest.mark.parametrize("raw, group", [
    ("1girl", "Count"),
    ("solo", "Count"),
    ("smile", "Expression"),
    ("open_mouth", "Expression"),
    ("long_hair", "Hair"),
    ("blue_eyes", "Eyes"),
    ("large_breasts", "Body"),
    ("school_uniform", "Clothing"),
    ("looking_at_viewer", "Pose/Action"),
    ("white_background", "Background/Setting"),
    ("katana", "Other"),
])
def test_general_group(raw, group):
    assert tagger.general_group(raw) == group


def test_map_path(monkeypatch):
    monkeypatch.setattr(cfg, "PATH_MAP", {"/data/": "D:/media/"})
    assert tagger.map_path("/data/a/b.mp4") == "D:/media/a/b.mp4"
    assert tagger.map_path("/other/b.mp4") == "/other/b.mp4"


# --- Hierarchy --------------------------------------------------------------------

def names_by_id(stash):
    return {t["id"]: t for t in stash.created}


def test_hierarchy_builds_tree_and_tag_ids():
    stash = FakeStash()
    H = tagger.Hierarchy(TagManager(stash))
    labels = {"rating": "general", "characters": ["saber_(fate)", "hatsune_miku"],
              "general": ["long_hair", "^_^", "katana"]}
    ids = H.tag_ids(labels)
    tags = names_by_id(stash)
    name = {tid: t["name"] for tid, t in tags.items()}
    by_name = {t["name"]: t for t in tags.values()}

    assert [name[i] for i in ids] == [
        cfg.MARKER_TAG, "rating:general", "Saber (Fate)", "Hatsune Miku",
        "long hair", "^_^", "katana"]
    assert by_name[cfg.MARKER_TAG]["parent_ids"] == [H.root]
    assert by_name["[Series] Fate"]["parent_ids"] == [H.characters]
    assert by_name["Saber (Fate)"]["parent_ids"] == [by_name["[Series] Fate"]["id"]]
    assert by_name["Saber (Fate)"]["aliases"] == ["saber_(fate)"]
    assert by_name["Hatsune Miku"]["parent_ids"] == [H.characters]
    assert by_name["long hair"]["parent_ids"] == [by_name["WD14: Hair"]["id"]]
    assert by_name["long hair"]["aliases"] == ["long_hair"]
    assert by_name["WD14: Hair"]["parent_ids"] == [H.general]
    assert by_name["katana"]["parent_ids"] == [by_name["WD14: Other"]["id"]]


def test_hierarchy_reuses_tags_and_dedupes_ids():
    stash = FakeStash([{"id": "1", "name": "Long Hair", "aliases": []}])
    H = tagger.Hierarchy(TagManager(stash))
    n_created = len(stash.created)
    ids = H.tag_ids({"rating": "general", "characters": [],
                     "general": ["long_hair", "long hair"]})
    assert ids.count("1") == 1
    assert "long hair" not in [t["name"].lower() for t in stash.created[n_created:]]


def test_untagged_ids_excludes_marker_unless_dry():
    stash = FakeStash(items={("image", "1"): {"m"}, ("image", "2"): set()})
    assert tagger.untagged_ids(stash, "image", "m") == ["2"]
    assert sorted(tagger.untagged_ids(stash, "image", "dry-1")) == ["1", "2"]


# --- processing -------------------------------------------------------------------

class FakeModel:
    def __init__(self, labels, dim=3):
        self.labels, self.dim, self.predicted = labels, dim, []

    def predict(self, images):
        self.predicted.append(len(images))
        n = len(images)
        return np.zeros((n, 2), np.float32), np.ones((n, self.dim), np.float32)

    def labels_for_image(self, p, general_thr, char_thr):
        return self.labels

    def labels_for_frames(self, P, *args):
        self.frames_args = (len(P), args)
        return self.labels


LABELS = {"rating": "general", "characters": [], "general": ["smile"]}


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "PNG")
    return buf.getvalue()


class ImageStash(FakeStash):
    def __init__(self, ids, broken=()):
        super().__init__(items={("image", i): set() for i in ids})
        self.broken = set(broken)

    def image_info(self, iid):
        return {"id": iid, "paths": {"image": f"http://x/{iid}"}}

    def get_bytes(self, url):
        if url.rsplit("/", 1)[1] in self.broken:
            return b"not an image"
        return png_bytes()


def test_process_images_tags_and_stores_embeddings(monkeypatch):
    monkeypatch.setattr(cfg, "BATCH_SIZE", 2)
    stash = ImageStash(["1", "2", "3"], broken=["2"])
    H = tagger.Hierarchy(TagManager(stash))
    db = EmbDB(":memory:")
    model = FakeModel(LABELS)

    tagger.process_images(stash, H, model, db, limit=0, dry=False)

    assert model.predicted == [1, 1]  # batch [1,2] lost the broken image, batch [3]
    ids, M = db.load("image", cfg.MODEL_REPO)
    assert sorted(ids) == ["1", "3"]
    assert stash.items[("image", "1")] >= {H.marker}
    assert stash.items[("image", "2")] == set()
    # second run finds nothing new except the broken image
    assert tagger.untagged_ids(stash, "image", H.marker) == ["2"]


def test_process_images_respects_limit(monkeypatch):
    stash = ImageStash(["1", "2", "3"])
    H = tagger.Hierarchy(TagManager(stash))
    tagger.process_images(stash, H, FakeModel(LABELS), EmbDB(":memory:"), limit=2, dry=False)
    assert len(stash.updates) == 2


def test_process_images_dry_run_writes_nothing(capsys):
    stash = ImageStash(["1"])
    H = tagger.Hierarchy(TagManager(stash, dry_run=True))
    db = EmbDB(":memory:")
    tagger.process_images(stash, H, FakeModel(LABELS), db, limit=0, dry=True)
    assert stash.created == [] and stash.updates == []
    assert db.load("image", cfg.MODEL_REPO)[0] == []
    assert "image 1: rating=general" in capsys.readouterr().out


class SceneStash(FakeStash):
    def __init__(self, files_by_id):
        super().__init__(items={("scene", i): set() for i in files_by_id})
        self.files_by_id = files_by_id

    def scene_info(self, sid):
        return {"id": sid, "files": self.files_by_id[sid]}


def test_process_scenes(monkeypatch):
    monkeypatch.setattr(cfg, "BATCH_SIZE", 2)
    monkeypatch.setattr(cfg, "PATH_MAP", {"/data/": "D:/"})
    seen = []

    def fake_sample(path, duration):
        seen.append((path, duration))
        return [] if path.endswith("empty.mp4") else [Image.new("RGB", (2, 2))] * 5

    monkeypatch.setattr(tagger, "sample_frames", fake_sample)
    stash = SceneStash({
        "1": [{"path": "/data/a.mp4", "duration": 100}],
        "2": [],
        "3": [{"path": "/data/empty.mp4", "duration": 5}],
    })
    H = tagger.Hierarchy(TagManager(stash))
    db = EmbDB(":memory:")
    model = FakeModel(LABELS)

    tagger.process_scenes(stash, H, model, db, limit=0, dry=False)

    assert ("D:/a.mp4", 100) in seen
    assert model.predicted == [2, 2, 1]
    assert model.frames_args[0] == 5
    ids, M = db.load("scene", cfg.MODEL_REPO)
    assert ids == ["1"]
    assert np.isclose(np.linalg.norm(M[0]), 1.0, atol=1e-5)
    assert H.marker in stash.items[("scene", "1")]
    assert stash.items[("scene", "2")] == set() and stash.items[("scene", "3")] == set()


def test_sample_frames_reads_video(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    if not w.isOpened():
        pytest.skip("no MJPG writer")
    for _ in range(300):  # 30 s at 10 fps
        w.write(np.full((32, 32, 3), 128, np.uint8))
    w.release()

    monkeypatch.setattr(cfg, "SCENE_FRAME_INTERVAL", 10)
    monkeypatch.setattr(cfg, "SCENE_MAX_FRAMES", 32)
    frames = tagger.sample_frames(path, None)  # duration from frame count / fps
    assert len(frames) == 3
    assert frames[0].size == (32, 32) and frames[0].mode == "RGB"

    monkeypatch.setattr(cfg, "SCENE_MAX_FRAMES", 2)
    assert len(tagger.sample_frames(path, 30)) == 2


def test_sample_frames_missing_file(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(IOError):
        tagger.sample_frames(str(tmp_path / "nope.mp4"), 10)
