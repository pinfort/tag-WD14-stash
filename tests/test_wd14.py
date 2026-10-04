import numpy as np
import pytest
from PIL import Image

from wd14 import WD14, _pad_square, _to_rgb, display_name

# selected_tags.csv layout: ratings (cat 9), general (cat 0), characters (cat 4)
NAMES = ["general", "sensitive", "questionable", "explicit",
         "1girl", "long_hair", "smile", "saber_(fate)", "hatsune_miku"]


@pytest.fixture
def model():
    m = WD14.__new__(WD14)  # skip model download
    m.names = NAMES
    m.rating_idx = np.array([0, 1, 2, 3])
    m.general_idx = np.array([4, 5, 6])
    m.char_idx = np.array([7, 8])
    return m


def probs(rating, general=(0, 0, 0), chars=(0, 0)):
    return np.array([*rating, *general, *chars], np.float32)


def test_display_name():
    assert display_name("long_hair") == "long hair"
    assert display_name("^_^") == "^_^"
    assert display_name("o_o") == "o_o"


def test_to_rgb_composites_alpha_on_white():
    im = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    out = _to_rgb(im)
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)


def test_to_rgb_keeps_opaque_pixels():
    im = Image.new("LA", (1, 1), (10, 255))
    assert _to_rgb(im).getpixel((0, 0)) == (10, 10, 10)
    assert _to_rgb(Image.new("L", (1, 1), 7)).getpixel((0, 0)) == (7, 7, 7)


def test_pad_square_centers_on_white():
    im = Image.new("RGB", (4, 2), (0, 0, 0))
    out = _pad_square(im)
    assert out.size == (4, 4)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((0, 1)) == (0, 0, 0)
    assert out.getpixel((3, 3)) == (255, 255, 255)


def test_labels_for_image_thresholds_and_sort_order(model):
    p = probs((0.1, 0.8, 0.05, 0.05), general=(0.5, 0.9, 0.2), chars=(0.86, 0.7))
    labels = model.labels_for_image(p, general_thr=0.35, char_thr=0.85)
    assert labels == {"rating": "sensitive", "characters": ["saber_(fate)"],
                      "general": ["long_hair", "1girl"]}


def test_threshold_is_inclusive(model):
    p = probs((1, 0, 0, 0), general=(0.35, 0, 0))
    assert model.labels_for_image(p, 0.35, 0.85)["general"] == ["1girl"]


def test_frames_require_fraction(model):
    P = np.stack([
        probs((1, 0, 0, 0), general=(0.9, 0.9, 0), chars=(0.9, 0)),
        probs((1, 0, 0, 0), general=(0.9, 0.0, 0), chars=(0.0, 0)),
        probs((1, 0, 0, 0), general=(0.9, 0.0, 0), chars=(0.0, 0)),
        probs((1, 0, 0, 0), general=(0.9, 0.0, 0), chars=(0.0, 0)),
    ])
    labels = model.labels_for_frames(P, 0.35, 0.85, general_frac=0.25, char_frac=0.5)
    assert labels["general"] == ["1girl", "long_hair"]  # long_hair in exactly 1/4 frames
    assert labels["characters"] == []                   # saber in 1/4 < 0.5


def test_rating_most_severe_reaching_fraction_wins(model):
    general, explicit, questionable = (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)
    P = np.stack([probs(general)] * 7 + [probs(questionable)] * 2 + [probs(explicit)])
    # 10% explicit qualifies at 0.10
    assert model.labels_for_frames(P, 0.35, 0.85, rating_frac=0.10)["rating"] == "explicit"
    # at 0.2 explicit (1/10) fails, questionable (2/10) wins
    assert model.labels_for_frames(P, 0.35, 0.85, rating_frac=0.2)["rating"] == "questionable"
    # at 0.5 only general qualifies
    assert model.labels_for_frames(P, 0.35, 0.85, rating_frac=0.5)["rating"] == "general"


def test_rating_falls_back_to_least_severe(model):
    P = np.stack([probs((0, 1, 0, 0)), probs((0, 0, 1, 0)), probs((0, 0, 0, 1))])
    assert model.labels_for_frames(P, 0.35, 0.85, rating_frac=0.5)["rating"] == "general"
