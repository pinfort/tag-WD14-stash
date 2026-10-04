"""Settings for the WD14 -> Stash tagging pipeline. Edit to taste."""

# --- Stash connection ---------------------------------------------------------
STASH_URL = "http://192.168.2.55/graphql"
STASH_API_KEY = ""  # Stash: Settings > Security > API Key (leave empty if auth is off)

# Map Stash's file paths to paths this script can read (needed for scenes when
# the script runs on a different machine or outside Stash's Docker container).
# Example: {"/data/": "D:/media/"}
PATH_MAP = {}

# --- Model ----------------------------------------------------------------------
MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"  # or wd-swinv2-tagger-v3 (faster, less accurate)
MODEL_REVISION = None  # pin to a commit hash once you're happy, so results stay consistent
DEVICE = None          # None = auto (cuda if available)
BATCH_SIZE = 4  # EVA02-Large is ~3x the size of SwinV2; raise if VRAM allows

# Changing the model? Change this too, so items get re-tagged by the new one.
MARKER_TAG = "wd14:eva02-large-v3"

# --- Thresholds -----------------------------------------------------------------
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85

# Scenes: frames sampled across the video; a tag is kept if it appears in at
# least this fraction of frames.
SCENE_FRAME_INTERVAL = 10       # seconds between sampled frames
SCENE_MAX_FRAMES = 32
SCENE_GENERAL_MIN_FRACTION = 0.25
SCENE_CHARACTER_MIN_FRACTION = 0.15
SCENE_RATING_MIN_FRACTION = 0.10  # most severe rating reaching this share of frames wins

# --- Tag hierarchy --------------------------------------------------------------
ROOT_TAG = "WD14"  # group tags become "WD14: Hair", "WD14: Characters", etc.
SERIES_PREFIX = "[Series] "

DB_PATH = "wd14_embeddings.sqlite"

# --- General tag grouping -------------------------------------------------------
# Checked in order; first match wins. An entry matches either the full Danbooru
# tag (e.g. "open_mouth") or any single word of it (e.g. "hair").
COUNT_TAGS = {
    "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
    "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
    "1other", "2others", "multiple_others", "solo", "solo_focus", "no_humans", "male_focus",
}

GENERAL_GROUPS = [
    ("Expression", {
        "smile", "blush", "grin", "tears", "crying", "frown", "pout", "angry", "sad",
        "surprised", "embarrassed", "expressionless", "smirk", "closed_eyes", "open_mouth",
        "closed_mouth", "one_eye_closed", "tongue_out", ":d", ":o", ":p", ";)",
    }),
    ("Hair", {
        "hair", "hairband", "hairclip", "hairpin", "ponytail", "twintails", "braid",
        "bangs", "ahoge", "sidelocks", "bun", "bob", "drill",
    }),
    ("Eyes", {"eyes", "eye", "eyelashes", "pupils", "heterochromia", "eyepatch"}),
    ("Body", {
        "breasts", "ears", "tail", "wings", "horns", "skin", "fang", "fangs", "teeth",
        "navel", "thighs", "legs", "feet", "barefoot", "muscular", "abs", "collarbone",
        "halo", "nails", "lips",
    }),
    ("Clothing", {
        "shirt", "skirt", "dress", "uniform", "jacket", "coat", "pants", "shorts",
        "thighhighs", "socks", "kneehighs", "boots", "shoes", "gloves", "hat", "cap",
        "swimsuit", "bikini", "ribbon", "bow", "necktie", "sleeves", "sleeveless", "hood",
        "hoodie", "kimono", "apron", "armor", "panties", "bra", "leotard", "pantyhose",
        "cape", "scarf", "collar", "choker", "earrings", "jewelry", "necklace", "glasses",
        "serafuku", "sweater", "vest", "belt", "headwear", "headband", "mask", "costume",
    }),
    ("Pose/Action", {
        "standing", "sitting", "lying", "kneeling", "squatting", "walking", "running",
        "looking", "holding", "hand", "hands", "arm", "arms", "leg", "pose", "outstretched",
        "crossed", "from_behind", "from_side", "from_above", "from_below", "upper_body",
        "full_body", "cowboy_shot", "portrait", "close-up", "v",
    }),
    ("Background/Setting", {
        "background", "outdoors", "indoors", "sky", "cloud", "clouds", "tree", "trees",
        "water", "beach", "ocean", "night", "day", "room", "bed", "city", "building",
        "window", "grass", "flower", "flowers", "snow", "rain", "sunset", "scenery",
        "forest", "street",
    }),
]
