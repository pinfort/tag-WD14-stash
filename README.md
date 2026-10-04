# WD14 -> Stash tagger

## Setup
1. `pip install -r requirements.txt` (install the CUDA build of torch first if you have a GPU, e.g.
   `pip install torch --index-url https://download.pytorch.org/whl/cu128`; check with
   `python -c "import torch; print(torch.cuda.is_available())"`)
2. Edit `config.py`: `STASH_URL`, `STASH_API_KEY`, and `PATH_MAP` if needed for scenes.

## Auto-tagging
    python tagger.py --kind images --limit 20 --dry-run   # preview, writes nothing
    python tagger.py --kind images --limit 200            # small real run, check in Stash
    python tagger.py --kind all                           # everything

Re-running only processes items without the marker tag (`wd14:eva02-large-v3`).

Resulting hierarchy:

    WD14
    ├── wd14:eva02-large-v3                 (marker)
    ├── WD14: Rating      -> rating:general / sensitive / questionable / explicit
    ├── WD14: Characters  -> [Series] Fate -> Saber (Fate)   (alias saber_(fate))
    ├── WD14: General     -> WD14: Hair, Eyes, Clothing, Count, ... , Other
    └── WD14: Character Review -> Character?/<name>

Existing tags with the same name or alias are reused, not duplicated.
All created tags have "ignore auto tag" on.

## New characters (unknown to WD14)
    # tag 5-20 images by hand with e.g. "Frieren" in Stash, then:
    python characters.py suggest "Frieren" --dry-run
    python characters.py suggest "Frieren" --threshold 0.8
    # in Stash: remove "Character?/Frieren" from wrong matches
    python characters.py promote "Frieren" --series "Sousou no Frieren"

## Undo
In Stash's Tags page, filter by parent tag `WD14` (include sub-tags) and delete
those tags, plus `wd14_embeddings.sqlite`. Note: if an existing tag of yours was
reused (same name), it stays applied to the items WD14 added it to.

## Tests
    pip install -r requirements-dev.txt
    python -m pytest

torch/timm aren't needed: the tests stub them and never load the model or contact Stash.
