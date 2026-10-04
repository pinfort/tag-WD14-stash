#!/usr/bin/env python3
"""Find characters WD14 doesn't know, using embeddings saved by tagger.py.

1. In Stash, create a tag (e.g. "Frieren") and apply it to 5-20 images by hand.
2. python characters.py suggest "Frieren" --dry-run      # check scores first
   python characters.py suggest "Frieren" --threshold 0.8
   -> matches get the review tag "Character?/Frieren"
3. In Stash, filter by "Character?/Frieren" and remove the tag from wrong matches.
4. python characters.py promote "Frieren" --series "Sousou no Frieren"
   -> remaining matches get "Frieren"; the review tag is removed.
Items already suggested once are never suggested again for that character.
"""
import argparse
import sys

import numpy as np

import config as cfg
from db import EmbDB
from stash_client import Stash, TagManager, chunks, tag_filter

KINDS = {"images": ["image"], "scenes": ["scene"], "all": ["image", "scene"]}


def review_name(name):
    return f"Character?/{name}"


def cmd_suggest(a, stash, tm, db):
    seed_tag = tm.find(a.name)
    if not seed_tag:
        sys.exit(f"Tag {a.name!r} not found in Stash. Create it and tag some seed images first.")
    root = tm.get(cfg.ROOT_TAG)
    parent = tm.get(f"{cfg.ROOT_TAG}: Character Review", parents=[root])
    review = tm.get(review_name(a.name), parents=[parent])

    for kind in KINDS[a.kind]:
        ids, M = db.load(kind, cfg.MODEL_REPO)
        if not ids:
            print(f"[{kind}] no embeddings yet; run tagger.py first")
            continue
        row = {i: n for n, i in enumerate(ids)}
        seeds = stash.find_ids(kind, tag_filter(seed_tag))
        seed_rows = [row[s] for s in seeds if s in row]
        if len(seed_rows) < a.min_seeds:
            print(f"[{kind}] only {len(seed_rows)} seeds with embeddings (need {a.min_seeds})")
            continue

        S = M[seed_rows]
        if a.mode == "centroid":
            c = S.mean(axis=0)
            c /= np.linalg.norm(c)
            scores = M @ c
            seed_scores = S @ c
            print(f"[{kind}] seed scores: min {seed_scores.min():.3f}, "
                  f"median {np.median(seed_scores):.3f}  (start the threshold a bit below min)")
        else:
            scores = (M @ S.T).max(axis=1)

        skip = set(seeds) | db.suggested_ids(a.name, kind)
        ranked = [(ids[n], float(scores[n])) for n in np.argsort(-scores) if ids[n] not in skip]
        hits = [(i, s) for i, s in ranked if s >= a.threshold][:a.max or None]
        print(f"[{kind}] {len(hits)} matches at >= {a.threshold}")
        print("  top candidates:", ", ".join(f"{i}:{s:.3f}" for i, s in ranked[:10]))
        if a.dry_run or not hits:
            continue
        for batch in chunks(hits, 200):
            stash.update_tags(kind, [i for i, _ in batch], [review], "ADD")
        db.mark_suggested(a.name, kind, hits)
        db.commit()
        print(f"  tagged with {review_name(a.name)!r}; review them in Stash, then run promote")


def cmd_promote(a, stash, tm, db):
    review = tm.find(review_name(a.name))
    if not review:
        sys.exit(f"No review tag {review_name(a.name)!r}; run suggest first.")
    root = tm.get(cfg.ROOT_TAG)
    chars = tm.get(f"{cfg.ROOT_TAG}: Characters", parents=[root])
    parent = tm.get(f"{cfg.SERIES_PREFIX}{a.series}", parents=[chars]) if a.series else chars
    real = tm.get(a.name, parents=[parent])

    for kind in KINDS[a.kind]:
        ids = stash.find_ids(kind, tag_filter(review))
        print(f"[{kind}] promoting {len(ids)} items to {a.name!r}")
        if a.dry_run:
            continue
        for batch in chunks(ids, 200):
            stash.update_tags(kind, batch, [real], "ADD")
            stash.update_tags(kind, batch, [review], "REMOVE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suggest", help="tag look-alikes of a hand-tagged character for review")
    s.add_argument("name", help="existing Stash tag applied to seed images")
    s.add_argument("--threshold", type=float, default=0.8)
    s.add_argument("--mode", choices=["centroid", "max"], default="centroid",
                   help="centroid: similarity to the seeds' average (robust); "
                        "max: to the closest seed (catches varied outfits, noisier)")
    s.add_argument("--min-seeds", type=int, default=3)
    s.add_argument("--max", type=int, default=0, help="cap on matches (0 = no cap)")

    p = sub.add_parser("promote", help="turn reviewed matches into the real character tag")
    p.add_argument("name")
    p.add_argument("--series", help="series parent tag, e.g. 'Sousou no Frieren'")

    for sp in (s, p):
        sp.add_argument("--kind", choices=list(KINDS), default="images")
        sp.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    stash = Stash(cfg.STASH_URL, cfg.STASH_API_KEY)
    tm = TagManager(stash, dry_run=a.dry_run)
    db = EmbDB(cfg.DB_PATH)
    {"suggest": cmd_suggest, "promote": cmd_promote}[a.cmd](a, stash, tm, db)


if __name__ == "__main__":
    main()
