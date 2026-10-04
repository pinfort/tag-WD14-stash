"""SQLite store for embeddings (keyed by Stash ID) and character-suggestion history."""
import sqlite3

import numpy as np


class EmbDB:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS emb(
                kind TEXT, id TEXT, model TEXT, vec BLOB, PRIMARY KEY(kind, id, model));
            CREATE TABLE IF NOT EXISTS suggested(
                character TEXT, kind TEXT, id TEXT, score REAL, PRIMARY KEY(character, kind, id));
        """)

    def put(self, kind, item_id, model, vec):
        self.con.execute("INSERT OR REPLACE INTO emb VALUES (?,?,?,?)",
                         (kind, str(item_id), model, np.asarray(vec, np.float32).tobytes()))

    def load(self, kind, model):
        rows = self.con.execute("SELECT id, vec FROM emb WHERE kind=? AND model=?",
                                (kind, model)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), np.float32)
        return [r[0] for r in rows], np.stack([np.frombuffer(r[1], np.float32) for r in rows])

    def suggested_ids(self, character, kind):
        rows = self.con.execute("SELECT id FROM suggested WHERE character=? AND kind=?",
                                (character, kind))
        return {r[0] for r in rows}

    def mark_suggested(self, character, kind, pairs):
        self.con.executemany("INSERT OR REPLACE INTO suggested VALUES (?,?,?,?)",
                             [(character, kind, str(i), s) for i, s in pairs])

    def commit(self):
        self.con.commit()
