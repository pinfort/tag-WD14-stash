"""Shared fixtures. torch/timm are stubbed when missing so the pure logic in
wd14.py and tagger.py can be tested without downloading or loading a model."""
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub_heavy_modules():
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
        return
    except ImportError:
        pass
    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: (lambda f: f)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    timm = types.ModuleType("timm")
    timm_data = types.ModuleType("timm.data")
    timm_data.create_transform = timm_data.resolve_data_config = None
    timm.data = timm_data
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("timm", timm)
    sys.modules.setdefault("timm.data", timm_data)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        hf = types.ModuleType("huggingface_hub")
        hf.hf_hub_download = None
        sys.modules["huggingface_hub"] = hf


_stub_heavy_modules()


class FakeStash:
    """In-memory stand-in for stash_client.Stash."""

    def __init__(self, tags=(), items=None):
        self.tags = [dict(t) for t in tags]  # {"id", "name", "aliases"}
        self.items = items or {}             # {(kind, id): set(tag_ids)}
        self.created = []
        self.updates = []
        self._next = 1000
        self.reject_aliases = False

    def all_tags(self):
        return self.tags

    def create_tag(self, name, aliases=None, parent_ids=None, description=None):
        if aliases and self.reject_aliases:
            raise RuntimeError("alias clash")
        self._next += 1
        tid = str(self._next)
        self.tags.append({"id": tid, "name": name, "aliases": aliases or []})
        self.created.append({"id": tid, "name": name, "aliases": aliases or [],
                             "parent_ids": parent_ids or [], "description": description})
        return tid

    def find_ids(self, kind, item_filter=None):
        out = []
        for (k, iid), tags in self.items.items():
            if k != kind:
                continue
            if item_filter:
                f = item_filter["tags"]
                has = f["value"][0] in tags
                if (f["modifier"] == "INCLUDES") != has:
                    continue
            out.append(iid)
        return out

    def update_tags(self, kind, ids, tag_ids, mode="ADD"):
        self.updates.append((kind, list(ids), list(tag_ids), mode))
        for iid in ids:
            tags = self.items.setdefault((kind, iid), set())
            (tags.update if mode == "ADD" else tags.difference_update)(tag_ids)


@pytest.fixture
def fake_stash():
    return FakeStash()
