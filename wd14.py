"""WD14 v3 tagger via timm: returns tag probabilities and image embeddings."""
import csv
import math

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import create_transform, resolve_data_config

Image.MAX_IMAGE_PIXELS = None

KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3",
    "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}


def display_name(raw):
    return raw if raw in KAOMOJI else raw.replace("_", " ")


def _to_rgb(img):
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    return img.convert("RGB")


def _pad_square(img):
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


class WD14:
    def __init__(self, repo, revision=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = f"hf-hub:{repo}" + (f"@{revision}" if revision else "")
        self.model = timm.create_model(name, pretrained=True).eval().to(self.device)
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg,
                                                                model=self.model))
        path = hf_hub_download(repo, "selected_tags.csv", revision=revision)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"] for r in rows]
        cats = np.array([int(r["category"]) for r in rows])
        self.rating_idx = np.where(cats == 9)[0]   # general, sensitive, questionable, explicit
        self.general_idx = np.where(cats == 0)[0]
        self.char_idx = np.where(cats == 4)[0]

    @torch.inference_mode()
    def predict(self, images):
        """images: list of PIL images -> (probs [N, tags], embeddings [N, dim], L2-normalized)"""
        x = torch.stack([self.transform(_pad_square(_to_rgb(im))) for im in images])
        x = x[:, [2, 1, 0]].to(self.device)  # model expects BGR
        feats = self.model.forward_features(x)
        emb = self.model.forward_head(feats, pre_logits=True)
        logits = self.model.get_classifier()(emb)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
        return probs, emb

    def labels_for_frames(self, P, general_thr, char_thr,
                          general_frac=1.0, char_frac=1.0, rating_frac=1.0):
        """P: [frames, tags]. A tag is kept if above threshold in >= frac of frames."""
        n = len(P)

        def pick(idx, thr, frac):
            hits = (P[:, idx] >= thr).mean(axis=0)
            order = np.argsort(-P[:, idx].mean(axis=0))
            return [self.names[idx[i]] for i in order if hits[i] >= frac - 1e-9]

        top = self.rating_idx[P[:, self.rating_idx].argmax(axis=1)]
        need = max(1, math.ceil(rating_frac * n))
        rating = self.names[self.rating_idx[0]]
        for i in self.rating_idx:  # least -> most severe; most severe that qualifies wins
            if (top == i).sum() >= need:
                rating = self.names[i]
        return {
            "rating": rating,
            "characters": pick(self.char_idx, char_thr, char_frac),
            "general": pick(self.general_idx, general_thr, general_frac),
        }

    def labels_for_image(self, p, general_thr, char_thr):
        return self.labels_for_frames(p[None, :], general_thr, char_thr)
