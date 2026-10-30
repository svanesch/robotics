"""Closed-vocabulary word tokenizer: every instruction the data generator can produce is covered."""
from __future__ import annotations

import re

import numpy as np

from robotics.data.randomize import LIFT_TEMPLATES, PLACE_TEMPLATES
from robotics.sim.builder import CUBE_COLORS

PAD, UNK, CLS = 0, 1, 2


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_vocab() -> dict[str, int]:
    words = {"<pad>": PAD, "<unk>": UNK, "<cls>": CLS}
    corpus = [t.format(c=c, side=s) for t in LIFT_TEMPLATES + PLACE_TEMPLATES for c in CUBE_COLORS for s in ("left", "right")]
    for text in corpus:
        for w in _words(text):
            words.setdefault(w, len(words))
    return words


class Tokenizer:
    def __init__(self, max_len: int = 16):
        self.vocab = build_vocab()
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> np.ndarray:
        ids = [CLS] + [self.vocab.get(w, UNK) for w in _words(text)]
        ids = ids[: self.max_len] + [PAD] * max(0, self.max_len - len(ids))
        return np.asarray(ids, np.int64)
