"""Dataset for the small VLA: reads a LeRobot-style dataset (parquet + mp4) written by robotics.data.dataset.

Videos are decoded once into on-disk uint8 memmaps (`<root>/cache/<split>_<cam>.npy`); batches then read frames
directly. Only whitelisted keys reach the model: both camera images, the joint state, the instruction and the
future action chunk. Privileged simulator state (`observation.environment_state`, `observation.ee_pose`) is never loaded.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from robotics.policy.tokenizer import Tokenizer

CAMERAS = ("wrist", "external")
MODEL_INPUT_KEYS = ("wrist", "external", "state", "tokens")  # what the network may see
LABEL_KEYS = ("actions", "is_pad")


def _decode(path: str) -> np.ndarray:
    return np.stack(imageio.mimread(path, memtest=False))


def build_cache(root: Path, split: str, workers: int = 6) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    eps = {e["episode_index"]: e for e in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())}
    ids = info["custom_splits"][split]
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    lengths = [eps[i]["length"] for i in ids]
    total = int(sum(lengths))
    first = np.stack(_decode(str(root / info["video_path"].format(episode_chunk=ids[0] // info["chunks_size"],
                                                                video_key="observation.images.wrist", episode_index=ids[0]))))
    h, w = first.shape[1:3]
    for cam in CAMERAS:
        out = np.lib.format.open_memmap(cache / f"{split}_{cam}.npy", mode="w+", dtype=np.uint8, shape=(total, h, w, 3))
        paths = [str(root / info["video_path"].format(episode_chunk=i // info["chunks_size"],
                                                      video_key=f"observation.images.{cam}", episode_index=i)) for i in ids]
        pos = 0
        with ProcessPoolExecutor(workers) as pool:
            for n, frames in zip(lengths, pool.map(_decode, paths, chunksize=4)):
                assert len(frames) == n, "video/parquet length mismatch"
                out[pos:pos + n] = frames
                pos += n
        out.flush()
    state, action, starts = [], [], []
    pos = 0
    for i, n in zip(ids, lengths):
        t = pq.read_table(root / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i),
                          columns=["observation.state", "action"]).to_pydict()
        state.append(np.asarray(t["observation.state"], np.float32))
        action.append(np.asarray(t["action"], np.float32))
        starts.append(pos)
        pos += n
    np.savez(cache / f"{split}_arrays.npz", state=np.concatenate(state), action=np.concatenate(action),
             starts=np.asarray(starts), lengths=np.asarray(lengths), episode_ids=np.asarray(ids))


class VLADataset(Dataset):
    def __init__(self, root: str | Path, split: str, chunk: int = 16, stats: dict | None = None, augment: bool = False):
        self.root, self.split, self.chunk, self.augment = Path(root), split, chunk, augment
        cache = self.root / "cache"
        if not (cache / f"{split}_arrays.npz").exists():
            build_cache(self.root, split)
        z = np.load(cache / f"{split}_arrays.npz")
        self.state, self.action = z["state"], z["action"]
        self.starts, self.lengths, self.episode_ids = z["starts"], z["lengths"], z["episode_ids"]
        self.images = {c: np.load(cache / f"{split}_{c}.npy", mmap_mode="r") for c in CAMERAS}
        eps = {e["episode_index"]: e for e in map(json.loads, (self.root / "meta/episodes.jsonl").read_text().splitlines())}
        self.meta = [eps[i] for i in self.episode_ids]
        tok = Tokenizer()
        self.tokens = np.stack([tok.encode(e["tasks"][0]) for e in self.meta])
        self.frame_ep = np.repeat(np.arange(len(self.lengths)), self.lengths)
        self.stats = stats or {"state_mean": self.state.mean(0), "state_std": self.state.std(0) + 1e-6,
                               "action_mean": self.action.mean(0), "action_std": self.action.std(0) + 1e-6}

    def __len__(self) -> int:
        return len(self.frame_ep)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep = self.frame_ep[idx]
        s, n = self.starts[ep], self.lengths[ep]
        t = idx - s
        steps = np.minimum(np.arange(t, t + self.chunk), n - 1)
        a = (self.action[s + steps] - self.stats["action_mean"]) / self.stats["action_std"]
        return {
            "wrist": torch.from_numpy(np.array(self.images["wrist"][idx])),
            "external": torch.from_numpy(np.array(self.images["external"][idx])),
            "state": torch.from_numpy((self.state[idx] - self.stats["state_mean"]) / self.stats["state_std"]).float(),
            "tokens": torch.from_numpy(self.tokens[ep]),
            "actions": torch.from_numpy(a).float(),
            "is_pad": torch.from_numpy(np.arange(t, t + self.chunk) >= n),
        }
