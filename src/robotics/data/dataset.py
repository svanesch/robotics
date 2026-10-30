"""LeRobot-v2.1-style dataset writer/merger (parquet per episode + one mp4 per camera per episode).

Layout::

    meta/info.json  meta/tasks.jsonl  meta/episodes.jsonl  meta/episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.<cam>/episode_000000.mp4
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def _image_stats(frames: np.ndarray) -> dict:
    sub = frames[:: max(1, len(frames) // 50)].astype(np.float64) / 255.0  # (n,H,W,3)
    ch = lambda f: f(sub, axis=(0, 1, 2)).reshape(3, 1, 1).tolist()  # noqa: E731
    return {"min": ch(np.min), "max": ch(np.max), "mean": ch(np.mean), "std": ch(np.std), "count": [len(frames)]}


class DatasetWriter:
    def __init__(self, root: str | Path, robot_type: str, fps: int, state_names: list[str],
                 cameras: tuple[str, ...], image_size: int, extra: dict | None = None):
        self.root = Path(root)
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "meta").mkdir(parents=True)
        self.robot_type, self.fps, self.cameras, self.image_size = robot_type, fps, cameras, image_size
        self.state_names = state_names
        self.extra = extra or {}
        self.tasks: dict[str, int] = {}
        self.episodes: list[dict] = []
        self.stats: list[dict] = []
        self.total_frames = 0
        self.env_dim = 0

    def add_episode(self, ep: dict, task: str, meta: dict | None = None) -> None:
        """ep: arrays keyed state (T,S), action (T,S), ee_pose (T,7), env_state (T,E), <cam> (T,H,W,3) uint8."""
        idx, T = len(self.episodes), len(ep["state"])
        task_index = self.tasks.setdefault(task, len(self.tasks))
        self.env_dim = ep["env_state"].shape[1]
        chunk = idx // CHUNK
        done = np.zeros(T, bool)
        done[-1] = True
        table = pa.table({
            "observation.state": pa.array(list(ep["state"].astype(np.float32))),
            "action": pa.array(list(ep["action"].astype(np.float32))),
            "observation.ee_pose": pa.array(list(ep["ee_pose"].astype(np.float32))),
            "observation.environment_state": pa.array(list(ep["env_state"].astype(np.float32))),
            "timestamp": pa.array((np.arange(T) / self.fps).astype(np.float32)),
            "frame_index": pa.array(np.arange(T, dtype=np.int64)),
            "episode_index": pa.array(np.full(T, idx, np.int64)),
            "index": pa.array(np.arange(self.total_frames, self.total_frames + T, dtype=np.int64)),
            "task_index": pa.array(np.full(T, task_index, np.int64)),
            "next.done": pa.array(done),
        })
        path = self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        stats = {"observation.state": _stats(ep["state"]), "action": _stats(ep["action"]),
                 "observation.ee_pose": _stats(ep["ee_pose"]), "observation.environment_state": _stats(ep["env_state"])}
        for cam in self.cameras:
            key = f"observation.images.{cam}"
            vpath = self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=idx)
            vpath.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(vpath, fps=self.fps, codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=1, quality=8) as w:
                for f in ep[cam]:
                    w.append_data(f)
            stats[key] = _image_stats(ep[cam])
        self.episodes.append({"episode_index": idx, "tasks": [task], "length": T, **(meta or {})})
        self.stats.append({"episode_index": idx, "stats": stats})
        self.total_frames += T

    def finalize(self) -> Path:
        write_meta(self.root, self.robot_type, self.fps, self.state_names, self.cameras, self.image_size,
                   self.env_dim, self.tasks, self.episodes, self.stats, self.total_frames)
        return self.root


def write_meta(root, robot_type, fps, state_names, cameras, image_size, env_dim, tasks, episodes, stats, total_frames):
    root = Path(root)
    n = len(state_names)
    vid = lambda: {"dtype": "video", "shape": [image_size, image_size, 3], "names": ["height", "width", "channel"],  # noqa: E731
                   "info": {"video.fps": fps, "video.height": image_size, "video.width": image_size,
                            "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False, "has_audio": False}}
    features = {
        "observation.state": {"dtype": "float32", "shape": [n], "names": state_names},
        "action": {"dtype": "float32", "shape": [n], "names": state_names},
        "observation.ee_pose": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        "observation.environment_state": {"dtype": "float32", "shape": [env_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = vid()
    info = {
        "codebase_version": "v2.1", "robot_type": robot_type, "total_episodes": len(episodes),
        "total_frames": total_frames, "total_tasks": len(tasks), "total_videos": len(episodes) * len(cameras),
        "total_chunks": (len(episodes) + CHUNK - 1) // CHUNK, "chunks_size": CHUNK, "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "custom_splits": {s: [e["episode_index"] for e in episodes if e.get("split", "train") == s]
                          for s in ("train", "val", "test")},
        "data_path": DATA_PATH, "video_path": VIDEO_PATH,
        "features": features,
    }
    meta = root / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))


def merge_shards(shards: list[Path], out: str | Path) -> Path:
    """Concatenate shard datasets (written by parallel workers) into one with global indices."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    tasks: dict[str, int] = {}
    episodes, stats, total = [], [], 0
    first_info = None
    for shard in shards:
        shard = Path(shard)
        info = json.loads((shard / "meta" / "info.json").read_text())
        first_info = first_info or info
        s_tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (shard / "meta/tasks.jsonl").read_text().splitlines()}
        s_eps = [json.loads(l) for l in (shard / "meta/episodes.jsonl").read_text().splitlines()]
        s_stats = [json.loads(l) for l in (shard / "meta/episodes_stats.jsonl").read_text().splitlines()]
        for e, st in zip(s_eps, s_stats):
            old, new = e["episode_index"], len(episodes)
            task = e["tasks"][0]
            ti = tasks.setdefault(task, len(tasks))
            t = pq.read_table(shard / DATA_PATH.format(episode_chunk=old // CHUNK, episode_index=old))
            T = t.num_rows
            t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", pa.array(np.full(T, new, np.int64)))
            t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(np.arange(total, total + T, dtype=np.int64)))
            t = t.set_column(t.schema.get_field_index("task_index"), "task_index", pa.array(np.full(T, ti, np.int64)))
            dst = out / DATA_PATH.format(episode_chunk=new // CHUNK, episode_index=new)
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst)
            for cam in [k for k in info["features"] if k.startswith("observation.images.")]:
                src = shard / VIDEO_PATH.format(episode_chunk=old // CHUNK, video_key=cam, episode_index=old)
                vdst = out / VIDEO_PATH.format(episode_chunk=new // CHUNK, video_key=cam, episode_index=new)
                vdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, vdst)
            episodes.append({**e, "episode_index": new})
            stats.append({**st, "episode_index": new})
            total += T
    (out / "meta").mkdir(parents=True, exist_ok=True)
    f = first_info["features"]
    cams = tuple(k.split(".")[-1] for k in f if k.startswith("observation.images."))
    write_meta(out, first_info["robot_type"], first_info["fps"], f["action"]["names"], cams,
               f[f"observation.images.{cams[0]}"]["shape"][0], f["observation.environment_state"]["shape"][0],
               tasks, episodes, stats, total)
    for shard in shards:
        shutil.rmtree(shard, ignore_errors=True)
    return out
