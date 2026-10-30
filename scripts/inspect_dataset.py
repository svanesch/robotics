"""Validate a dataset and render a contact sheet: pixi run inspect-data datasets/so101_pick"""
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/so101_pick")
info = json.loads((root / "meta/info.json").read_text())
eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]
print(f"{info['robot_type']}: {info['total_episodes']} episodes, {info['total_frames']} frames @ {info['fps']} fps")
print("tasks:", info["total_tasks"], "| features:", {k: v["shape"] for k, v in info["features"].items()})
cams = [k for k in info["features"] if k.startswith("observation.images.")]
from collections import Counter
print("tasks:", dict(Counter(e.get("task", "?") for e in eps)), "| splits:", dict(Counter(e.get("split", "?") for e in eps)),
      "| noisy:", sum(bool(e.get("noise")) for e in eps), "| mean length:", round(sum(e["length"] for e in eps) / len(eps)))
# one example per task in the contact sheet
seen, picks = set(), []
for e in eps:
    if e.get("task") not in seen:
        seen.add(e.get("task")); picks.append(e)
picks += [e for e in eps if e not in picks][: max(0, 4 - len(picks))]
rows = []
for e in picks[:4]:
    i = e["episode_index"]
    t = pq.read_table(root / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)).to_pandas() if False else pq.read_table(root / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i))
    assert t.num_rows == e["length"], "parquet/episodes.jsonl length mismatch"
    frames = []
    for c in cams:
        v = imageio.mimread(root / info["video_path"].format(episode_chunk=i // info["chunks_size"], video_key=c, episode_index=i), memtest=False)
        assert len(v) == e["length"], f"{c}: {len(v)} frames vs {e['length']}"
        frames.append(np.stack(v))
    idx = np.linspace(0, e["length"] - 1, 6).astype(int)
    rows.append(np.concatenate([np.concatenate([f[j] for j in idx], 1) for f in frames], 0))
    print(f"episode {i}: {e['length']} frames  '{e['tasks'][0]}'")
out = root / "contact_sheet.png"
imageio.imwrite(out, np.concatenate(rows, 0))
print("wrote", out)
