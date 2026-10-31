"""Render validation samples for a finished training run so the dashboard's history page can show them.

  pixi run python scripts/make_views.py --run d2          # uses runs/d2/best.pt
Writes runs/<run>/views/val_best.jpg (truth vs prediction on held-out frames) and train_best.jpg (raw training frames).
Runs whose checkpoint was made with an older model version cannot be loaded and are skipped.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from robotics.policy.data import VLADataset
from robotics.policy.train_utils import load_policy
from robotics.policy.views import save_view, train_view, val_view

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--checkpoint", default="best.pt")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
run = Path("runs") / a.run
try:
    model, ck = load_policy(run / a.checkpoint, a.device)
except Exception as e:  # checkpoint from an older architecture
    raise SystemExit(f"{a.run}: cannot load {a.checkpoint} with the current model code ({type(e).__name__}); skipped")
root = ck["train_cfg"]["dataset"]
root = root[0] if isinstance(root, list) else root
stats = {k: np.asarray(v) for k, v in ck["stats"].items()}
val = VLADataset(root, "val", model.cfg.chunk, stats=stats)
tr = VLADataset(root, "train", model.cfg.chunk, stats=stats)
idx = np.random.default_rng(0).choice(len(val), 6, replace=False)
save_view(run / "views" / "val_best.jpg", val_view(model, val, torch.device(a.device), idx))
raw = torch.utils.data.default_collate([tr[int(i)] for i in np.random.default_rng(1).choice(len(tr), 4, replace=False)])
x = {k: (raw[k].permute(0, 3, 1, 2).float() / 255.0) for k in ("wrist", "external")}
save_view(run / "views" / "train_best.jpg", train_view(x))
print(f"wrote {run}/views/val_best.jpg and train_best.jpg (checkpoint step {ck['step']})")
