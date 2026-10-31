"""Pictures of what a policy is trained and validated on (shown by the dashboard and saved next to every checkpoint)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from robotics.policy.train_utils import to_device

PALETTE = [(255, 107, 107), (255, 176, 56), (94, 214, 129), (86, 180, 233), (167, 139, 250), (244, 114, 182), (190, 190, 190)]


def to_u8(t):
    return (t.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def train_view(x, n=4):
    """What the network is trained on right now: (wrist | external) after augmentation, 2 x 2 grid."""
    w, e = to_u8(x["wrist"][:n]), to_u8(x["external"][:n])
    rows = [np.concatenate([w[i], e[i]], 1) for i in range(n)]
    return np.concatenate([np.concatenate(rows[:2], 0), np.concatenate(rows[2:], 0)], 1)


def chunk_plot(gt, pred, size=(256, 128)):
    """Truth (dim) vs prediction (bright) of the next K actions: one small panel per joint with its own scale.

    gt, pred: (K, A) normalised actions; the last column is the gripper. Flat joints stay flat (minimum span), so only real
    motion shows. The last panel prints the mean absolute error of this sample (normalised units).
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (18, 22, 29))
    d = ImageDraw.Draw(img)
    K, A = gt.shape
    cols = 4
    pw, ph = size[0] // cols, size[1] // 2
    for j in range(A):
        x0, y0 = (j % cols) * pw, (j // cols) * ph
        d.rectangle([x0, y0, x0 + pw - 1, y0 + ph - 1], outline=(38, 44, 54))
        both = np.concatenate([gt[:, j], pred[:, j]])
        mid, span = (both.max() + both.min()) / 2, max(float(both.max() - both.min()), 0.4)
        lo, hi = mid - span / 2, mid + span / 2
        pt = lambda i, v: (x0 + 4 + i * (pw - 8) / max(K - 1, 1), y0 + ph - 6 - (v - lo) / (hi - lo) * (ph - 20))  # noqa: E731
        c = PALETTE[j % len(PALETTE)]
        d.line([pt(i, gt[i, j]) for i in range(K)], fill=tuple(int(0.4 * v) for v in c), width=3)
        d.line([pt(i, pred[i, j]) for i in range(K)], fill=c, width=1)
        d.text((x0 + 3, y0 + 1), "grip" if j == A - 1 else f"j{j + 1}", fill=c)
    if A < cols * 2:
        x0, y0 = (A % cols) * pw, (A // cols) * ph
        d.text((x0 + 3, y0 + 3), f"err {np.abs(gt - pred).mean():.2f}", fill=(235, 235, 235))
        d.text((x0 + 3, y0 + 16), "dim=truth", fill=(120, 126, 138))
        d.text((x0 + 3, y0 + 28), "bright=model", fill=(200, 205, 215))
    return np.asarray(img)


@torch.no_grad()
def val_view(model, val_ds, device, idxs):
    """Held-out frames: (wrist | external) next to truth-vs-prediction of the following action chunk."""
    model.eval()
    batch = torch.utils.data.default_collate([val_ds[int(i)] for i in idxs])
    x = to_device(batch, device)
    pred = model(x["wrist"], x["external"], x["state"], x["tokens"]).cpu().numpy()
    gt = x["actions"].cpu().numpy()
    w, e = to_u8(x["wrist"]), to_u8(x["external"])
    model.train()
    rows = [np.concatenate([w[i], e[i], chunk_plot(gt[i], pred[i], (256, w.shape[1]))], 1) for i in range(len(idxs))]
    if len(rows) % 2:
        rows.append(np.zeros_like(rows[0]))
    half = len(rows) // 2
    return np.concatenate([np.concatenate(rows[:half], 0), np.concatenate(rows[half:], 0)], 1)  # two columns of samples


def save_view(path, image: np.ndarray) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image), quality=88)
