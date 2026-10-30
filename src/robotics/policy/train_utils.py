"""Shared helpers: batch -> device tensors, GPU augmentation, checkpoint (de)serialisation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from robotics.policy.model import VLA, VLAConfig
from robotics.policy.tokenizer import Tokenizer


def to_device(batch: dict, device: torch.device) -> dict:
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    for cam in ("wrist", "external"):  # (B,H,W,3) uint8 -> (B,3,H,W) float
        out[cam] = out[cam].permute(0, 3, 1, 2).float() / 255.0
    return out


def augment(img: torch.Tensor, shift: int, brightness: float, contrast: float, saturation: float) -> torch.Tensor:
    """img: (B,3,H,W) in [0,1]. Random translation (replicate pad) + colour jitter, per sample."""
    B, _, H, W = img.shape
    if shift > 0:
        pad = F.pad(img, (shift,) * 4, mode="replicate")
        dx = torch.randint(0, 2 * shift + 1, (B,), device=img.device)
        dy = torch.randint(0, 2 * shift + 1, (B,), device=img.device)
        ix = (dx[:, None] + torch.arange(W, device=img.device)[None])  # (B,W)
        iy = (dy[:, None] + torch.arange(H, device=img.device)[None])  # (B,H)
        img = pad[torch.arange(B, device=img.device)[:, None, None, None], torch.arange(3, device=img.device)[None, :, None, None],
                  iy[:, None, :, None], ix[:, None, None, :]]
    def rnd(a):
        return (1 + (torch.rand(B, 1, 1, 1, device=img.device) * 2 - 1) * a)
    img = img * rnd(brightness)
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    img = (img - mean) * rnd(contrast) + mean
    gray = img.mean(dim=1, keepdim=True)
    img = (img - gray) * rnd(saturation) + gray
    return img.clamp(0, 1)


def save_checkpoint(path, model: VLA, cfg: dict, stats: dict, step: int, extra: dict | None = None, optim=None) -> None:
    payload = {"model": model.state_dict(), "model_cfg": model.cfg.__dict__, "train_cfg": cfg,
               "stats": {k: np.asarray(v) for k, v in stats.items()}, "step": step, "vocab": Tokenizer().vocab, **(extra or {})}
    if optim is not None:
        payload["optim"] = optim.state_dict()
    torch.save(payload, path)


def load_policy(path, device: str | torch.device = "cpu") -> tuple[VLA, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["model_cfg"])
    mc["pretrained"] = False  # weights come from the checkpoint
    model = VLA(VLAConfig(**mc)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
