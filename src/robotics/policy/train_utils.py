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


def robot_for_checkpoint(ck: dict, override: str | None = None) -> str:
    """Which robot a checkpoint was trained for (from the dataset it was trained on), unless one is given explicitly."""
    if override:
        return override
    ds = ck["train_cfg"]["dataset"]
    ds = " ".join(ds) if isinstance(ds, list) else str(ds)
    return "ur5e" if "ur5e" in ds else "so101"


def load_policy(path, device: str | torch.device = "cpu") -> tuple[VLA, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["model_cfg"])
    mc["pretrained"] = False  # weights come from the checkpoint
    model = VLA(VLAConfig(**mc)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def policy_loss(pred, actions, is_pad, stats, grip_bce: bool, bce_weight: float = 1.0):
    """L1 on the (normalised) joint targets. With grip_bce the last output is a logit for "close" (open/close is a decision,
    not a number): binary cross-entropy against the expert's command (raw > 0.5), so there is no value in between to settle on."""
    valid = (~is_pad)[..., None].float()
    if not grip_bce:
        return ((pred - actions).abs() * valid).sum() / (valid.sum() * pred.shape[-1] + 1e-8)
    l1 = ((pred[..., :-1] - actions[..., :-1]).abs() * valid).sum() / (valid.sum() * (pred.shape[-1] - 1) + 1e-8)
    mean = torch.as_tensor(stats["action_mean"][-1], device=actions.device).float()
    std = torch.as_tensor(stats["action_std"][-1], device=actions.device).float()
    target = ((actions[..., -1] * std + mean) > 0.5).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[..., -1], target, reduction="none")
    bce = (bce * valid[..., 0]).sum() / (valid.sum() + 1e-8)
    return l1 + bce_weight * bce
