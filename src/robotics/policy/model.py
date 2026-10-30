"""Small VLA: ResNet18 vision (both cameras) + word-level text encoder + joint state -> transformer -> action chunk.

ACT-style: a transformer encoder fuses [state, text tokens, wrist tokens, external tokens]; a transformer decoder with
learned queries predicts the next `chunk` absolute joint targets (+ gripper command) in normalised units.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision


@dataclass
class VLAConfig:
    vocab_size: int
    state_dim: int = 6
    action_dim: int = 6
    chunk: int = 16
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    ff: int = 1024
    dropout: float = 0.1
    max_text: int = 16
    image_size: int = 128
    pretrained: bool = True
    freeze_bn: bool = True
    stage: int = 3  # last ResNet stage used (3 = stride 16 -> 8x8 tokens per camera; 4 = stride 32)
    film: bool = True  # text-conditioned FiLM on the last two ResNet stages (zero-initialised: starts as plain ResNet)
    text_layers: int = 2
    readout_queries: int = 4  # prompt-driven spatial-softmax readouts per camera (0 = off)


def _sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """Fixed 2D sinusoidal position embedding, (h*w, d)."""
    assert d % 4 == 0
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(d // 4, dtype=torch.float32) / (d // 4)))
    out = [f(p.flatten()[:, None] * omega[None]) for p in (y, x) for f in (torch.sin, torch.cos)]
    return torch.cat(out, dim=1)


class VLA(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        layers = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4][: 4 + cfg.stage]
        self.backbone = nn.Sequential(*layers)
        chans = {5: 128, 6: 256, 7: 512}  # channels after backbone index k (layer2, layer3, layer4)
        self.film_at = {4 + cfg.stage - 2: 0, 4 + cfg.stage - 1: 1}  # FiLM after the last two stages
        self.proj = nn.Conv2d({3: 256, 4: 512}[cfg.stage], d, 1)
        fh = cfg.image_size // {3: 16, 4: 32}[cfg.stage]
        self.register_buffer("img_pos", _sincos_2d(fh, fh, d), persistent=False)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, fh), torch.linspace(-1, 1, fh), indexing="ij")
        self.register_buffer("grid_xy", torch.stack([xs.flatten(), ys.flatten()], 1), persistent=False)  # (N,2)
        K = cfg.readout_queries
        if K:  # the prompt asks K questions of each camera's feature map ("where is the red block?") and gets back
            self.query_proj = nn.Linear(d, K * d)  # soft coordinates + an attended feature vector per question
            self.key_proj = nn.Linear(d, d)
            self.coord_proj = nn.Linear(2, d)
            self.readout_cam = nn.Parameter(torch.zeros(2, K, d))
            nn.init.normal_(self.readout_cam, std=0.02)
        self.cam_emb = nn.Parameter(torch.zeros(2, 1, d))
        self.text_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=0)
        self.text_pos = nn.Parameter(torch.zeros(cfg.max_text, d))
        tl = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True)
        self.text_enc = nn.TransformerEncoder(tl, cfg.text_layers)  # the prompt gets its own encoder; token 0 (<cls>) pools it
        film_ch = [chans[k] for k in sorted(self.film_at)]
        self.films = nn.ModuleList([nn.Linear(d, 2 * c) for c in film_ch]) if cfg.film else None
        for f in (self.films or []):
            nn.init.zeros_(f.weight), nn.init.zeros_(f.bias)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.type_emb = nn.Parameter(torch.zeros(3, 1, d))  # state / text / vision
        enc = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg.enc_layers)
        dec = nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, cfg.dec_layers)
        self.queries = nn.Parameter(torch.randn(cfg.chunk, d) * 0.02)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.action_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        nn.init.normal_(self.cam_emb, std=0.02), nn.init.normal_(self.text_pos, std=0.02), nn.init.normal_(self.type_emb, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_bn:  # tiny batches + pretrained statistics: keep BatchNorm in eval mode
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def encode_text(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (B,L) -> per-token features (B,L,d) and the pooled prompt vector (B,d)."""
        t = self.text_emb(tokens) + self.text_pos[None, : tokens.shape[1]]
        t = self.text_enc(t, src_key_padding_mask=tokens == 0)
        return t, t[:, 0]

    def readout(self, f: torch.Tensor, prompt: torch.Tensor, cam: int) -> torch.Tensor:
        """f (B,N,d) feature map tokens, prompt (B,d) -> (B,K,d) tokens: attended features + soft (x,y) of each question."""
        B, N, d = f.shape
        K = self.cfg.readout_queries
        q = self.query_proj(prompt).view(B, K, d)
        attn = torch.softmax(torch.einsum("bkd,bnd->bkn", q, self.key_proj(f)) / d ** 0.5, dim=-1)
        xy = attn @ self.grid_xy  # (B,K,2) expected image coordinates in [-1,1]
        return attn @ f + self.coord_proj(xy) + self.readout_cam[cam]

    def encode_image(self, img: torch.Tensor, cam: int, prompt: torch.Tensor | None = None) -> torch.Tensor:
        """img: (B,3,H,W) float in [0,1] -> (B, tokens, d). `prompt` (B,d) drives the FiLM layers."""
        x = (img - self.mean) / self.std
        for k, layer in enumerate(self.backbone):
            x = layer(x)
            if self.films is not None and prompt is not None and k in self.film_at:
                g, b = self.films[self.film_at[k]](prompt)[:, :, None, None].chunk(2, dim=1)
                x = x * (1 + g) + b
        f = self.proj(x)  # (B,d,h,w)
        f = f.flatten(2).transpose(1, 2)  # (B,hw,d)
        f = f + self.img_pos[None] + self.cam_emb[cam]
        if self.cfg.readout_queries and prompt is not None:
            f = torch.cat([f, self.readout(f, prompt, cam)], dim=1)
        return f

    def forward(self, wrist: torch.Tensor, external: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        s = self.state_proj(state)[:, None] + self.type_emb[0]
        t, prompt = self.encode_text(tokens)
        t = t + self.type_emb[1]
        v = torch.cat([self.encode_image(wrist, 0, prompt), self.encode_image(external, 1, prompt)], dim=1) + self.type_emb[2]
        x = torch.cat([s, t, v], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), tokens == 0,
                         torch.zeros(B, v.shape[1], dtype=torch.bool, device=x.device)], dim=1)
        mem = self.encoder(x, src_key_padding_mask=pad)
        q = self.queries[None].expand(B, -1, -1)
        out = self.decoder(q, mem, memory_key_padding_mask=pad)
        return self.head(self.out_norm(out))  # (B, chunk, action_dim)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
