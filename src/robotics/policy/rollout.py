"""Closed-loop evaluation: run a trained policy in the simulator from a text prompt.

The policy only gets both camera images, the joint state and the prompt. Ground truth (block poses) is used solely to
score the episode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from robotics.data.randomize import HELD_OUT, sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.tokenizer import Tokenizer
from robotics.sim.builder import CUBE_COLORS

MAX_STEPS = 560
SUCCESS_HOLD = 12  # consecutive steps the success condition must hold


def others_mask(s) -> np.ndarray:
    m = s.active.copy()
    m[s.target] = False
    return m


def parse_prompt(prompt: str) -> tuple[str, str | None]:
    """Prompt -> (task, target colour) used only to *build the scene* so the requested block exists."""
    words = re.findall(r"[a-z]+", prompt.lower())
    color = next((w for w in words if w in CUBE_COLORS), None)
    task = "place_left" if "left" in words else "place_right" if "right" in words else "lift"
    return task, color


class TemporalEnsemble:
    """ACT-style: every step predicts a chunk; the action for time t averages all chunks covering t (older = heavier)."""

    def __init__(self, n: int, chunk: int, m: float = 0.01, replan: int | None = None):
        """replan=None: exponential ensemble over all overlapping chunks (m = decay). replan=k: execute the first k actions
        of each chunk open-loop, then take a fresh chunk (no averaging)."""
        self.n, self.chunk, self.m, self.replan = n, chunk, m, replan
        self.buf = [[] for _ in range(n)]  # per cell: list of (start_step, chunk)
        self.cur = [None] * n
        self.age = [0] * n

    def reset(self, i: int) -> None:
        self.buf[i], self.cur[i], self.age[i] = [], None, 0

    def add(self, i: int, t: int, pred: np.ndarray) -> np.ndarray:
        if self.replan is not None:
            if self.cur[i] is None or self.age[i] >= self.replan:
                self.cur[i], self.age[i] = pred, 0
            a = self.cur[i][self.age[i]]
            self.age[i] += 1
            return a
        self.buf[i] = [(s, c) for s, c in self.buf[i] if s + self.chunk > t] + [(t, pred)]
        acts = np.stack([c[t - s] for s, c in self.buf[i]])  # oldest first
        w = np.exp(-self.m * np.arange(len(acts)))
        return (acts * w[:, None]).sum(0) / w.sum()


@dataclass
class EpisodeResult:
    task: str
    color: str
    instruction: str
    success: bool
    steps: int
    wrong_block: bool
    held_out: bool
    d_target: float = 9.0  # closest gripper approach [m] to the requested block
    d_other: float = 9.0  # ... to the nearest other block
    grasp_target: bool = False  # closed the gripper (>5 steps) with the requested block between the jaws
    grasp_other: bool = False
    n_blocks: int = 0
    other_lifted: bool = False  # a block that was NOT requested got lifted off the table
    target_lifted: bool = False  # the requested block was lifted at least once
    frames: list = field(default_factory=list)


class PolicyRunner:
    def __init__(self, model, ckpt: dict, device: str):
        self.model, self.device = model, torch.device(device)
        self.stats = {k: np.asarray(v, np.float32) for k, v in ckpt["stats"].items()}
        self.chunk = model.cfg.chunk
        self.tok = Tokenizer(model.cfg.max_text)

    @torch.no_grad()
    def predict(self, wrist: np.ndarray, external: np.ndarray, state: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Batched (N,...) uint8 images, raw state, token ids -> raw action chunks (N, chunk, A)."""
        f = lambda im: torch.from_numpy(im).to(self.device).permute(0, 3, 1, 2).float() / 255.0  # noqa: E731
        s = torch.from_numpy((state - self.stats["state_mean"]) / self.stats["state_std"]).float().to(self.device)
        out = self.model(f(wrist), f(external), s, torch.from_numpy(tokens).to(self.device))
        return out.cpu().numpy() * self.stats["action_std"] + self.stats["action_mean"]


def run_episodes(runner: PolicyRunner, env: MultiCellEnv, scenes: list, max_steps: int = MAX_STEPS,
                 record_cell0=None, ens_m: float = 0.01, replan: int | None = None) -> list[EpisodeResult]:
    """Run len(scenes) episodes, `env.n` at a time in parallel cells. record_cell0(frame_fn) gets called each step for cell 0."""
    results: list[EpisodeResult] = []
    cfg = env.cfg
    for batch_start in range(0, len(scenes), env.n):
        group = scenes[batch_start: batch_start + env.n]
        n = len(group)
        for i, s in enumerate(group):
            env.reset_cell(i, s)
        for i in range(n, env.n):  # unused cells just idle
            env.reset_cell(i, group[0])
        tokens = np.stack([runner.tok.encode(s.instruction) for s in group] + [runner.tok.encode(group[0].instruction)] * (env.n - n))
        ens = TemporalEnsemble(env.n, runner.chunk, ens_m, replan)
        d_t = [9.0] * n
        d_o = [9.0] * n
        closed_t = [0] * n
        closed_o = [0] * n
        hold = [0] * n
        done = [False] * n
        start_pos = [env.cube_pos(i).copy() for i in range(n)]
        wrong = [False] * n
        success_t = [max_steps] * n
        other_lifted = [False] * n
        target_lifted = [False] * n
        for t in range(max_steps):
            imgs = [env.render(i) for i in range(env.n)]
            pred = runner.predict(np.stack([x["wrist"] for x in imgs]), np.stack([x["external"] for x in imgs]),
                                  np.stack([env.state(i) for i in range(env.n)]), tokens)
            actions = np.stack([ens.add(i, t, pred[i]) for i in range(env.n)])
            actions[:, -1] = np.clip(actions[:, -1], 0.0, 1.0)
            if record_cell0 is not None and batch_start == 0:
                record_cell0(imgs[0])
            env.step(actions)
            for i in range(n):
                if done[i]:
                    continue
                s = group[i]
                cp = env.cube_pos(i)
                moved = np.linalg.norm(cp[:, :2] - start_pos[i][:, :2], axis=1) > 0.04
                moved &= s.active
                moved[s.target] = False
                wrong[i] |= bool(moved.any())
                dist = np.linalg.norm(cp - env.ee_pose(i)[:3], axis=1)
                dist[~s.active] = 9.0
                d_t[i] = min(d_t[i], float(dist[s.target]))
                others_d = np.where(others_mask(s), dist, 9.0)
                d_o[i] = min(d_o[i], float(others_d.min()))
                if env.state(i)[-1] > 0.5 and dist.min() < 0.035:  # jaws closing around a block
                    if int(dist.argmin()) == s.target:
                        closed_t[i] += 1
                    else:
                        closed_o[i] += 1
                lift = cp[:, 2] - s.cube_half[:, 2]
                others = s.active.copy()
                others[s.target] = False
                other_lifted[i] |= bool((lift[others] > 0.02).any())
                target_lifted[i] |= bool(lift[s.target] > 0.02)
                if s.task == "lift":
                    ok = env.target_lifted(i) > 0.04
                else:
                    zone = env.zone_center(1 if s.task == "place_left" else -1)
                    ok = (bool(np.all(np.abs(cp[s.target][:2] - zone) < cfg.zone_half)) and env.target_lifted(i) < 0.012
                          and env.state(i)[-1] < 0.3)
                hold[i] = hold[i] + 1 if ok else 0
                if hold[i] >= SUCCESS_HOLD:
                    done[i], success_t[i] = True, t
            if all(done):
                break
        for i, s in enumerate(group):
            results.append(EpisodeResult(s.task, s.cube_colors[s.target], s.instruction, done[i], success_t[i] + 1, wrong[i],
                                         (s.cube_colors[s.target], s.task) in HELD_OUT,
                                         d_t[i], d_o[i], closed_t[i] > 5, closed_o[i] > 5,
                                         int(s.active.sum()), other_lifted[i], target_lifted[i]))
    return results


def scene_for_prompt(cfg, rng: np.random.Generator, prompt: str):
    task, color = parse_prompt(prompt)
    s = sample_scene(cfg, rng, split="val", task=task, target_color=color)
    s.instruction = prompt
    return s
