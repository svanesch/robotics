"""Collect scripted-expert demonstrations in N parallel cells; optionally across worker processes."""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from robotics.data.dataset import DatasetWriter, merge_shards
from robotics.data.expert import PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import CAMERAS, MultiCellEnv
from robotics.robots.spec import load_robot
from robotics.sim.kinematics import ArmIK

MAX_STEPS = 560
SPLITS = ("train", "val", "test")
SPLIT_P = (0.8, 0.1, 0.1)
NOISE_START_P = 0.03  # per step, chance of starting a perturbation burst
NOISE_STEPS = (8, 20)
NOISE_STD = 0.06  # rad, per joint


def collect_shard(robot: str, out: str, n_episodes: int, cells: int, image_size: int, seed: int,
                  keep_failures: bool = False, shadows: bool = False, noise: float = 0.0,
                  task: str | None = None) -> dict:
    """noise: fraction of episodes that get random joint perturbations while executing; labels stay the clean expert action."""
    cfg = load_robot(robot)
    env = MultiCellEnv(cfg, cells, image_size, shadows=shadows)
    ik = ArmIK(cfg)
    rng = np.random.default_rng(seed)
    state_names = [*cfg.arm_joints, "gripper"]
    writer = DatasetWriter(out, cfg.name, env.fps, state_names, CAMERAS, image_size)
    experts: list[PickExpert | None] = [None] * cells
    bufs: list[dict] = [{} for _ in range(cells)]
    steps = [0] * cells
    noisy = [False] * cells  # this episode gets perturbations
    burst = [0] * cells  # remaining steps of the current perturbation
    offset = [np.zeros(cfg.n_arm) for _ in range(cells)]
    splits = ["train"] * cells

    def start(i: int) -> None:
        splits[i] = str(rng.choice(SPLITS, p=SPLIT_P))
        env.reset_cell(i, sample_scene(cfg, rng, split=splits[i], task=task))
        noisy[i], burst[i] = bool(rng.random() < noise), 0
        experts[i] = PickExpert(env, ik, i)
        bufs[i] = {k: [] for k in ("state", "action", "ee_pose", "env_state", *CAMERAS)}
        steps[i] = 0

    for i in range(cells):
        start(i)
    saved = attempted = 0
    t0 = time.time()
    while saved < n_episodes:
        actions = np.zeros((cells, cfg.action_dim))
        finished = []
        for i in range(cells):
            b = bufs[i]
            b["state"].append(env.state(i))
            b["ee_pose"].append(env.ee_pose(i))
            b["env_state"].append(env.cube_pos(i).reshape(-1))
            for cam, img in env.render(i).items():
                b[cam].append(img)
            a, done = experts[i].act()
            b["action"].append(a)
            actions[i] = a
            steps[i] += 1
            if done or steps[i] >= MAX_STEPS:
                finished.append((i, done and experts[i].success))
        executed = actions.copy()
        for i in range(cells):  # perturb what is executed; the recorded label stays the clean expert action
            if not noisy[i]:
                continue
            if burst[i] == 0 and rng.random() < NOISE_START_P and experts[i].phase < 8:
                burst[i] = int(rng.integers(*NOISE_STEPS))
                offset[i] = rng.normal(0, NOISE_STD, cfg.n_arm)
            if burst[i] > 0:
                executed[i, :-1] += offset[i]
                burst[i] -= 1
        env.step(executed)
        for i, ok in finished:
            attempted += 1
            if (ok or keep_failures) and saved < n_episodes:
                ep = {k: np.asarray(v) for k, v in bufs[i].items()}
                s = experts[i].s
                writer.add_episode(ep, s.instruction, {
                    "success": bool(ok), "robot": cfg.name, "task": s.task, "split": splits[i], "noise": noisy[i],
                    "target_color": s.cube_colors[s.target], "n_blocks": int(s.active.sum())})
                saved += 1
            start(i)
    writer.finalize()
    return {"saved": saved, "attempted": attempted, "seconds": time.time() - t0, "frames": writer.total_frames}


def _worker(args):
    return collect_shard(*args)


def collect(robot: str, out: str, episodes: int, cells: int = 4, workers: int = 1, image_size: int = 128,
            seed: int = 0, keep_failures: bool = False, shadows: bool = False, noise: float = 0.0,
            task: str | None = None) -> dict:
    out = Path(out)
    if workers <= 1:
        return collect_shard(robot, str(out), episodes, cells, image_size, seed, keep_failures, shadows, noise, task)
    per = [episodes // workers + (1 if k < episodes % workers else 0) for k in range(workers)]
    shards = [out.parent / f".{out.name}_shard{k}" for k in range(workers)]
    jobs = [(robot, str(s), n, cells, image_size, seed + 1000 * k, keep_failures, shadows, noise, task) for k, (s, n) in enumerate(zip(shards, per)) if n]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    merge_shards([s for s, n in zip(shards, per) if n], out)
    return {k: sum(r[k] for r in results) for k in ("saved", "attempted", "frames")} | {"seconds": max(r["seconds"] for r in results)}
