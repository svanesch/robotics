"""Collect scripted-expert demonstrations in N parallel cells; optionally across worker processes."""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from robotics.data.dataset import DatasetWriter, merge_shards
from robotics.data.expert import APPROACH, LIFT, PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import CAMERAS, MultiCellEnv
from robotics.robots.spec import load_robot
from robotics.sim.kinematics import ArmIK
from robotics.status import StatusReporter

MAX_STEPS = 560
SPLITS = ("train", "val", "test")
SPLIT_P = (0.8, 0.1, 0.1)
NOISE_START_P = 0.03  # per step, chance of starting a perturbation burst
NOISE_STEPS = (8, 20)
NOISE_STD = 0.06  # rad, per joint


def collect_shard(robot: str, out: str, n_episodes: int, cells: int, image_size: int, seed: int,
                  keep_failures: bool = False, shadows: bool = False, noise: float = 0.0,
                  task: str | None = None, drop: float = 0.0) -> dict:
    """noise: fraction of episodes that get random joint perturbations while executing; labels stay the clean expert action.
    drop: fraction of episodes in which the gripper is forced open for a few steps during the lift (the block falls); the expert
    then re-plans from the real state (PickExpert.sync) and picks it up again, so the data contains failed grasps and the re-pick."""
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
    drop_ep, dropped, drop_left = [False] * cells, [False] * cells, [0] * cells
    rep = StatusReporter("collect", f"collect-{Path(out).name}", total=n_episodes,
                         meta={"robot": robot, "cells": cells, "image": image_size, "noise": noise, "task": task or "mixed"})

    def start(i: int) -> None:
        splits[i] = str(rng.choice(SPLITS, p=SPLIT_P))
        env.reset_cell(i, sample_scene(cfg, rng, split=splits[i], task=task))
        noisy[i], burst[i] = bool(rng.random() < noise), 0
        drop_ep[i], dropped[i], drop_left[i] = bool(rng.random() < drop), False, 0
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
            e = experts[i]
            if drop_ep[i] and not dropped[i] and e.phase == LIFT and env.target_lifted(i) > 0.01 and rng.random() < 0.5:
                dropped[i], drop_left[i] = True, int(rng.integers(8, 16))  # open the gripper: the block falls
            if dropped[i]:
                e.sync()  # label what the expert would do in the state the arm is really in
                if not e.confident:  # block toppled or out of reach: the scripted expert cannot label this
                    finished.append((i, False))
                    actions[i] = env.state(i)
                    continue
            b = bufs[i]
            b["state"].append(env.state(i))
            b["ee_pose"].append(env.ee_pose(i))
            b["env_state"].append(env.cube_pos(i).reshape(-1))
            imgs = env.render(i)
            for cam, img in imgs.items():
                b[cam].append(img)
            if i == 0:
                rep.preview(np.concatenate([imgs["wrist"], imgs["external"]], 1))
            a, done = e.act()
            if done and dropped[i] and env.target_lifted(i) < 0.03:  # the lift 'finished' while the block lies on the table: go again
                done, e.phase, e.t_phase, e.grip = False, APPROACH, 0, 0.0
            b["action"].append(a)
            actions[i] = a
            steps[i] += 1
            if done or steps[i] >= MAX_STEPS:
                finished.append((i, done and experts[i].success))
        executed = actions.copy()
        for i in range(cells):  # perturb what is executed; the recorded label stays the clean expert action
            if drop_left[i] > 0:
                executed[i, -1] = 0.0
                drop_left[i] -= 1
            if not noisy[i]:
                continue
            if burst[i] == 0 and rng.random() < NOISE_START_P and experts[i].phase < 8:
                burst[i] = int(rng.integers(*NOISE_STEPS))
                offset[i] = rng.normal(0, NOISE_STD, cfg.n_arm)
            if burst[i] > 0:
                executed[i, :-1] += offset[i]
                burst[i] -= 1
        env.step(executed)
        rep.snapshot(env.model, env.data)
        for i, ok in finished:
            attempted += 1
            if (ok or keep_failures) and saved < n_episodes:
                ep = {k: np.asarray(v) for k, v in bufs[i].items()}
                s = experts[i].s
                writer.add_episode(ep, s.instruction, {
                    "success": bool(ok), "robot": cfg.name, "task": s.task, "split": splits[i], "noise": noisy[i],
                    "target_color": s.cube_colors[s.target], "n_blocks": int(s.active.sum()), "dropped": dropped[i]})
                saved += 1
            rep.update(progress=saved, attempted=attempted, success_rate=saved / max(attempted, 1), frames=writer.total_frames,
                       episodes_per_min=60 * saved / max(time.time() - t0, 1e-6))
            start(i)
    writer.finalize()
    rep.finish("done", frames=writer.total_frames)
    return {"saved": saved, "attempted": attempted, "seconds": time.time() - t0, "frames": writer.total_frames}


def _worker(args):
    return collect_shard(*args)


def collect(robot: str, out: str, episodes: int, cells: int = 4, workers: int = 1, image_size: int = 128,
            seed: int = 0, keep_failures: bool = False, shadows: bool = False, noise: float = 0.0,
            task: str | None = None, drop: float = 0.0) -> dict:
    out = Path(out)
    if workers <= 1:
        return collect_shard(robot, str(out), episodes, cells, image_size, seed, keep_failures, shadows, noise, task, drop)
    per = [episodes // workers + (1 if k < episodes % workers else 0) for k in range(workers)]
    shards = [out.parent / f".{out.name}_shard{k}" for k in range(workers)]
    jobs = [(robot, str(s), n, cells, image_size, seed + 1000 * k, keep_failures, shadows, noise, task, drop) for k, (s, n) in enumerate(zip(shards, per)) if n]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    merge_shards([s for s, n in zip(shards, per) if n], out)
    return {k: sum(r[k] for r in results) for k in ("saved", "attempted", "frames")} | {"seconds": max(r["seconds"] for r in results)}
