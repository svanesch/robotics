"""DAgger data collection: a trained policy drives the arm, the scripted expert labels every state it visits.

Per episode the policy executes the first `handover` steps (or the whole episode with probability `policy_prob_full`);
afterwards the expert takes over and recovers/finishes. At EVERY step the label is the expert's action for the state the
arm is actually in (`PickExpert.sync()` re-plans from the measured world state), so the dataset contains the off-nominal
states a policy ends up in, with the correct recovery action. The policy only receives images, joint state and the prompt;
privileged block poses are only used by the expert to produce labels.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from robotics.data.dataset import DatasetWriter
from robotics.data.expert import DONE, PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import CAMERAS, MultiCellEnv
from robotics.policy.rollout import PolicyRunner, TemporalEnsemble
from robotics.policy.train_utils import load_policy, robot_for_checkpoint
from robotics.robots.spec import load_robot
from robotics.status import StatusReporter
from robotics.sim.kinematics import ArmIK

MIN_STEPS = 25


def collect_dagger(checkpoint: str, out: str, episodes: int, cells: int = 8, seed: int = 0, policy_prob_full: float = 0.3,
                   handover_range: tuple[int, int] = (30, 230), max_steps: int = 480, device: str = "cuda",
                   ens_m: float = 0.01, replan: int | None = None, robot: str | None = None) -> dict:
    model, ck = load_policy(checkpoint, device)
    robot = robot_for_checkpoint(ck, robot)  # the robot the policy was trained for
    cfg = load_robot(robot)
    runner = PolicyRunner(model, ck, device)
    size = model.cfg.image_size
    env = MultiCellEnv(cfg, cells, size)
    ik = ArmIK(cfg)
    rng = np.random.default_rng(seed)
    writer = DatasetWriter(out, cfg.name, env.fps, [*cfg.arm_joints, "gripper"], CAMERAS, size)
    ens = TemporalEnsemble(cells, runner.chunk, ens_m, replan)
    experts: list[PickExpert | None] = [None] * cells
    tokens = np.zeros((cells, model.cfg.max_text), np.int64)
    bufs: list[dict] = [{} for _ in range(cells)]
    steps, handover, split = [0] * cells, [0] * cells, ["train"] * cells
    rep = StatusReporter("dagger", f"dagger-{Path(out).name}", total=episodes, meta={"robot": robot, "policy": str(checkpoint), "cells": cells})
    stats = {"episodes": 0, "frames": 0, "policy_frames": 0, "cut_unconfident": 0, "expert_success": 0}

    def start(i: int) -> None:
        split[i] = "val" if rng.random() < 0.1 else "train"
        env.reset_cell(i, sample_scene(cfg, rng, split=split[i]))
        experts[i] = PickExpert(env, ik, i)
        ens.reset(i)
        tokens[i] = runner.tok.encode(env.cells[i]["scene"].instruction)
        bufs[i] = {k: [] for k in ("state", "action", "ee_pose", "env_state", *CAMERAS)}
        steps[i] = 0
        handover[i] = max_steps + 1 if rng.random() < policy_prob_full else int(rng.integers(*handover_range))

    def finish(i: int) -> None:
        if steps[i] >= MIN_STEPS and len(bufs[i]["state"]) >= MIN_STEPS:
            s = experts[i].s
            ep = {k: np.asarray(v) for k, v in bufs[i].items()}
            writer.add_episode(ep, s.instruction, {
                "success": bool(experts[i].success), "robot": cfg.name, "task": s.task, "split": split[i], "noise": False,
                "target_color": s.cube_colors[s.target], "n_blocks": int(s.active.sum()), "source": "dagger",
                "policy_steps": int(min(handover[i], steps[i]))})
            stats["episodes"] += 1
            stats["frames"] += len(ep["state"])
            stats["policy_frames"] += int(min(handover[i], steps[i]))
            stats["expert_success"] += int(experts[i].success)
            rep.update(progress=stats["episodes"], frames=stats["frames"], policy_frames=stats["policy_frames"],
                       expert_success=stats["expert_success"], cut_no_label=stats["cut_unconfident"])
        if stats["episodes"] < episodes:
            start(i)
        else:
            experts[i] = None

    for i in range(cells):
        start(i)
    t0 = time.time()
    while any(e is not None for e in experts):
        live = [i for i in range(cells) if experts[i] is not None]
        imgs = {i: env.render(i) for i in live}
        for i in range(cells):  # idle cells just need some input for the batched call
            imgs.setdefault(i, imgs[live[0]])
        pred = runner.predict(np.stack([imgs[i]["wrist"] for i in range(cells)]), np.stack([imgs[i]["external"] for i in range(cells)]),
                              np.stack([env.state(i) for i in range(cells)]), tokens)
        rep.preview(np.concatenate([imgs[live[0]]["wrist"], imgs[live[0]]["external"]], 1))
        actions = np.tile(np.append(env.home, 0.0), (cells, 1))
        finished = []
        for i in live:
            e = experts[i]
            pol = ens.add(i, steps[i], pred[i])
            pol[-1] = np.clip(pol[-1], 0, 1)
            e.sync()
            if not e.confident:  # toppled / out of reach: the scripted expert has no valid label here, end the episode
                stats["cut_unconfident"] += 1
                finished.append(i)
                continue
            lab, done = e.act()
            b = bufs[i]
            b["state"].append(env.state(i)); b["ee_pose"].append(env.ee_pose(i)); b["env_state"].append(env.cube_pos(i).reshape(-1))
            b["wrist"].append(imgs[i]["wrist"]); b["external"].append(imgs[i]["external"]); b["action"].append(lab)
            actions[i] = pol if steps[i] < handover[i] else lab
            steps[i] += 1
            if done or steps[i] >= max_steps:
                finished.append(i)
        env.step(actions)
        rep.snapshot(env.model, env.data)
        for i in finished:
            finish(i)
        if stats["episodes"] and stats["episodes"] % 20 == 0:
            pass
    writer.finalize()
    rep.finish("done")
    stats["seconds"] = time.time() - t0
    return stats
