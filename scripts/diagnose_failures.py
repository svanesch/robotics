"""What goes wrong after the gripper closes? Plays test scenes and classifies the failures from the traces.

  pixi run python scripts/diagnose_failures.py --checkpoint runs/ur5e_d1/best.pt --episodes 48
"""
import argparse
import collections

import numpy as np
import torch

from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.rollout import PolicyRunner, TemporalEnsemble
from robotics.policy.train_utils import load_policy, robot_for_checkpoint
from robotics.robots.spec import load_robot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--cells", type=int, default=6)
    p.add_argument("--steps", type=int, default=420)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    model, ck = load_policy(a.checkpoint, a.device)
    cfg = load_robot(robot_for_checkpoint(ck))
    runner = PolicyRunner(model, ck, a.device)
    env = MultiCellEnv(cfg, a.cells, model.cfg.image_size)
    rng = np.random.default_rng(a.seed)
    scenes = [sample_scene(cfg, rng, split="test") for _ in range(a.episodes)]
    classes, closes_to_lift = collections.Counter(), []
    for g0 in range(0, len(scenes), a.cells):
        group = scenes[g0:g0 + a.cells]
        for i, s in enumerate(group):
            env.reset_cell(i, s)
        tokens = np.stack([runner.tok.encode(s.instruction) for s in group])
        ens = TemporalEnsemble(len(group), runner.chunk)
        tr = [{"ez": [], "g": [], "bz": []} for _ in group]
        for t in range(a.steps):
            imgs = [env.render(i) for i in range(len(group))]
            pred = runner.predict(np.stack([x["wrist"] for x in imgs]), np.stack([x["external"] for x in imgs]),
                                  np.stack([env.state(i) for i in range(len(group))]), tokens)
            act = np.stack([ens.add(i, t, pred[i]) for i in range(len(group))])
            act[:, -1] = np.clip(act[:, -1], 0, 1)
            env.step(act)
            for i, s in enumerate(group):
                tr[i]["ez"].append(float(env.ee_pose(i)[2]))
                tr[i]["g"].append(float(env.state(i)[-1]))
                tr[i]["bz"].append(float(env.target_lifted(i)))
        for i, s in enumerate(group):
            ez, gr, bz = map(np.array, (tr[i]["ez"], tr[i]["g"], tr[i]["bz"]))
            lifted = bz > 0.04
            held12 = any(lifted[k:k + 12].all() for k in range(len(lifted) - 11))
            closed = np.where(gr > 0.5)[0]
            tc = closed[0] if len(closed) else None
            if held12:
                classes["success"] += 1
                if tc is not None and lifted.any():
                    closes_to_lift.append(int(np.argmax(lifted)) - int(tc))
            elif tc is None:
                classes["never closed the gripper"] += 1
            else:
                after = slice(tc, None)
                rise, lift = ez[after].max() - ez[tc], bz[after].max()
                reopen = np.where(gr[tc + 5:] < 0.3)[0]
                if lift > 0.02:
                    classes["lifted the block, then lost it"] += 1
                elif rise < 0.03:
                    classes["closed, but the arm never rose (policy stalls)"] += 1
                elif len(reopen) and reopen[0] < 60:
                    classes["gripper reopened soon after closing"] += 1
                else:
                    classes["arm rose, block stayed (grasp missed or slipped)"] += 1
    n = sum(classes.values())
    print(f"{a.checkpoint}: {n} test episodes")
    for k, v in classes.most_common():
        print(f"  {v:3d} ({100 * v / n:3.0f}%)  {k}")
    if closes_to_lift:
        print(f"  successful episodes: block in the air a median of {np.median(closes_to_lift):.0f} steps after the gripper closed")


if __name__ == "__main__":
    main()
