"""Where does the stacking policy fail? Plays stack scenes and classifies every episode from the block traces.

  pixi run python scripts/diagnose_stack.py --checkpoint runs/ur5e_stack_a/best.pt --episodes 48
"""
import argparse
import collections

import numpy as np
import torch

from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.rollout import STACK_MAX_STEPS, PolicyRunner, TemporalEnsemble
from robotics.policy.train_utils import load_policy, robot_for_checkpoint
from robotics.robots.spec import load_robot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--steps", type=int, default=STACK_MAX_STEPS)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    model, ck = load_policy(a.checkpoint, a.device)
    cfg = load_robot(robot_for_checkpoint(ck))
    runner = PolicyRunner(model, ck, a.device)
    env = MultiCellEnv(cfg, a.cells, model.cfg.image_size)
    rng = np.random.default_rng(a.seed)
    scenes = [sample_scene(cfg, rng, split=a.split, task="stack") for _ in range(a.episodes)]
    classes = collections.Counter()
    for g0 in range(0, len(scenes), a.cells):
        group = scenes[g0:g0 + a.cells]
        n = len(group)
        for i in range(env.n):
            env.reset_cell(i, group[min(i, n - 1)])
        tokens = np.stack([runner.tok.encode(group[min(i, n - 1)].instruction) for i in range(env.n)])
        ens = TemporalEnsemble(env.n, runner.chunk)
        tr = [collections.defaultdict(list) for _ in range(n)]
        for t in range(a.steps):
            imgs = [env.render(i) for i in range(env.n)]
            pred = runner.predict(np.stack([x["wrist"] for x in imgs]), np.stack([x["external"] for x in imgs]),
                                  np.stack([env.state(i) for i in range(env.n)]), tokens)
            act = np.stack([ens.add(i, t, pred[i]) for i in range(env.n)])
            act[:, -1] = np.clip(act[:, -1], 0, 1)
            env.step(act)
            for i in range(n):
                s, cp = group[i], env.cube_pos(i)
                lift = cp[:, 2] - s.cube_half[:, 2]
                g = float(env.state(i)[-1])
                tr[i]["s1"].append(env.stacked(i, 1, 0) and g < 0.3)
                tr[i]["s2"].append(env.stacked(i, 2, 1) and g < 0.3)
                tr[i]["l1"].append(lift[1] > 0.02)
                tr[i]["l2"].append(lift[2] > 0.02)
                tr[i]["g"].append(g)
        for i in range(n):
            s1, s2, l1, l2, g = (np.array(tr[i][k]) for k in ("s1", "s2", "l1", "l2", "g"))
            tower = s1 & s2
            held12 = any(tower[k:k + 12].all() for k in range(len(tower) - 11))
            if held12:
                classes["success: tower of three"] += 1
            elif not (g > 0.5).any():
                classes["gripper never closed"] += 1
            elif not s1.any():
                cp = env.cube_pos(i)
                print(f"  ep{g0 + i}: green-red dxy {np.linalg.norm(cp[1, :2] - cp[0, :2]) * 100:.1f} cm, green height {(cp[1, 2] - group[i].cube_half[1, 2]) * 100:.1f} cm, "
                      f"gripper {g[-1]:.2f}, closed at end {(g > 0.5).mean():.0%} of steps, lifted {l1.sum()} steps", flush=True)
                classes["green lifted, never placed on red" if l1.any() else "green never lifted (grasp failed)"] += 1
            elif s1.any() and not s1[-1] and not l2.any():
                classes["green was on red, then knocked off (before blue)"] += 1
            elif not l2.any():
                classes["green on red; blue never lifted"] += 1
            elif s2.any():
                classes["tower stood, then collapsed"] += 1
            else:
                classes["blue lifted, never placed on green"] += 1
    n = sum(classes.values())
    print(f"{a.checkpoint}: {n} stack episodes")
    for k, v in classes.most_common():
        print(f"  {v:3d} ({100 * v / n:3.0f}%)  {k}")


if __name__ == "__main__":
    main()
