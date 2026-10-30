"""Does the policy follow the prompt? Matched-scene test.

For each random scene with >=2 reachable blocks, the SAME scene (same layout, same images) is run once per block with the prompt
"pick up the <colour> block" naming that block. Only colours that are actually in the scene are asked for.

Reported:
  * open-loop shift: how much the first predicted action chunk changes when only the prompt changes (identical images)
  * follow rate: closest-approach block == requested block (chance = 1 / number of reachable blocks)
  * controllable: both prompts of a scene sent the gripper to their own block

  pixi run python scripts/language_probe.py --checkpoint runs/r4/best.pt --scenes 40
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.rollout import PolicyRunner, run_episodes
from robotics.policy.train_utils import load_policy
from robotics.robots.spec import load_robot


def reachable(cfg, s, k) -> bool:
    x, y = s.cube_xy[k]
    return cfg.workspace_x[0] <= x <= cfg.workspace_x[1] and cfg.workspace_y[0] <= y <= cfg.workspace_y[1]


def variants(cfg, s):
    out = []
    for k in np.where(s.active)[0]:
        if reachable(cfg, s, k):
            out.append(dataclasses.replace(s, target=int(k), task="lift", instruction=f"pick up the {s.cube_colors[k]} block"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--robot", default="so101")
    p.add_argument("--scenes", type=int, default=40)
    p.add_argument("--cells", type=int, default=6)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--ens-m", type=float, default=0.01)
    p.add_argument("--replan", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = load_robot(a.robot)
    model, ck = load_policy(a.checkpoint, a.device)
    runner = PolicyRunner(model, ck, a.device)
    env = MultiCellEnv(cfg, a.cells, model.cfg.image_size)
    rng = np.random.default_rng(a.seed)
    groups = []
    while len(groups) < a.scenes:
        s = sample_scene(cfg, rng, split="val", task="lift")
        v = variants(cfg, s)
        if len(v) >= 2:
            groups.append(v[:3])

    # 1. open-loop: identical images, different prompts
    shifts = []
    for g in groups:
        env.reset_cell(0, g[0])
        im = env.render(0)
        preds = []
        for v in g:
            tok = np.stack([runner.tok.encode(v.instruction)])
            preds.append(runner.predict(im["wrist"][None], im["external"][None], env.state(0)[None], tok)[0])
        shifts.append(np.mean([np.abs(preds[i] - preds[j])[:, :5].mean() for i in range(len(g)) for j in range(i + 1, len(g))]))

    # 2. closed loop: does the arm go to the block that was asked for?
    flat = [v for g in groups for v in g]
    res = run_episodes(runner, env, flat, ens_m=a.ens_m, replan=a.replan)
    follow, grasp, success, controllable, chance = [], [], [], [], []
    pos = 0
    for g in groups:
        r = res[pos:pos + len(g)]
        pos += len(g)
        ok = [(x.d_target < x.d_other) and x.d_target < 0.04 for x in r]
        follow += ok
        grasp += [x.grasp_target for x in r]
        success += [x.success for x in r]
        controllable.append(all(ok))
        chance += [1 / len(g)] * len(g)
    n = len(follow)
    out = {"scenes": len(groups), "episodes": n,
           "open_loop_action_shift_rad": float(np.mean(shifts)),
           "follow_requested_block": float(np.mean(follow)), "chance_follow": float(np.mean(chance)),
           "grasp_requested_block": float(np.mean(grasp)), "lift_success": float(np.mean(success)),
           "scenes_controllable": float(np.mean(controllable))}
    print(json.dumps(out, indent=2))
    Path(a.checkpoint).parent.joinpath("language_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
