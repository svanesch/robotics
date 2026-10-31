"""Evaluate a trained VLA on val/test: offline action error + closed-loop success in the simulator.

  pixi run evaluate --checkpoint runs/<run>/best.pt --split test --episodes 60
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.data import VLADataset
from robotics.policy.rollout import PolicyRunner, run_episodes
from robotics.policy.train_utils import load_policy, robot_for_checkpoint, to_device
from robotics.robots.spec import load_robot
from robotics.status import StatusReporter


@torch.no_grad()
def offline(runner, ck, dataset, split, n_frames, seed):
    ds = VLADataset(dataset, split, runner.chunk, stats={k: np.asarray(v) for k, v in ck["stats"].items()})
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n_frames, len(ds)), replace=False)
    per_task = collections.defaultdict(list)
    for s in range(0, len(idx), 64):
        batch = torch.utils.data.default_collate([ds[int(i)] for i in idx[s:s + 64]])
        x = to_device(batch, runner.device)
        pred = runner.model(x["wrist"], x["external"], x["state"], x["tokens"])
        valid = (~x["is_pad"])[..., None].float()
        err = ((pred - x["actions"]).abs() * valid).sum(1) / valid.sum(1) * torch.from_numpy(ds.stats["action_std"]).to(runner.device)
        for e, i in zip(err.cpu().numpy(), idx[s:s + 64]):
            per_task[ds.meta[ds.frame_ep[i]]["task"]].append(e)
    out = {t: np.mean(v, 0).round(4).tolist() for t, v in per_task.items()}
    out["all"] = np.mean([e for v in per_task.values() for e in v], 0).round(4).tolist()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default=None, help="default: the dataset the checkpoint was trained on")
    p.add_argument("--robot", default=None, help="default: inferred from the checkpoint")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--cells", type=int, default=6)
    p.add_argument("--offline-frames", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", default="", help="suffix for the output json (eval_<split><tag>.json)")
    p.add_argument("--ens-m", type=float, default=0.01, help="temporal-ensemble decay")
    p.add_argument("--replan", type=int, default=None, help="open-loop steps per chunk (disables ensembling)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    model, ck = load_policy(a.checkpoint, a.device)
    runner = PolicyRunner(model, ck, a.device)
    dataset = a.dataset or ck["train_cfg"]["dataset"]
    dataset = dataset[0] if isinstance(dataset, list) else dataset  # offline error: the expert-only test split
    print(f"checkpoint step {ck['step']}, best val l1 {ck.get('best_val')}")
    off = offline(runner, ck, dataset, a.split, a.offline_frames, a.seed)
    print(f"offline mean abs action error on '{a.split}' (rad; last = gripper in [0,1]):")
    for t, v in off.items():
        print(f"  {t:12s} {v}")
    cfg = load_robot(robot_for_checkpoint(ck, a.robot))
    rng = np.random.default_rng(a.seed)
    env = MultiCellEnv(cfg, a.cells, model.cfg.image_size)
    scenes = [sample_scene(cfg, rng, split=a.split) for _ in range(a.episodes)]
    rep = StatusReporter("eval", f"eval-{Path(a.checkpoint).parent.name}-{a.split}{a.tag}", total=len(scenes),
                         meta={"robot": cfg.name, "checkpoint": a.checkpoint, "split": a.split, "cells": a.cells})
    res = run_episodes(runner, env, scenes, ens_m=a.ens_m, replan=a.replan, status=rep)
    rep.finish("done", success_rate=float(np.mean([r.success for r in res])))
    rows = collections.defaultdict(lambda: [0, 0])
    for r in res:
        for key in ("all", r.task, "held-out combos" if r.held_out else "seen combos"):
            rows[key][0] += r.success; rows[key][1] += 1
    for r in res:
        rows[f"{r.n_blocks} blocks"][0] += r.success; rows[f"{r.n_blocks} blocks"][1] += 1
    wrong = np.mean([r.wrong_block for r in res])
    fails = [r for r in res if not r.success]
    why = {"lifted a wrong block": sum(r.other_lifted for r in fails),
           "never lifted the target (and no wrong block)": sum((not r.target_lifted) and (not r.other_lifted) for r in fails),
           "lifted the target but did not finish the task": sum(r.target_lifted and not r.other_lifted for r in fails)}
    print("failure breakdown (", len(fails), "failed ):", why)
    print(f"closed-loop on '{a.split}' scenes ({len(res)} episodes):")
    for k, (s, n) in sorted(rows.items()):
        print(f"  {k:16s} {s}/{n} = {s / n:.0%}")
    print(f"  touched a wrong block in {wrong:.0%} of episodes; mean steps to success {np.mean([r.steps for r in res if r.success] or [0]):.0f}")
    n = len(res)
    beh = {"median_closest_cm_requested": float(np.median([r.d_target for r in res]) * 100),
           "median_closest_cm_other": float(np.median([r.d_other for r in res]) * 100),
           "within_4cm_requested": sum(r.d_target < 0.04 for r in res), "grasp_requested": sum(r.grasp_target for r in res),
           "grasp_other": sum(r.grasp_other for r in res), "grasp_neither": sum(not r.grasp_target and not r.grasp_other for r in res),
           "lifted_requested": sum(r.target_lifted for r in res), "lifted_other": sum(r.other_lifted for r in res), "episodes": n}
    print(f"behaviour: closest approach median {beh['median_closest_cm_requested']:.1f} cm (requested) vs {beh['median_closest_cm_other']:.1f} cm (nearest other); "
          f"within 4 cm of requested {beh['within_4cm_requested']}/{n}; grasped requested {beh['grasp_requested']}, other {beh['grasp_other']}, neither {beh['grasp_neither']}; "
          f"lifted requested {beh['lifted_requested']}, other {beh['lifted_other']}")
    out = Path(a.checkpoint).parent / f"eval_{a.split}{a.tag}.json"
    out.write_text(json.dumps({"behaviour": beh, "settings": {"ens_m": a.ens_m, "replan": a.replan, "seed": a.seed}, "failure_breakdown": why, "offline": off, "closed_loop": {k: v for k, v in rows.items()}, "wrong_block_rate": wrong,
                               "step": ck["step"], "episodes": len(res)}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
