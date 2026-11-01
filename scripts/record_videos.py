"""Record test videos of a trained policy for the dashboard (they play in a loop on the run's page).

  pixi run python scripts/record_videos.py --run d2 --episodes 8

The scenes are the FIRST N scenes of the evaluation (same seed as scripts/evaluate.py), not hand-picked, so successes and
failures show up in the proportion the policy really has. Each video shows the 3D world (left) and the two camera views the
policy sees (right: wrist above, external below). Output: runs/<run>/videos/ep00_ok.mp4 ... plus index.json.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.rollout import PolicyRunner, run_episodes
from robotics.policy.train_utils import load_policy, robot_for_checkpoint
from robotics.robots.spec import load_robot


def free_camera(cfg) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if cfg.name == "ur5e":
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.45, 0.0, 0.10], 1.15, 90, -38  # side view over the table; azimuth 145 looks through the cell's back wall
    else:
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [cfg.workspace_x[0] + 0.1, 0.0, 0.08], 0.95, 140, -22
    return cam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None, help="default: runs/<run>/best.pt")
    p.add_argument("--robot", default=None, help="default: inferred from the dataset the run was trained on")
    p.add_argument("--split", default="test")
    p.add_argument("--task", default=None, help="e.g. stack")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    run = Path("runs") / a.run
    model, ck = load_policy(a.checkpoint or run / "best.pt", a.device)
    cfg = load_robot(robot_for_checkpoint(ck, a.robot))
    runner = PolicyRunner(model, ck, a.device)
    env = MultiCellEnv(cfg, 1, model.cfg.image_size)
    rng = np.random.default_rng(a.seed)
    scenes = [sample_scene(cfg, rng, split=a.split, task=a.task) for _ in range(a.episodes)]
    big = mujoco.Renderer(env.model, 512, 512)
    cam = free_camera(cfg)
    up = lambda img: np.kron(img, np.ones((2, 2, 1), np.uint8))  # noqa: E731
    out = run / "videos"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.mp4"):
        old.unlink()
    index = []
    for k, scene in enumerate(scenes):
        frames = []

        def rec(im):
            big.update_scene(env.data, cam)
            frames.append(np.concatenate([big.render(), np.concatenate([up(im["wrist"]), up(im["external"])], 0)], 1))

        r = run_episodes(runner, env, [scene], record_cell0=rec)[0]
        frames += [frames[-1]] * 12  # hold the last frame a moment before the loop restarts
        name = f"ep{k:02d}_{'ok' if r.success else 'fail'}.mp4"
        imageio.mimsave(out / name, frames[::2], fps=15, codec="libx264", macro_block_size=1)
        index.append({"file": name, "success": bool(r.success), "steps": int(r.steps), "task": scene.task, "instruction": scene.instruction,
                      "color": r.color, "split": a.split, "seed": a.seed})
        print(f"{name}: {'SUCCESS' if r.success else 'fail'} after {r.steps} steps  ({scene.instruction})", flush=True)
    (out / "index.json").write_text(json.dumps({"checkpoint_step": ck["step"], "robot": cfg.name, "episodes": index}, indent=1))
    print(f"{sum(e['success'] for e in index)}/{len(index)} successful; wrote {out}")


if __name__ == "__main__":
    main()
