"""Render a side-by-side demo video (free camera + wrist + external) of the expert: pixi run python scripts/demo_video.py --robot so101"""
import argparse

import imageio
import mujoco
import numpy as np

from robotics.data.expert import PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.robots.spec import load_robot
from robotics.sim.kinematics import ArmIK

p = argparse.ArgumentParser()
p.add_argument("--robot", default="so101")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--out", default=None)
a = p.parse_args()
cfg = load_robot(a.robot)
env = MultiCellEnv(cfg, 1, 256)
ik, rng = ArmIK(cfg), np.random.default_rng(a.seed)
big = mujoco.Renderer(env.model, 512, 512)
cam = mujoco.MjvCamera()
cam.type, cam.lookat, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, [cfg.workspace_x[0] + 0.1, 0, 0.1], 135, -30
cam.distance = 0.9 if cfg.name == "so101" else 1.6
frames = []
for ep in range(a.episodes):
    s = sample_scene(cfg, rng)
    env.reset_cell(0, s)
    e = PickExpert(env, ik, 0)
    for t in range(300):
        act, done = e.act()
        env.step(act[None])
        if t % 2 == 0:
            big.update_scene(env.data, cam)
            im = env.render(0)
            small = np.concatenate([im["wrist"], im["external"]], 0)
            frames.append(np.concatenate([big.render(), small], 1))
        if done:
            break
    print(f"episode {ep}: '{s.instruction}' success={e.success}")
out = a.out or f"demo/{a.robot}.mp4"
imageio.mimsave(out, frames, fps=15, codec="libx264", macro_block_size=1)
print("wrote", out)
