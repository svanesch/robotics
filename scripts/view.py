"""Interactive viewer with the scripted expert running: pixi run view --robot so101"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

from robotics.data.expert import PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.robots.spec import load_robot
from robotics.sim.kinematics import ArmIK

p = argparse.ArgumentParser()
p.add_argument("--robot", default="so101")
p.add_argument("--cells", type=int, default=2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--shadows", action=argparse.BooleanOptionalAction, default=False, help="render shadows (off by default; slow with many cells)")
a = p.parse_args()
cfg = load_robot(a.robot)
env = MultiCellEnv(cfg, a.cells, render=False, shadows=a.shadows)
ik, rng = ArmIK(cfg), np.random.default_rng(a.seed)
experts = []
for i in range(a.cells):
    env.reset_cell(i, sample_scene(cfg, rng))
    experts.append(PickExpert(env, ik, i))
with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as v:
    # polished presentation: framed 3/4 view, shadows + reflections on, debug overlays off
    v.cam.lookat[:] = [cfg.workspace_x[0] + 0.1, 0, 0.08]
    v.cam.azimuth, v.cam.elevation = 140, -22
    v.cam.distance = 0.95 if cfg.name == "so101" else 1.7
    with v.lock():
        for flag in (mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE, mujoco.mjtVisFlag.mjVIS_JOINT):
            v.opt.flags[flag] = 0
        v.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        v.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    while v.is_running():
        acts = []
        for i, e in enumerate(experts):
            act, done = e.act()
            acts.append(act)
            if done:
                env.reset_cell(i, sample_scene(cfg, rng))
                experts[i] = PickExpert(env, ik, i)
        env.step(np.stack(acts))
        v.sync()
