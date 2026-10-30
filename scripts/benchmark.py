"""Throughput of the batched sim vs number of cells: pixi run benchmark --robot so101"""
import argparse
import time

import numpy as np

from robotics.envs.multi_cell import MultiCellEnv
from robotics.data.randomize import sample_scene
from robotics.robots.spec import load_robot

p = argparse.ArgumentParser()
p.add_argument("--robot", default="so101")
p.add_argument("--image-size", type=int, default=128)
p.add_argument("--shadows", action="store_true")
a = p.parse_args()
cfg = load_robot(a.robot)
print("cells | sim steps/s (all cells) | steps+render frames/s")
for n in (1, 2, 4, 8, 16):
    env = MultiCellEnv(cfg, n, a.image_size, shadows=a.shadows)
    rng = np.random.default_rng(0)
    for i in range(n):
        env.reset_cell(i, sample_scene(cfg, rng))
    act = np.tile(np.append(env.home, 0.0), (n, 1))
    t = time.time()
    for _ in range(50):
        env.step(act)
    sim = 50 * n / (time.time() - t)
    t = time.time()
    for _ in range(20):
        env.step(act)
        for i in range(n):
            env.render(i)
    ren = 20 * n / (time.time() - t)
    print(f"{n:5d} | {sim:10.0f} | {ren:10.0f}")
