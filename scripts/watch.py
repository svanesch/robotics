"""Live 3D view of a running job: pixi run watch [--run NAME]

Mirrors the simulated world of a job that reports through robotics.status (collection, DAgger, evaluation) in a MuJoCo
viewer: every robot cell, the blocks as they are in the current episode, and the cameras drawn as small camera objects.
The job itself is never slowed down; it only publishes a snapshot file ~20 times a second.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from robotics.robots.spec import load_robot
from robotics.sim.builder import build_world
from robotics.status import STATUS_DIR, list_runs

MODEL_ARRAYS = ("geom_size", "geom_rgba", "geom_rbound", "cam_pos", "cam_quat", "light_pos", "light_diffuse")


def pick_run(name: str | None) -> dict:
    runs = [r for r in list_runs() if r.get("has_state") and (name is None or name in r["name"])]
    alive = [r for r in runs if r["alive"]]
    if not runs:
        raise SystemExit("no run with a 3D state found: start `pixi run collect ...` (or evaluate / collect_dagger) first")
    return (alive or runs)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="part of the run name (default: the newest running job)")
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args()
    run = pick_run(a.run)
    meta = run["meta"]
    cfg = load_robot(meta.get("robot", "ur5e"))
    n_cells = int(meta.get("cells", 1))
    model = build_world(cfg, n_cells)
    data = mujoco.MjData(model)
    state_file = STATUS_DIR / f"{run['name']}.state.npz"
    print(f"watching {run['name']} ({cfg.name}, {n_cells} cells); close the window to stop", flush=True)

    centres = np.array([model.body(f"c{i}/table").pos for i in range(n_cells)])
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = centres.mean(0) + np.array([0.25, 0.0, 0.15])
        v.cam.azimuth, v.cam.elevation = 150, -24
        v.cam.distance = 1.9 + 1.1 * float(np.ptp(centres, axis=0).max())
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1  # draw the cameras so you can see where wrist and external views come from
        last = 0.0
        while v.is_running():
            try:
                mtime = state_file.stat().st_mtime
                if mtime != last:
                    snap = np.load(state_file)
                    with v.lock():
                        if snap["qpos"].shape == data.qpos.shape:
                            data.qpos[:] = snap["qpos"]
                        for k in MODEL_ARRAYS:
                            arr = getattr(model, k)
                            if snap[k].shape == arr.shape:
                                arr[:] = snap[k]
                        mujoco.mj_forward(model, data)
                    v.sync()
                    last = mtime
            except (OSError, ValueError, KeyError):
                pass  # the job is rewriting the file, try again on the next tick
            time.sleep(1.0 / a.fps)


if __name__ == "__main__":
    main()
