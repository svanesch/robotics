"""Check what the two cameras actually see along expert trajectories: pixi run python scripts/audit_cameras.py

External cam: are all blocks, the gripper tip, the arm base and the elbow inside the frame?
Wrist cam: is the target block visible (in frame and not occluded) while approaching/grasping, and are the jaws in view?
Writes a contact sheet of the worst frames to demo/camera_audit.png.
"""
import argparse

import imageio
import mujoco
import numpy as np

from robotics.data.expert import APPROACH, DESCEND, DONE, GRASP, PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.robots.spec import load_robot
from robotics.sim.kinematics import ArmIK

MARGIN = 0.04  # fraction of the image side that counts as "too close to the edge"


def project(model, data, cam_id, points, size):
    """Pixel coordinates (u, v) and depth of world points for a MuJoCo camera."""
    pos, mat = data.cam_xpos[cam_id], data.cam_xmat[cam_id].reshape(3, 3)
    f = 0.5 * size / np.tan(np.radians(model.cam_fovy[cam_id]) / 2)
    out = []
    for p in points:
        x, y, z = mat.T @ (np.asarray(p) - pos)  # camera looks along -z, image y points up
        depth = -z
        out.append((size / 2 + f * x / depth, size / 2 - f * y / depth, depth) if depth > 1e-3 else (np.nan, np.nan, depth))
    return np.array(out)


def inside(uv, size, margin=MARGIN):
    lo, hi = margin * size, (1 - margin) * size
    return (uv[:, 0] > lo) & (uv[:, 0] < hi) & (uv[:, 1] > lo) & (uv[:, 1] < hi) & (uv[:, 2] > 0)


def visible(model, data, cam_id, point, body_id):
    """True if the first thing the ray from the camera towards `point` hits belongs to `body_id`."""
    origin = data.cam_xpos[cam_id]
    d = np.asarray(point) - origin
    dist = np.linalg.norm(d)
    gid = np.zeros(1, np.int32)
    hit = mujoco.mj_ray(model, data, origin, d / dist, None, 1, -1, gid)
    return gid[0] >= 0 and model.geom_bodyid[gid[0]] == body_id and hit >= 0


def first_hit_body(model, data, cam_id, point):
    """Name of the body that a ray from the camera towards `point` hits first ('' if nothing)."""
    origin = data.cam_xpos[cam_id]
    d = np.asarray(point) - origin
    dist = np.linalg.norm(d)
    gid = np.zeros(1, np.int32)
    mujoco.mj_ray(model, data, origin, d / dist, None, 1, -1, gid)
    return model.body(model.geom_bodyid[gid[0]]).name if gid[0] >= 0 else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="so101")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--min-external", type=float, default=0.99)
    p.add_argument("--min-wrist", type=float, default=0.90)
    p.add_argument("--external-items", default="blocks,tip,base,elbow",
                   help="what the external camera must keep in frame (a base-mounted camera needs only blocks,tip)")
    p.add_argument("--min-visible", type=float, default=0.0, help="required fraction of frames in which the blocks / gripper tip are not hidden behind the arm")
    p.add_argument("--min-fingers", type=float, default=0.0, help="required fraction of grasp-phase frames in which EVERY finger body is in the wrist view (0 = report only)")
    a = p.parse_args()
    cfg = load_robot(a.robot)
    env = MultiCellEnv(cfg, a.cells, a.image_size)
    ik, rng = ArmIK(cfg), np.random.default_rng(a.seed)
    m, d, S = env.model, env.data, a.image_size
    items = a.external_items.split(",")
    ext = {k: [] for k in ("blocks", "tip", "base", "elbow", "blocks visible", "tip visible", "all")}
    wrist = {"target_in_frame": [], "target_visible": []}
    rs = mujoco.Renderer(m, S, S)
    rs.enable_segmentation_rendering()
    finger_geoms = [{n: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == f"c{i}/{n}"] for n in cfg.gripper_bodies}
                    for i in range(a.cells)]
    fingers = {n: [] for n in cfg.gripper_bodies}
    fingers_all = []
    worst = []  # (score, frame)
    done_eps = 0
    while done_eps < a.episodes:
        experts = []
        for i in range(a.cells):
            env.reset_cell(i, sample_scene(cfg, rng))
            experts.append(PickExpert(env, ik, i))
        finished = [False] * a.cells
        for t in range(300):
            acts = []
            for e in experts:
                act, dn = e.act()
                acts.append(act)
            env.step(np.stack(acts))
            for i, e in enumerate(experts):
                if finished[i]:
                    continue
                if t % 3 == 0:
                    c = env.cells[i]
                    pre = e.phase in (APPROACH, DESCEND, GRASP)
                    pre_world = env.cube_pos(i) + c["origin"]
                    tip = d.site_xpos[c["site"]]
                    base = c["origin"]
                    elbow = d.xpos[m.jnt_bodyid[m.joint(c["prefix"] + cfg.arm_joints[2]).id]]
                    act = np.where(e.s.active)[0]  # parked spare blocks are not part of the scene
                    uv = project(m, d, c["cam"]["external"], [*pre_world[act], tip, base, elbow], S)
                    ok = inside(uv, S)
                    n = len(act)
                    got = {"blocks": bool(ok[:n].all()), "tip": bool(ok[n]), "base": bool(ok[n + 1]), "elbow": bool(ok[n + 2])}
                    for k, v in got.items():
                        ext[k].append(v)
                    cam_ext = c["cam"]["external"]
                    ext["blocks visible"].append(all(first_hit_body(m, d, cam_ext, pre_world[k]) == m.body(c["cubes"][k]).name for k in act))
                    hit = first_hit_body(m, d, cam_ext, tip)
                    ext["tip visible"].append(hit.split("/", 1)[-1].startswith("grip") or hit.split("/", 1)[-1] in cfg.gripper_bodies)
                    keep = all(got[k] for k in items)
                    ext["all"].append(keep)
                    if not keep:
                        worst.append((float(np.mean(list(got.values()))), env.render(i)["external"]))
                    if pre:
                        tgt = e.s.target
                        tw = pre_world[tgt]
                        wuv = project(m, d, c["cam"]["wrist"], [tw], S)
                        inf = bool(inside(wuv, S, 0.0)[0])
                        wrist["target_in_frame"].append(inf)
                        wrist["target_visible"].append(inf and visible(m, d, c["cam"]["wrist"], tw, c["cubes"][tgt]))
                        if not wrist["target_visible"][-1]:
                            worst.append((0.0, env.render(i)["wrist"]))
                        rs.update_scene(d, camera=c["cam"]["wrist"])
                        seg = rs.render()[..., 0]
                        seen = {n: bool(np.isin(seg, g).mean() > 0.002) for n, g in finger_geoms[i].items()}
                        for n, v in seen.items():
                            fingers[n].append(v)
                        fingers_all.append(all(seen.values()))
                if e.phase == DONE:
                    finished[i] = True
            if all(finished):
                break
        done_eps += a.cells
    print(f"{a.robot}: {done_eps} episodes")
    print("external cam, fraction of sampled frames with item inside the frame (margin %.0f%%):" % (MARGIN * 100))
    for k, v in ext.items():
        print(f"  {k:7s} {np.mean(v):6.1%}")
    print("wrist cam during approach/descend/grasp:")
    for k, v in wrist.items():
        if v:
            print(f"  {k:16s} {np.mean(v):6.1%}")
    print("gripper fingers in the wrist view during approach/descend/grasp (more than 0.2% of the pixels):")
    for n, v in fingers.items():
        print(f"  {n:24s} {np.mean(v):6.1%}")
    print(f"  {'every finger at once':24s} {np.mean(fingers_all):6.1%}")
    ok_ext = np.mean(ext["all"]) >= a.min_external and min(np.mean(ext["blocks visible"]), np.mean(ext["tip visible"])) >= a.min_visible
    ok_wrist = np.mean(wrist["target_visible"]) >= a.min_wrist
    ok_fingers = np.mean(fingers_all) >= a.min_fingers
    print("EXTERNAL", "PASS" if ok_ext else "FAIL", "| WRIST", "PASS" if ok_wrist else "FAIL", "| FINGERS", "PASS" if ok_fingers else "FAIL")
    if worst:
        worst.sort(key=lambda x: x[0])
        sel = [f for _, f in worst[:: max(1, len(worst) // 16)][:16]]
        sel += [np.zeros_like(sel[0])] * (-len(sel) % 8)
        rows = [np.concatenate(sel[r:r + 8], 1) for r in range(0, len(sel), 8)]
        imageio.imwrite("demo/camera_audit.png", np.concatenate(rows, 0))
        print("wrote demo/camera_audit.png (worst frames)")
    raise SystemExit(0 if ok_ext and ok_wrist and ok_fingers else 1)


if __name__ == "__main__":
    main()
