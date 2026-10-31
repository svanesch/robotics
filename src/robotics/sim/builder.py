"""Builds a MuJoCo world with N independent robot 'cells' from a RobotConfig using MjSpec.

Each cell (prefix ``c{i}/``) holds: the robot (+ optional attached gripper), a table, a few
coloured cubes (free bodies), a wrist camera and an external camera. Cells are spaced so far
apart that they never touch, which lets one MjModel/MjData simulate N episodes at once.
"""
from __future__ import annotations

import mujoco
import numpy as np

from robotics.robots.spec import RobotConfig, look_at_xyaxes

CUBE_COLORS = {
    "red": (0.85, 0.15, 0.15, 1),
    "white": (0.92, 0.92, 0.9, 1),
    "black": (0.08, 0.08, 0.09, 1),
    "green": (0.15, 0.7, 0.2, 1),
    "blue": (0.15, 0.3, 0.85, 1),
    "cyan": (0.1, 0.8, 0.85, 1),
    "pink": (0.95, 0.45, 0.7, 1),
    "orange": (0.95, 0.5, 0.1, 1),
    "purple": (0.55, 0.2, 0.7, 1),
}
N_CUBES = 4
FLOOR_Z = -0.75
CONTROL_FPS = 30
SUBSTEPS = 16


def load_robot_spec(cfg: RobotConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(cfg.abs_path(cfg.mjcf))
    g = cfg.gripper
    if g.mjcf is not None:
        gspec = mujoco.MjSpec.from_file(cfg.abs_path(g.mjcf))
        gbody = gspec.body(g.attach_body)
        spec.site(g.attach_site).attach_body(gbody, g.attach_prefix, "")
    for name in cfg.gripper_bodies:  # sticky fingers: parallel-jaw/pliers sim grasps need high friction
        for geom in spec.body(name).geoms:
            geom.friction[0] = cfg.gripper_friction
            if cfg.contact_timeconst:
                geom.solref = [cfg.contact_timeconst, 1.0]
    if cfg.gripper_force:
        act = spec.actuator(cfg.gripper.actuator)
        act.forcerange = [-cfg.gripper_force, cfg.gripper_force]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    if cfg.strip_keyframes:
        for key in list(spec.keys):
            spec.delete(key)
    wc = cfg.wrist_cam
    spec.body(wc.body).add_camera(name="wrist_cam", pos=wc.pos, xyaxes=wc.xyaxes, fovy=wc.fovy)
    return spec


def build_world(cfg: RobotConfig, n_cells: int = 1, image_size: int = 128, shadows: bool = False) -> mujoco.MjModel:
    """shadows=False is ~5x faster to render with many cells (each cell's light adds a shadow pass)."""
    world = mujoco.MjSpec()
    world.option.timestep = 1.0 / (CONTROL_FPS * SUBSTEPS)
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.noslip_iterations = cfg.noslip_iterations
    world.visual.global_.offwidth = max(image_size, 1024)
    world.visual.global_.offheight = max(image_size, 1024)
    # studio look: soft shadows, anti-aliasing, light floor, glossy materials
    world.visual.quality.shadowsize = 4096
    world.visual.quality.offsamples = 8 if shadows else 4
    world.visual.headlight.ambient = [0.2, 0.2, 0.22]
    world.visual.headlight.diffuse = [0.12, 0.12, 0.12]
    world.visual.headlight.specular = [0.0, 0.0, 0.0]
    world.visual.rgba.haze = [0.82, 0.86, 0.92, 1]
    world.visual.map.haze = 0.35
    world.visual.map.fogstart = 2.0
    world.visual.map.fogend = 9.0
    world.visual.map.shadowclip = 2.0
    world.stat.extent = 1.0

    world.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=[0.86, 0.9, 0.96], rgb2=[0.55, 0.62, 0.72], width=512, height=512)
    world.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_NONE,
                      rgb1=[0.86, 0.87, 0.89], rgb2=[0.82, 0.83, 0.86], width=512, height=512)
    world.add_material(name="floor", textures=["", "grid"], texuniform=True, texrepeat=[6, 6],
                       reflectance=0.18, specular=0.2, shininess=0.3)
    world.add_material(name="table", rgba=[0.62, 0.45, 0.3, 1], specular=0.25, shininess=0.35, reflectance=0.04)
    world.add_material(name="backdrop", rgba=[0.78, 0.8, 0.84, 1], specular=0.05, shininess=0.05)
    world.add_material(name="block", rgba=[0.8, 0.2, 0.2, 1], specular=0.6, shininess=0.7)
    world.add_material(name="metal", rgba=[0.22, 0.23, 0.25, 1], specular=0.7, shininess=0.8)
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                             pos=[0, 0, FLOOR_Z], material="floor")

    cols = int(np.ceil(np.sqrt(n_cells)))
    for i in range(n_cells):
        origin = np.array([(i % cols) * cfg.cell_spacing, (i // cols) * cfg.cell_spacing, 0.0])
        _add_cell(world, cfg, i, origin, shadows)
    return world.compile()


def _add_cell(world: mujoco.MjSpec, cfg: RobotConfig, i: int, origin: np.ndarray, shadows: bool = False) -> None:
    p = f"c{i}/"
    wb = world.worldbody
    hx, hy = cfg.table_half

    # table: slab with top at z=0 (robot base height) plus legs
    top = wb.add_body(name=p + "table", pos=origin + [*cfg.table_center, 0.0])
    top.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.02], pos=[0, 0, -0.02],
                 material="table", rgba=[*cfg.table_color, 1], friction=[1, 0.005, 0.0001],
                 name=p + "table_top", **({"solref": [cfg.contact_timeconst, 1.0]} if cfg.contact_timeconst else {}))
    for sx in (-1, 1):
        for sy in (-1, 1):
            top.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, -FLOOR_Z / 2 - 0.02],
                         pos=[sx * (hx - 0.03), sy * (hy - 0.03), FLOOR_Z / 2 - 0.02],
                         material="metal", contype=0, conaffinity=0)

    # place pads (flat, no collision): left = +y, right = -y, as seen from the robot
    zx, zy = cfg.zone_xy
    for sgn, side in (((1, "left"), (-1, "right")) if cfg.show_pads else ()):
        top.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[cfg.zone_half, cfg.zone_half, 0.0006],
                     pos=[zx - cfg.table_center[0], sgn * zy - cfg.table_center[1], 0.0006],
                     rgba=[0.16, 0.2, 0.26, 1], contype=0, conaffinity=0, name=f"{p}pad_{side}")

    # (backdrop below)
    # plain backdrop at the far end of the external camera's view (same in every cell, so no cell sees its neighbour)
    toward = 1.0 if cfg.external_cam_lookat[0] >= cfg.external_cam_pos[0] else -1.0
    wx = cfg.table_center[0] + toward * (hx + 0.03)
    top.parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.9, 0.6], pos=origin + [wx, 0, 0.6 + FLOOR_Z],
                        rgba=[*cfg.backdrop_color, 1], contype=0, conaffinity=0, material="backdrop")

    # lights and external camera
    # soft key light from the front-left (shadows), plus a dim fill from the opposite side
    key = origin + [0.5, -0.5, 1.6]
    wb.add_light(name=p + "light", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=key,
                 dir=(origin + [0.15, 0, 0]) - key, cutoff=55, exponent=2,
                 diffuse=[0.42, 0.42, 0.4], specular=[0.3, 0.3, 0.3], castshadow=shadows)
    wb.add_light(name=p + "fill", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=origin + [-0.4, 0.7, 1.2],
                 dir=[0.4, -0.5, -1.0], cutoff=70, exponent=1, diffuse=[0.25, 0.27, 0.32], castshadow=False)
    ext_pos = origin + cfg.external_cam_pos
    wb.add_camera(name=p + "external_cam", pos=ext_pos, fovy=cfg.external_cam_fovy,
                  xyaxes=look_at_xyaxes(ext_pos, origin + cfg.external_cam_lookat))

    # cubes
    s, h = cfg.cube_half_size, cfg.block_h
    for k in range(N_CUBES):
        b = wb.add_body(name=f"{p}cube{k}", pos=origin + [0.0, 25.0 + k, FLOOR_Z + h + 0.001])  # parked far away (beyond the haze) until used
        b.add_freejoint(name=f"{p}cube{k}_joint")
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, h], mass=0.03, material="block",
                   friction=[cfg.object_friction, 0.01, 0.001], rgba=[0.8, 0.2, 0.2, 1], name=f"{p}cube{k}_geom",
                   condim=4, **({"solref": [cfg.contact_timeconst, 1.0]} if cfg.contact_timeconst else {}))

    # robot (a fresh spec per cell: attach consumes/copies the child)
    frame = wb.add_frame(pos=origin)
    world.attach(load_robot_spec(cfg), frame=frame, prefix=p)
