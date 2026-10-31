"""N independent pick-cube cells simulated in a single MjModel, with wrist + external cameras."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotics.robots.spec import RobotConfig, look_at_xyaxes
from robotics.sim.builder import CUBE_COLORS, N_CUBES, SUBSTEPS, CONTROL_FPS, build_world

CAMERAS = ("wrist", "external")


@dataclass
class SceneSample:
    """Everything randomised per episode for one cell."""
    cube_xy: np.ndarray  # (N_CUBES, 2) relative to robot base
    cube_yaw: np.ndarray  # (N_CUBES,)
    cube_colors: list[str]
    target: int  # index of the cube to pick
    home_noise: np.ndarray
    light_offset: np.ndarray
    light_intensity: float
    cam_pos_jitter: np.ndarray
    cam_look_jitter: np.ndarray
    speed_scale: float
    task: str  # "lift" | "place_left" | "place_right"
    active: np.ndarray  # (N_CUBES,) bool: blocks on the table; the rest are parked out of view
    cube_half: np.ndarray  # (N_CUBES, 3) half extents
    hover_extra: float
    instruction: str


class MultiCellEnv:
    def __init__(self, cfg: RobotConfig, n_cells: int = 1, image_size: int = 128, render: bool = True,
                 shadows: bool = False):
        self.cfg, self.n, self.image_size = cfg, n_cells, image_size
        self.model = build_world(cfg, n_cells, image_size, shadows)
        self.data = mujoco.MjData(self.model)
        self.fps = CONTROL_FPS
        self.renderer = mujoco.Renderer(self.model, image_size, image_size) if render else None
        m = self.model
        self.cells = []
        for i in range(n_cells):
            p = f"c{i}/"
            c = dict(
                prefix=p,
                origin=m.body(p + "table").pos - [*cfg.table_center, 0.0],
                arm_qadr=np.array([m.joint(p + j).qposadr[0] for j in cfg.arm_joints]),
                arm_ctrl=np.array([m.actuator(p + a).id for a in cfg.arm_actuators]),
                grip_ctrl=m.actuator(p + cfg.gripper.actuator).id,
                grip_qadr=m.joint(p + cfg.gripper.state_joint).qposadr[0],
                site=m.site(p + cfg.ee_site).id,
                cubes=[m.body(f"{p}cube{k}").id for k in range(N_CUBES)],
                cube_qadr=[m.joint(f"{p}cube{k}_joint").qposadr[0] for k in range(N_CUBES)],
                cube_geom=[m.geom(f"{p}cube{k}_geom").id for k in range(N_CUBES)],
                cam={"wrist": m.camera(p + "wrist_cam").id, "external": m.camera(p + "external_cam").id},
                light=m.light(p + "light").id,
                table_geom=m.geom(p + "table_top").id,
                joints=[j for j in range(m.njnt) if m.joint(j).name.startswith(p)],
            )
            c["ext_pos0"] = m.cam_pos[c["cam"]["external"]].copy()
            c["light_pos0"] = m.light_pos[c["light"]].copy()
            c["light_diff0"] = m.light_diffuse[c["light"]].copy()
            c["scene"] = None
            self.cells.append(c)
        self.home = np.asarray(cfg.home_qpos, float)
        if cfg.ready_pos is not None:
            from robotics.sim.kinematics import ArmIK

            ik = ArmIK(cfg)
            tool, grasp = ik.tool_dirs(np.array([1.0, 0.0]), 0.0)
            self.home = ik.solve(self.home, np.asarray(cfg.ready_pos, float), tool, grasp, iters=200)
        self.grasp_offset = np.asarray(cfg.grasp_offset or [0, 0, 0], float)

    # ---------------------------------------------------------------- reset
    def reset_cell(self, i: int, s: SceneSample) -> None:
        m, d, c, cfg = self.model, self.data, self.cells[i], self.cfg
        for j in c["joints"]:
            a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
            n = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(m.jnt_type[j], 1)
            nv = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(m.jnt_type[j], 1)
            d.qpos[a:a + n] = m.qpos0[a:a + n]
            d.qvel[v:v + nv] = 0
        q_home = self.home + s.home_noise
        d.qpos[c["arm_qadr"]] = q_home
        d.ctrl[c["arm_ctrl"]] = q_home
        d.ctrl[c["grip_ctrl"]] = cfg.gripper.ctrl_open
        for k in range(N_CUBES):
            g = c["cube_geom"][k]
            m.geom_size[g] = s.cube_half[k]
            m.geom_rbound[g] = np.linalg.norm(s.cube_half[k])  # keep broadphase bounds consistent with the size
            m.geom_aabb[g] = [0, 0, 0, *s.cube_half[k]]
            m.geom_rgba[g] = CUBE_COLORS[s.cube_colors[k]]
            if s.active[k]:  # inactive blocks stay parked behind the backdrop (their qpos0)
                a = c["cube_qadr"][k]
                d.qpos[a:a + 3] = c["origin"] + [*s.cube_xy[k], s.cube_half[k][2] + 5e-4]
                d.qpos[a + 3:a + 7] = [np.cos(s.cube_yaw[k] / 2), 0, 0, np.sin(s.cube_yaw[k] / 2)]
        m.light_pos[c["light"]] = c["light_pos0"] + s.light_offset
        m.light_diffuse[c["light"]] = c["light_diff0"] * s.light_intensity
        cam = c["cam"]["external"]
        pos = c["origin"] + self.cfg.external_cam_pos + s.cam_pos_jitter
        look = c["origin"] + self.cfg.external_cam_lookat + s.cam_look_jitter
        axes = look_at_xyaxes(pos, look)
        xa, ya = axes[:3], axes[3:]
        mat = np.stack([xa, ya, np.cross(xa, ya)], axis=1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        m.cam_pos[cam], m.cam_quat[cam] = pos, quat
        c["scene"] = s
        mujoco.mj_forward(m, d)

    # ----------------------------------------------------------------- step
    def step(self, actions: np.ndarray) -> None:
        """actions: (n, n_arm+1): absolute arm joint targets (rad) and gripper close command in [0,1]."""
        g = self.cfg.gripper
        for i, c in enumerate(self.cells):
            a = actions[i]
            self.data.ctrl[c["arm_ctrl"]] = a[:-1]
            close = float(a[-1] > 0.5) if self.cfg.gripper_binary else float(np.clip(a[-1], 0, 1))  # binary: an open/close signal
            self.data.ctrl[c["grip_ctrl"]] = g.ctrl_open + close * (g.ctrl_closed - g.ctrl_open)
        mujoco.mj_step(self.model, self.data, nstep=SUBSTEPS)

    # ----------------------------------------------------------------- obs
    def state(self, i: int) -> np.ndarray:
        c, g = self.cells[i], self.cfg.gripper
        gq = (self.data.qpos[c["grip_qadr"]] - g.q_open) / (g.q_closed - g.q_open)
        return np.append(self.data.qpos[c["arm_qadr"]], gq).astype(np.float32)

    def ee_pose(self, i: int) -> np.ndarray:
        c = self.cells[i]
        quat = np.zeros(4)
        mat = self.data.site_xmat[c["site"]]
        mujoco.mju_mat2Quat(quat, mat)
        pos = self.data.site_xpos[c["site"]] + mat.reshape(3, 3) @ self.grasp_offset
        return np.concatenate([pos - c["origin"], quat]).astype(np.float32)

    def cube_pos(self, i: int) -> np.ndarray:
        """(N_CUBES, 3) positions relative to the robot base."""
        c = self.cells[i]
        return np.stack([self.data.xpos[b] - c["origin"] for b in c["cubes"]]).astype(np.float32)

    def render(self, i: int) -> dict[str, np.ndarray]:
        out = {}
        for name in CAMERAS:
            self.renderer.update_scene(self.data, camera=self.cells[i]["cam"][name])
            out[name] = self.renderer.render().copy()
        return out

    def target_lifted(self, i: int) -> float:
        s = self.cells[i]["scene"]
        return float(self.cube_pos(i)[s.target][2] - s.cube_half[s.target][2])

    def zone_center(self, side: int) -> np.ndarray:
        """xy of the place pad (+1 = left/+y, -1 = right/-y) relative to the robot base."""
        return np.array([self.cfg.zone_xy[0], side * self.cfg.zone_xy[1]])
