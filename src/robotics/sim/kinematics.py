"""Damped-least-squares IK on a single-cell copy of the robot (robot-agnostic via RobotConfig)."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from robotics.robots.spec import RobotConfig
from robotics.sim.builder import build_world

DOWN = np.array([0.0, 0.0, -1.0])


class ArmIK:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.model = build_world(cfg, 1)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(f"c0/{j}").qposadr[0] for j in cfg.arm_joints])
        self.dadr = np.array([self.model.joint(f"c0/{j}").dofadr[0] for j in cfg.arm_joints])
        self.lo = np.array([self.model.joint(f"c0/{j}").range for j in cfg.arm_joints])[:, 0]
        self.hi = np.array([self.model.joint(f"c0/{j}").range for j in cfg.arm_joints])[:, 1]
        self.site = self.model.site(f"c0/{cfg.ee_site}").id
        self.tool = np.asarray(cfg.tool_axis, float)
        self.grasp = np.asarray(cfg.grasp_axis, float)
        self.offset = np.asarray(cfg.grasp_offset or [0, 0, 0], float)
        self.site_body = self.model.site_bodyid[self.site]

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        r = self.data.site_xmat[self.site].reshape(3, 3).copy()
        return self.data.site_xpos[self.site] + r @ self.offset, r

    def tool_dirs(self, cube_xy: np.ndarray, cube_yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """Target tool direction and horizontal closing-axis direction for a block at cube_xy."""
        phi = np.radians(self.cfg.approach_tilt_deg)
        if phi == 0.0:
            return DOWN, np.array([np.cos(cube_yaw), np.sin(cube_yaw), 0.0])
        u = np.append(cube_xy / np.linalg.norm(cube_xy), 0.0)
        return np.sin(phi) * u + np.cos(phi) * DOWN, np.cross([0, 0, 1.0], u)

    def target_rotation(self, tool_dir: np.ndarray, grasp_dir: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
        """Tool axis pointing down, fingers closing along the +-90deg-symmetric horizontal direction
        closest to the current grasp-axis heading."""
        cur = r_cur @ self.grasp
        best, best_dot = None, -2.0
        for k in range(4):
            a = np.arctan2(grasp_dir[1], grasp_dir[0]) + k * np.pi / 2
            d = np.array([np.cos(a), np.sin(a), 0.0])
            if d @ cur > best_dot:
                best, best_dot = d, d @ cur
        d = best - (best @ tool_dir) * tool_dir
        d /= np.linalg.norm(d)
        w = np.stack([tool_dir, d, np.cross(tool_dir, d)], axis=1)
        b = np.stack([self.tool, self.grasp, np.cross(self.tool, self.grasp)], axis=1)
        return w @ b.T

    def solve(self, q0: np.ndarray, pos: np.ndarray, tool_dir: np.ndarray, grasp_dir: np.ndarray, iters: int = 40,
              damping: float = 1e-2) -> np.ndarray:
        q = np.clip(q0.copy(), self.lo, self.hi)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        wp, wr = self.cfg.ik_pos_weight, self.cfg.ik_ori_weight
        for _ in range(iters):
            p, r_cur = self.fk(q)
            r_t = self.target_rotation(tool_dir, grasp_dir, r_cur)
            e_p = pos - p
            e_r = R.from_matrix(r_t @ r_cur.T).as_rotvec()
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-2:
                break
            mujoco.mj_jac(self.model, self.data, jp, jr, p, self.site_body)
            J = np.vstack([wp * jp[:, self.dadr], wr * jr[:, self.dadr]])
            e = np.concatenate([wp * e_p, wr * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q
