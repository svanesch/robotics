"""Scripted pick/place expert: privileged block pose -> Cartesian waypoints -> IK joint targets.

Tasks: "lift" (pick up a block and hold it up), "place_left"/"place_right" (carry it onto a pad and release),
"stack" (block 1 onto block 0, then block 2 onto block 1; the expert moves `scene.target` from 1 to 2 after the first place).
"""
from __future__ import annotations

import numpy as np

from robotics.envs.multi_cell import MultiCellEnv
from robotics.sim.kinematics import ArmIK

APPROACH, DESCEND, GRASP, LIFT, MOVE, LOWER, RELEASE, RETRACT, HOLD, DONE = range(10)
MOVING = (APPROACH, DESCEND, LIFT, MOVE, LOWER, RETRACT)
LIFT_HEIGHT = 0.12
CARRY_HEIGHT = 0.09  # block-centre height above its resting height while carrying it to a pad
SUCCESS_LIFT = 0.04
GRIP_RATE = 0.015
RELEASE_RATE = 0.04
MAX_DQ = 0.12  # rad per control step
STACK_GRASP_UP = 0.0  # grasp the block this far above its centre when stacking (0.008 / 0.01 did not help)
STACK_LIFT = 0.10  # lift height above the resting height when stacking (the arm sags short of a higher target)
STACK_TIMEOUT = 35  # steps to wait for the measured end-effector to catch up with a target it cannot quite reach
STACK_CLEAR = 0.03  # a block carried to a tower stays this far above the top it is going onto


class PickExpert:
    def __init__(self, env: MultiCellEnv, ik: ArmIK, cell: int):
        self.env, self.ik, self.i = env, ik, cell
        self.reset()

    def reset(self) -> None:
        env, i = self.env, self.i
        self.s = env.cells[i]["scene"]
        self.stack = self.s.task == "stack"
        self.place = self.s.task != "lift"
        self.zone = env.zone_center(1 if self.s.task == "place_left" else -1) if self.place and not self.stack else None
        self.stage = 0
        if self.stack:
            self.s.target = 1
        self.phase = APPROACH
        self.t_phase = 0
        self.q_cmd = env.state(i)[:-1].astype(float)
        self.ee_target = env.ee_pose(i)[:3].astype(float)
        self.grip = 0.0
        self.success_steps = 0
        self.hold_offset = np.zeros(2)  # ee xy minus block xy once grasped
        self.g_prev, self.settled = None, 0  # gripper closure last step / steps it has not moved (binary-gripper grasp)
        self.yaw = float(self.s.cube_yaw[self.s.target])  # block yaw used for the grasp orientation
        self.confident = True

    # --------------------------------------------------------------- goals
    def _goal(self) -> np.ndarray:
        cfg, s = self.env.cfg, self.s
        cube = self.env.cube_pos(self.i)[s.target].astype(float)
        self.cube = cube
        z0 = s.cube_half[s.target][2]
        grasp_z = z0 + cfg.grasp_height + (STACK_GRASP_UP if self.stack else 0.0)
        if self.phase == APPROACH:
            return np.array([cube[0], cube[1], grasp_z + cfg.approach_height + s.hover_extra])
        if self.phase == DESCEND:
            return np.array([cube[0], cube[1], grasp_z])
        if self.stack:
            base = self.env.cube_pos(self.i)[self.stage].astype(float)  # stage 0 -> on block 0, stage 1 -> on block 1
            base_top = base[2] + s.cube_half[self.stage][2]
            clear = base_top + z0 + cfg.grasp_height + STACK_CLEAR  # held block clears the tower it is going onto
            held_off = (self.env.ee_pose(self.i)[:2] - cube[:2]).astype(float)  # the block may have slid in the jaws since the grasp
            xy = base[:2] + held_off  # the held block ends up centred over the base block
            if self.phase == LIFT:
                return np.array([self.ee_target[0], self.ee_target[1], max(z0 + STACK_LIFT, clear)])
            if self.phase == MOVE:
                return np.array([*xy, clear])
            if self.phase == LOWER:
                return np.array([*xy, base_top + z0 + cfg.grasp_height + 0.004])  # block bottom just above the base block
            # RETRACT: first centre the open jaws on the block just placed, so no pad drags it (or the tower) up with it
            return np.array([cube[0], cube[1], max(z0 + STACK_LIFT, clear)])
        if self.phase == LIFT:
            return np.array([self.ee_target[0], self.ee_target[1], z0 + LIFT_HEIGHT])
        xy = self.zone + self.hold_offset if self.place else self.ee_target[:2]
        if self.phase == MOVE:
            return np.array([*xy, grasp_z + CARRY_HEIGHT])
        if self.phase == LOWER:
            return np.array([*xy, grasp_z + 0.004])  # block bottom just above the pad
        return np.array([self.ee_target[0], self.ee_target[1], z0 + LIFT_HEIGHT])  # RETRACT

    def _after(self, phase: int) -> int:
        if phase == LIFT:
            return MOVE if self.place else HOLD
        return {APPROACH: DESCEND, DESCEND: GRASP, MOVE: LOWER, LOWER: RELEASE, RETRACT: DONE}[phase]

    # ----------------------------------------------------------------- step
    def act(self) -> tuple[np.ndarray, bool]:
        env, cfg, s = self.env, self.env.cfg, self.s
        dt = 1.0 / env.fps
        self.t_phase += 1
        goal = self._goal()
        if self.phase in MOVING:
            step = cfg.max_ee_speed * s.speed_scale * dt
            if self.phase in (DESCEND, LOWER):
                step *= 0.6
            delta = goal - self.ee_target
            dist = np.linalg.norm(delta)
            self.ee_target = goal if dist < step else self.ee_target + delta / dist * step
            at = np.linalg.norm(self.ee_target - goal) < 1e-6
            ee = env.ee_pose(self.i)[:3]
            reached = at and np.linalg.norm(ee - goal) < 0.012
            if reached or (at and self.t_phase > (STACK_TIMEOUT if self.stack else 90)):
                self.phase, self.t_phase = self._after(self.phase), 0
                if self.phase == DONE and self.stack and self.stage == 0:
                    self._next_stage()
        elif self.phase == GRASP:
            if cfg.gripper_binary:  # 'close' is a signal, not a force: the jaws close by themselves, lift once they have stopped moving
                self.grip = 1.0
                g = float(env.state(self.i)[-1])
                still = self.g_prev is not None and abs(g - self.g_prev) < 0.004 and g > cfg.held_grip
                self.settled, self.g_prev = (self.settled + 1 if still else 0), g
                ready = self.settled >= 4 or self.t_phase > 80
            else:
                self.grip = min(1.0, self.grip + cfg.grip_rate)
                ready = self.grip >= 1.0 and self.t_phase > 1 / cfg.grip_rate + cfg.grasp_settle
            if ready:
                self.g_prev, self.settled = None, 0
                self.hold_offset = (env.ee_pose(self.i)[:2] - self.cube[:2]).astype(float)
                self.phase, self.t_phase = LIFT, 0
        elif self.phase == RELEASE:
            self.grip = 0.0 if cfg.gripper_binary else max(0.0, self.grip - RELEASE_RATE)
            if self.grip <= 0.0 and self.t_phase > (12 if cfg.gripper_binary else 1 / RELEASE_RATE + 8):
                self.phase, self.t_phase = RETRACT, 0
        elif self.phase == HOLD:
            if self.t_phase > 10:
                self.phase = DONE
        if self.phase == LIFT and not self.place and self.grip >= 1.0 and self.t_phase > 90:
            self.phase = HOLD

        tool_dir, grasp_dir = self.ik.tool_dirs(self.cube[:2], self.yaw)
        q_new = self.ik.solve(self.q_cmd, self.ee_target, tool_dir, grasp_dir, iters=25)
        self.q_cmd = self.q_cmd + np.clip(q_new - self.q_cmd, -MAX_DQ, MAX_DQ)

        if not self.place:
            lifted = env.target_lifted(self.i) > SUCCESS_LIFT
            self.success_steps = self.success_steps + 1 if lifted and self.phase in (LIFT, HOLD, DONE) else 0
            if self.phase == LIFT and self.success_steps > 0 and np.linalg.norm(self.ee_target - goal) < 1e-6:
                self.phase, self.t_phase = HOLD, 0
        return np.append(self.q_cmd, self.grip), self.phase == DONE

    def _next_stage(self) -> None:
        """First block is on the base: pick the third block next and put it on the second one."""
        self.stage, self.s.target = 1, 2
        self.phase, self.t_phase, self.grip = APPROACH, 0, 0.0
        self.g_prev, self.settled = None, 0
        self.yaw = float(self.s.cube_yaw[2])

    # ----------------------------------------------------------------- sync
    def sync(self, max_dev_q: float = 0.12, max_dev_ee: float = 0.03) -> None:
        """Re-plan from the *actual* world state (used for DAgger, when another policy has been driving the arm).

        Resets the commanded joints/end-effector to the measured ones and infers the phase from the gripper and block
        state. Measured: the clean expert's command leads the arm by < 1 control step, so measured pose + one step is
        consistent with the labels in the original dataset. Sets `self.confident = False` if the situation is not one the
        scripted expert can handle (block toppled, out of reach).
        """
        env, i, s, cfg = self.env, self.i, self.s, self.env.cfg
        st = env.state(i)
        q_meas, ee, g = st[:-1].astype(float), env.ee_pose(i)[:3].astype(float), float(st[-1])
        # re-anchor the commanded pose to the measured one only when the arm really is elsewhere (another policy drove it);
        # otherwise the command keeps integrating as in the original expert, which is what lifts the arm against gravity
        if np.abs(self.q_cmd - q_meas).max() > max_dev_q or np.linalg.norm(self.ee_target - ee) > max_dev_ee:
            self.q_cmd, self.ee_target = q_meas, ee
        if self.stack:  # which block is next follows from the tower, not from where the previous controller was
            stage = 1 if (env.stacked(i, 1, 0) and g < 0.3) else 0  # block 1 counts as placed once the jaws let go of it
            if stage != self.stage:
                self.stage, s.target = stage, 1 + stage
        cube = env.cube_pos(i)[s.target].astype(float)
        self.cube = cube
        z0 = s.cube_half[s.target][2]
        rot = env.data.xmat[env.cells[i]["cubes"][s.target]].reshape(3, 3)
        upright = abs(rot[2, 2]) > 0.85
        reach_ok = (cfg.workspace_x[0] - 0.03 <= cube[0] <= cfg.workspace_x[1] + 0.03
                    and cfg.workspace_y[0] - 0.03 <= cube[1] <= cfg.workspace_y[1] + 0.03)
        zone_ok = (env.stacked(i, s.target, self.stage) if self.stack
                   else self.place and bool(np.all(np.abs(cube[:2] - self.zone) < cfg.zone_half)))
        # the grasp point sits grasp_height above the block centre when the block is gripped, whatever its height
        near_grasp = np.linalg.norm(ee - (cube + np.array([0.0, 0.0, cfg.grasp_height]))) < 0.035
        carrying = self.phase in (LIFT, MOVE, LOWER, HOLD)
        near_hold = np.linalg.norm(ee - (cube + np.array([0.0, 0.0, cfg.grasp_height]))) < 0.05  # the block may slip a little in the jaws
        held = near_hold and g > cfg.held_grip and (cube[2] - z0 > 0.012 or carrying)
        self.confident = bool(held or zone_ok or (upright and reach_ok))
        if upright:
            self.yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        grasp_pt = np.array([cube[0], cube[1], z0 + cfg.grasp_height])
        # keep the running phase (and its timers / gripper command) while it still matches the world; only re-infer otherwise
        far_xy = np.linalg.norm(ee[:2] - cube[:2])
        consistent = ((carrying and (held or (self.phase == LIFT and near_grasp and g > cfg.held_grip)))
                      or (self.phase == GRASP and near_grasp)
                      or (self.phase == APPROACH and not held and not near_grasp)
                      or (self.phase == DESCEND and not held and far_xy < 0.03)
                      or (self.phase in (RELEASE, RETRACT, DONE) and zone_ok))
        if consistent:
            return
        self.t_phase = 0
        self.grip = g
        self.g_prev, self.settled = None, 0
        if zone_ok and not held:
            self.phase = RELEASE if g > 0.3 else RETRACT
        elif held:
            self.hold_offset = (ee[:2] - cube[:2]).astype(float)
            self.phase = MOVE if self.place else LIFT
            self.grip = 1.0
        elif near_grasp and g > 0.05:
            self.phase = GRASP  # jaws are around the block and closing: carry on closing
        else:
            self.grip = 0.0  # jaws open while (re)approaching
            near = np.linalg.norm(ee[:2] - cube[:2]) < 0.012
            self.phase = DESCEND if near else APPROACH
        if cfg.gripper_binary:
            self.grip = 1.0 if self.phase in (GRASP, LIFT, MOVE, LOWER, HOLD) else 0.0

    # --------------------------------------------------------------- result
    @property
    def success(self) -> bool:
        if not self.place:
            return self.success_steps >= 5
        if self.stack:
            return self.tower()
        return self.placed()

    def tower(self) -> bool:
        """Block 1 rests on block 0, block 2 rests on block 1, and the gripper has let go."""
        return bool(self.env.stacked(self.i, 1, 0) and self.env.stacked(self.i, 2, 1) and self.env.state(self.i)[-1] < 0.3)

    def placed(self) -> bool:
        """Target block rests on the requested pad (within it, on the table) and the gripper is open."""
        env, s = self.env, self.s
        c = env.cube_pos(self.i)[s.target]
        in_zone = bool(np.all(np.abs(c[:2] - self.zone) < env.cfg.zone_half))
        resting = env.target_lifted(self.i) < 0.01
        return in_zone and resting and self.phase == DONE
