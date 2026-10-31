"""Robot description: everything robot-specific lives in a YAML under configs/robots."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "robots"


@dataclass
class GripperConfig:
    actuator: str  # actuator name (inside the robot, un-prefixed)
    ctrl_open: float
    ctrl_closed: float
    state_joint: str  # joint whose position is reported as gripper state
    q_open: float
    q_closed: float
    mjcf: str | None = None  # optional gripper model attached to `attach_site`
    attach_site: str | None = None
    attach_body: str | None = None
    attach_prefix: str = "grip_"
    attach_quat: list[float] = field(default_factory=lambda: [1, 0, 0, 0])


@dataclass
class CameraConfig:
    body: str  # body (un-prefixed, after gripper attach) the camera is mounted on
    pos: list[float]
    xyaxes: list[float]
    fovy: float = 75.0


@dataclass
class RobotConfig:
    name: str
    mjcf: str
    arm_joints: list[str]
    arm_actuators: list[str]
    home_qpos: list[float]
    gripper: GripperConfig
    ee_site: str  # grasp point; defined inside the robot or the attached gripper
    tool_axis: list[float]  # ee-frame axis pointing along the fingers (made to point down)
    grasp_axis: list[float]  # ee-frame axis along which the fingers close
    wrist_cam: CameraConfig
    external_cam_pos: list[float]  # relative to the robot base, camera looks at lookat
    external_cam_lookat: list[float]
    cube_half_size: float
    workspace_x: list[float]
    workspace_y: list[float]
    grasp_offset: list[float] | None = None  # grasp point relative to ee_site, in the site frame
    cube_half_height: float | None = None  # defaults to a cube
    approach_tilt_deg: float = 0.0  # tool tilted forward from vertical (tip away from base)
    ready_pos: list[float] | None = None  # if set, home pose = IK to this EE position (base frame)
    min_object_dist: float | None = None  # spacing between objects (open-jaw sweep)
    gripper_bodies: list[str] = field(default_factory=list)  # bodies whose geoms get gripper_friction
    gripper_friction: float = 4.0
    gripper_force: float | None = None  # torque limit [Nm] override: realistic grip forces keep soft contacts from sinking in
    contact_timeconst: float | None = None  # solref time constant for block/pad/table contacts (stiffer when smaller)
    noslip_iterations: int = 0  # post-solver anti-creep pass; essential for holding blocks
    object_friction: float = 4.0
    external_cam_fovy: float = 60.0
    zone_xy: list[float] = field(default_factory=lambda: [0.21, 0.20])  # place pads at (x, +-y) rel. to base
    zone_half: float = 0.045
    backdrop_color: list[float] = field(default_factory=lambda: [0.78, 0.8, 0.84])  # wall behind the robot
    tasks: list[str] = field(default_factory=lambda: ["lift", "place_left", "place_right"])
    task_weights: list[float] | None = None  # None: 0.4/0.3/0.3 for the default three tasks, else uniform
    n_blocks: list[int] = field(default_factory=lambda: [2, 3, 4])
    n_blocks_p: list[float] | None = None  # None: 0.25/0.45/0.30 for [2,3,4], else uniform
    colors: list[str] | None = None  # None: every colour in CUBE_COLORS
    block_half_xy: list[float] = field(default_factory=lambda: [0.0115, 0.0145])  # half width range [m]
    block_half_z: list[float] = field(default_factory=lambda: [0.016, 0.026])  # half height range [m]
    layout_pad: list[float] = field(default_factory=lambda: [0.03, 0.03])  # blocks may lie this far outside workspace_x/y (all blocks alike)
    gripper_binary: bool = False  # the gripper command is an open/close signal (no force or position ramp); the hardware closes by itself
    grip_rate: float = 0.015  # how fast the expert closes the gripper per control step (1/rate steps from open to closed)
    grasp_settle: int = 10  # extra steps the expert waits after the gripper command is fully closed before it lifts
    held_grip: float = 0.5  # gripper closed fraction above which a block between the jaws counts as held (sync/DAgger labelling)
    held_out_combos: bool = True  # keep colour x task combos for the test split only (only meaningful with prompts)
    show_pads: bool = True  # draw the left/right place pads
    table_half: list[float] = field(default_factory=lambda: [0.4, 0.4])
    table_color: list[float] = field(default_factory=lambda: [0.62, 0.48, 0.36])  # fixed per robot
    table_center: list[float] = field(default_factory=lambda: [0.0, 0.0])  # xy offset from the robot base
    cell_spacing: float = 1.5
    ik_pos_weight: float = 1.0
    ik_ori_weight: float = 0.3
    grasp_height: float = 0.0  # ee height above cube centre when grasping
    approach_height: float = 0.10
    max_ee_speed: float = 0.25  # m/s of the expert's Cartesian target
    strip_keyframes: bool = True
    notes: str = ""

    @property
    def block_h(self) -> float:
        return self.cube_half_height or self.cube_half_size

    @property
    def n_arm(self) -> int:
        return len(self.arm_joints)

    @property
    def action_dim(self) -> int:
        return self.n_arm + 1

    def abs_path(self, p: str) -> str:
        return str((REPO_ROOT / p).resolve())


def load_robot(name_or_path: str) -> RobotConfig:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["gripper"] = GripperConfig(**raw["gripper"])
    raw["wrist_cam"] = CameraConfig(**raw["wrist_cam"])
    return RobotConfig(**raw)


def look_at_xyaxes(pos, target, up=(0, 0, 1)) -> np.ndarray:
    """MuJoCo camera xyaxes for a camera at `pos` looking at `target`."""
    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return np.concatenate([x, y])
