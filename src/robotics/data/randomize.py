"""Per-episode scene randomisation, tasks and language instructions (all robot-specific ranges come from the RobotConfig)."""
from __future__ import annotations

import numpy as np

from robotics.envs.multi_cell import SceneSample
from robotics.robots.spec import RobotConfig
from robotics.sim.builder import CUBE_COLORS, N_CUBES

TASKS = ("lift", "place_left", "place_right")
STACK_COLORS = ("red", "green", "blue")  # stack task: block 0 is the base, 1 goes on it, 2 goes on block 1
TASK_WEIGHTS = (0.4, 0.3, 0.3)

LIFT_TEMPLATES = [
    "pick up the {c} cube", "grab the {c} cube", "lift the {c} cube", "pick the {c} block and lift it",
    "pick up the {c} block", "take the {c} cube and raise it", "grasp the {c} cube and lift it up",
    "lift up the {c} block off the table", "grab the {c} block", "raise the {c} block",
]
PLACE_TEMPLATES = [
    "put the {c} block on the {side}", "move the {c} cube to the {side}", "place the {c} block on the {side} side",
    "put the {c} cube on the {side} pad", "pick up the {c} block and place it on the {side}",
    "take the {c} cube to the {side} side", "move the {c} block to the {side} pad",
    "place the {c} cube on the {side}", "carry the {c} block to the {side} side",
]

# colour x task combinations never seen in train/val; the test split uses them half of the time (prompt experiments only)
HELD_OUT = (("pink", "place_right"), ("cyan", "place_left"), ("orange", "place_left"), ("purple", "lift"))
MIN_ZONE_DIST = 0.11  # keep every block this far from both pad centres
CONFIG_TRIES = 300


def _positions(cfg: RobotConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """All blocks come from the same distribution (uniform in the workspace plus a symmetric margin, minimum spacing, clear of the place pads), so a
    block's position says nothing about whether it is the target. Whole-layout rejection keeps the joint distribution symmetric."""
    min_dist = cfg.min_object_dist or 0.1
    zones = [np.array([cfg.zone_xy[0], s * cfg.zone_xy[1]]) for s in (1, -1)] if cfg.show_pads else []
    for _ in range(CONFIG_TRIES):
        px, py = cfg.layout_pad
        pts = np.stack([[rng.uniform(cfg.workspace_x[0] - px, cfg.workspace_x[1] + px),
                         rng.uniform(cfg.workspace_y[0] - py, cfg.workspace_y[1] + py)] for _ in range(n)])
        ok = all(np.linalg.norm(pts[i] - pts[j]) > min_dist for i in range(n) for j in range(i))
        ok = ok and all(np.linalg.norm(p - z) > MIN_ZONE_DIST for p in pts for z in zones)
        if ok:
            out = np.zeros((N_CUBES, 2))
            out[:n] = pts
            return out
    # crowded workspace (e.g. 4 blocks on the SO-101): place blocks one after another. The target index is drawn independently
    # of the order, so no block position is more likely to be the target.
    for _ in range(CONFIG_TRIES):
        pts: list[np.ndarray] = []
        for _ in range(n):
            for _ in range(2000):
                p = np.array([rng.uniform(cfg.workspace_x[0] - px, cfg.workspace_x[1] + px),
                              rng.uniform(cfg.workspace_y[0] - py, cfg.workspace_y[1] + py)])
                if all(np.linalg.norm(p - q) > min_dist for q in pts) and all(np.linalg.norm(p - z) > MIN_ZONE_DIST for z in zones):
                    pts.append(p)
                    break
            else:
                break
        if len(pts) == n:
            out = np.zeros((N_CUBES, 2))
            out[:n] = np.stack(pts)
            return out
    raise RuntimeError("could not place blocks")


def sample_scene(cfg: RobotConfig, rng: np.random.Generator, split: str = "train", task: str | None = None,
                 target_color: str | None = None) -> SceneSample:
    """split="test" draws held-out colour x task combos half of the time when cfg.held_out_combos; train/val never show them."""
    colors_all = list(cfg.colors or CUBE_COLORS)
    n_choices = list(cfg.n_blocks)
    p_n = cfg.n_blocks_p or ([0.25, 0.45, 0.30] if n_choices == [2, 3, 4] else None)
    n = int(rng.choice(n_choices, p=p_n))
    target = int(rng.integers(n))
    tasks = list(cfg.tasks)
    p_t = cfg.task_weights or (list(TASK_WEIGHTS) if tasks == list(TASKS) else None)
    chosen_task = task or str(rng.choice(tasks, p=p_t))
    # colours: distinct for as many blocks as there are colours, cycled for the rest
    base = list(rng.choice(colors_all, size=len(colors_all), replace=False))
    colors = [base[k % len(base)] for k in range(N_CUBES)]
    use_held_out = cfg.held_out_combos and len(colors_all) >= N_CUBES
    if target_color is not None:
        colors = [target_color] + [c for c in colors if c != target_color]
        colors = colors[:N_CUBES]
        colors[0], colors[target] = colors[target], colors[0]
    elif use_held_out and split == "test" and rng.random() < 0.5 and task is None:
        color, chosen_task = HELD_OUT[rng.integers(len(HELD_OUT))]
        colors = [color] + [c for c in colors if c != color]
        colors = colors[:N_CUBES]
        colors[0], colors[target] = colors[target], colors[0]
    elif use_held_out and task is None:  # train/val: resample until the combination is allowed
        while (colors[target], chosen_task) in HELD_OUT:
            base = list(rng.choice(colors_all, size=len(colors_all), replace=False))
            colors = [base[k % len(base)] for k in range(N_CUBES)]
            chosen_task = str(rng.choice(tasks, p=p_t))
    if chosen_task == "stack":
        n, target = 3, 1
        colors = list(STACK_COLORS) + colors[3:]
    lo_xy, hi_xy = cfg.block_half_xy
    lo_z, hi_z = cfg.block_half_z
    half = np.stack([rng.uniform(lo_xy, hi_xy, N_CUBES), rng.uniform(lo_xy, hi_xy, N_CUBES), rng.uniform(lo_z, hi_z, N_CUBES)], axis=1)
    active = np.arange(N_CUBES) < n
    side = {"place_left": "left", "place_right": "right"}.get(chosen_task)
    text = (str(rng.choice(LIFT_TEMPLATES)) if side is None else str(rng.choice(PLACE_TEMPLATES)))
    if chosen_task == "stack":
        text = "stack the green block on the red block, then the blue block on the green block"
    return SceneSample(
        cube_xy=_positions(cfg, rng, n),
        cube_yaw=rng.uniform(-np.pi, np.pi, N_CUBES),
        cube_colors=colors,
        target=target,
        home_noise=rng.normal(0, 0.04, cfg.n_arm),
        light_offset=rng.uniform(-0.4, 0.4, 3) * [1, 1, 0.5],
        light_intensity=float(rng.uniform(0.55, 1.1)),
        cam_pos_jitter=np.clip(rng.normal(0, 0.02, 3), -0.04, 0.04),
        cam_look_jitter=np.clip(rng.normal(0, 0.015, 3), -0.03, 0.03),
        speed_scale=float(rng.uniform(0.7, 1.3)),
        task=chosen_task,
        active=active,
        cube_half=half,
        hover_extra=float(rng.uniform(-0.02, 0.04)),
        instruction=text.format(c=colors[target], side=side),
    )
