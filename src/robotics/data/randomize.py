"""Per-episode scene randomisation, tasks and language instructions."""
from __future__ import annotations

import numpy as np

from robotics.envs.multi_cell import SceneSample
from robotics.robots.spec import RobotConfig
from robotics.sim.builder import CUBE_COLORS, N_CUBES

TASKS = ("lift", "place_left", "place_right")
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

# colour x task combinations never seen in train/val; the test split uses them half of the time
HELD_OUT = (("pink", "place_right"), ("cyan", "place_left"), ("orange", "place_left"), ("purple", "lift"))
MIN_ZONE_DIST = 0.11  # keep every block this far from both pad centres
DISTRACTOR_PAD = (0.03, 0.03)  # how far distractors may extend beyond the target workspace (x, y)


def _positions(cfg: RobotConfig, rng: np.random.Generator, n: int, target: int) -> np.ndarray:
    min_dist = cfg.min_object_dist or 0.1
    zones = [np.array([cfg.zone_xy[0], s * cfg.zone_xy[1]]) for s in (1, -1)]
    order = [target] + [k for k in range(n) if k != target]  # place the target first, it needs the clean spot
    for _ in range(100):  # rejection sampling; restart the whole layout if a block finds no free spot
        pts: list[np.ndarray] = []
        chosen = {}
        for k in order:
            px, py = (0.0, 0.0) if k == target else DISTRACTOR_PAD
            for _ in range(300):
                p = np.array([rng.uniform(cfg.workspace_x[0] - px, cfg.workspace_x[1] + px),
                              rng.uniform(cfg.workspace_y[0] - py, cfg.workspace_y[1] + py)])
                if all(np.linalg.norm(p - q) > min_dist for q in pts) and all(np.linalg.norm(p - z) > MIN_ZONE_DIST for z in zones):
                    pts.append(p)
                    chosen[k] = p
                    break
            else:
                break
        if len(chosen) == len(order):
            break
    else:
        raise RuntimeError("could not place blocks")
    out = np.zeros((N_CUBES, 2))
    for k, p in chosen.items():
        out[k] = p
    return out


def sample_scene(cfg: RobotConfig, rng: np.random.Generator, split: str = "train", task: str | None = None,
                 target_color: str | None = None) -> SceneSample:
    """split="test" draws held-out colour x task combos half of the time; train/val never show them."""
    colors_all = list(CUBE_COLORS)
    n = int(rng.choice([2, 3, 4], p=[0.25, 0.45, 0.30]))
    target = int(rng.integers(n))
    chosen_task = task or str(rng.choice(TASKS, p=TASK_WEIGHTS))
    colors = list(rng.choice(colors_all, size=N_CUBES, replace=False))
    if target_color is not None:
        colors = [target_color] + [c for c in colors if c != target_color]
        colors = colors[:N_CUBES]
        target = int(rng.integers(n))
        colors[0], colors[target] = colors[target], colors[0]
    elif split == "test" and rng.random() < 0.5 and task is None:
        color, chosen_task = HELD_OUT[rng.integers(len(HELD_OUT))]
        colors = [color] + [c for c in colors if c != color]
        colors = colors[:N_CUBES]
        colors[0], colors[target] = colors[target], colors[0]
    elif task is None:  # train/val: resample until the combination is allowed
        while (colors[target], chosen_task) in HELD_OUT:
            colors = list(rng.choice(colors_all, size=N_CUBES, replace=False))
            chosen_task = str(rng.choice(TASKS, p=TASK_WEIGHTS))
    half = np.stack([rng.uniform(0.0115, 0.0145, N_CUBES), rng.uniform(0.0115, 0.0145, N_CUBES),
                     rng.uniform(0.016, 0.026, N_CUBES)], axis=1)
    active = np.arange(N_CUBES) < n
    side = {"place_left": "left", "place_right": "right"}.get(chosen_task)
    text = (str(rng.choice(LIFT_TEMPLATES)) if side is None else str(rng.choice(PLACE_TEMPLATES)))
    return SceneSample(
        cube_xy=_positions(cfg, rng, n, target),
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
