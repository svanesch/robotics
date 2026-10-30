import numpy as np
import pytest

from robotics.data.expert import PickExpert
from robotics.data.randomize import sample_scene
from robotics.envs.multi_cell import MultiCellEnv
from robotics.robots.spec import load_robot
from robotics.sim.kinematics import ArmIK

ROBOTS = ["so101", "ur5e"]


@pytest.fixture(params=ROBOTS)
def cfg(request):
    return load_robot(request.param)


def test_world_builds_for_n_cells(cfg):
    env = MultiCellEnv(cfg, 3, 64)
    assert env.model.ncam == 6 and len(env.cells) == 3


def test_cameras_render_distinct_views(cfg):
    env = MultiCellEnv(cfg, 2, 64)
    rng = np.random.default_rng(0)
    for i in range(2):
        env.reset_cell(i, sample_scene(cfg, rng))
    im = env.render(0)
    assert im["wrist"].shape == (64, 64, 3) and im["external"].shape == (64, 64, 3)
    assert im["wrist"].std() > 1 and im["external"].std() > 1
    assert not np.array_equal(im["wrist"], im["external"])


def test_ik_reaches_hover_target(cfg):
    ik = ArmIK(cfg)
    env = MultiCellEnv(cfg, 1, 32, render=False)
    tool, grasp = ik.tool_dirs(np.array([1.0, 0.0]), 0.0)
    target = np.array([np.mean(cfg.workspace_x), 0.0, 0.06])
    q = ik.solve(env.home, target, tool, grasp, iters=200)
    p, _ = ik.fk(q)
    assert np.linalg.norm(p - target) < 0.01


def test_expert_succeeds_in_parallel_cells(cfg):
    n = 3
    env = MultiCellEnv(cfg, n, 32, render=False)
    ik, rng = ArmIK(cfg), np.random.default_rng(1)
    experts = []
    for i in range(n):
        env.reset_cell(i, sample_scene(cfg, rng))
        experts.append(PickExpert(env, ik, i))
    result = [None] * n
    for _ in range(300):
        acts = []
        for i, e in enumerate(experts):
            a, done = e.act()
            acts.append(a)
            if done and result[i] is None:
                result[i] = e.success
        env.step(np.stack(acts))
        if all(r is not None for r in result):
            break
    assert sum(bool(r) for r in result) >= n - 1, result


def test_camera_audit_so101():
    """External cam keeps robot + blocks in frame; wrist cam keeps the target visible (scripts/audit_cameras.py)."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "scripts/audit_cameras.py", "--episodes", "8", "--seed", "5"],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:]


@pytest.mark.parametrize("task", ["place_left", "place_right", "lift"])
def test_so101_expert_tasks(task):
    cfg = load_robot("so101")
    n = 4
    env = MultiCellEnv(cfg, n, 32, render=False)
    ik, rng = ArmIK(cfg), np.random.default_rng(21)
    experts = []
    for i in range(n):
        env.reset_cell(i, sample_scene(cfg, rng, task=task))
        experts.append(PickExpert(env, ik, i))
    result = [None] * n
    for _ in range(600):
        acts = []
        for i, e in enumerate(experts):
            a, done = e.act()
            acts.append(a)
            if done and result[i] is None:
                result[i] = e.success
        env.step(np.stack(acts))
        if all(r is not None for r in result):
            break
    assert sum(bool(r) for r in result) >= n - 1, (task, result)


def test_scene_sampler_respects_held_out_and_layout():
    from robotics.data.randomize import HELD_OUT, MIN_ZONE_DIST

    cfg = load_robot("so101")
    rng = np.random.default_rng(0)
    seen_held_out = {"train": 0, "test": 0}
    for split in ("train", "test"):
        for _ in range(150):
            s = sample_scene(cfg, rng, split=split)
            seen_held_out[split] += (s.cube_colors[s.target], s.task) in HELD_OUT
            pts = s.cube_xy[s.active]
            assert len(set(s.cube_colors[: len(s.active)])) == len(s.cube_colors)
            for z in ([cfg.zone_xy[0], cfg.zone_xy[1]], [cfg.zone_xy[0], -cfg.zone_xy[1]]):
                assert (np.linalg.norm(pts - z, axis=1) > MIN_ZONE_DIST - 1e-9).all()
    assert seen_held_out["train"] == 0 and seen_held_out["test"] > 20


@pytest.mark.parametrize("task", ["lift", "place_left"])
def test_expert_resumes_after_handover(task):
    """A fresh expert that only calls sync() every step can finish an episode started by another controller."""
    cfg = load_robot("so101")
    n = 3
    env = MultiCellEnv(cfg, n, 32, render=False)
    ik, rng = ArmIK(cfg), np.random.default_rng(9)
    first = []
    for i in range(n):
        env.reset_cell(i, sample_scene(cfg, rng, task=task))
        first.append(PickExpert(env, ik, i))
    for _ in range(70):  # the "other controller" (here the expert itself) gets part-way, then hands over
        env.step(np.stack([e.act()[0] for e in first]))
    second = [PickExpert(env, ik, i) for i in range(n)]
    result = [None] * n
    for _ in range(500):
        acts = []
        for i, e in enumerate(second):
            e.sync()
            a, done = e.act()
            acts.append(a)
            if done and result[i] is None:
                result[i] = e.success
        env.step(np.stack(acts))
        if all(r is not None for r in result):
            break
    assert sum(bool(r) for r in result) >= n - 1, result
