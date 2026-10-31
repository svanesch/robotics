"""Run a trained VLA from a text prompt in the simulator.

  pixi run python scripts/run_policy.py --checkpoint runs/<run>/best.pt --prompt "pick up the red block" --video demo/red.mp4
  pixi run demo --checkpoint runs/<run>/best.pt --interactive          # live viewer, type prompts in the terminal
"""
import argparse

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import torch

from robotics.envs.multi_cell import MultiCellEnv
from robotics.policy.rollout import PolicyRunner, TemporalEnsemble, parse_prompt, run_episodes, scene_for_prompt
from robotics.policy.train_utils import load_policy, robot_for_checkpoint
from robotics.robots.spec import load_robot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--robot", default=None, help="default: inferred from the checkpoint")
    p.add_argument("--prompt", default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", default=None, help="write an mp4 of the first episode (external + wrist + free camera)")
    p.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (needs a display; run via `pixi run demo`)")
    p.add_argument("--interactive", action="store_true", help="read prompts from stdin and execute them one after another (implies --viewer)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if not (a.prompt or a.interactive):
        p.error("give --prompt or --interactive")
    model, ck = load_policy(a.checkpoint, a.device)
    cfg = load_robot(robot_for_checkpoint(ck, a.robot))
    runner = PolicyRunner(model, ck, a.device)
    rng = np.random.default_rng(a.seed)
    size = model.cfg.image_size
    env = MultiCellEnv(cfg, 1, size)
    if a.viewer or a.interactive:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = [cfg.workspace_x[0] + 0.1, 0, 0.08]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 140, -22, 0.95
        step_hook = viewer.sync
    else:
        viewer, step_hook = None, None

    def run_prompt(prompt: str) -> None:
        s = scene_for_prompt(cfg, rng, prompt)
        task, color = parse_prompt(prompt)
        print(f"prompt: '{prompt}'  -> scene: task={task}, target colour={color or 'random'}")
        env.reset_cell(0, s)
        tokens = np.stack([runner.tok.encode(prompt)])
        ens = TemporalEnsemble(1, runner.chunk)
        for t in range(560):
            im = env.render(0)
            pred = runner.predict(im["wrist"][None], im["external"][None], env.state(0)[None], tokens)
            act = ens.add(0, t, pred[0])
            act[-1] = np.clip(act[-1], 0, 1)
            env.step(act[None])
            if step_hook:
                step_hook()
            if viewer is not None and not viewer.is_running():
                return
        print("  finished 560 steps")

    if a.interactive:
        print("Type a prompt (e.g. 'put the blue block on the left'), empty line to quit.")
        while viewer.is_running():
            try:
                prompt = input("prompt> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            run_prompt(prompt)
        return
    frames, rec = [], None
    if a.video:
        big = mujoco.Renderer(env.model, 512, 640)
        cam = mujoco.MjvCamera()
        cam.type, cam.lookat, cam.azimuth, cam.elevation, cam.distance = mujoco.mjtCamera.mjCAMERA_FREE, [cfg.workspace_x[0] + 0.1, 0, 0.08], 140, -22, 0.95
        up = lambda img: np.kron(img, np.ones((2, 2, 1), np.uint8))  # noqa: E731

        def rec(im):  # free-camera view on the left, wrist (top) and external (bottom) policy inputs on the right
            big.update_scene(env.data, cam)
            frames.append(np.concatenate([big.render(), np.concatenate([up(im["wrist"]), up(im["external"])], 0)], 1))
    scenes = [scene_for_prompt(cfg, rng, a.prompt) for _ in range(a.episodes)]
    res = run_episodes(runner, env, scenes, record_cell0=rec)
    for r in res:
        print(f"  '{r.instruction}': {'SUCCESS' if r.success else 'fail'} after {r.steps} steps{' (touched another block)' if r.wrong_block else ''}")
    print(f"success rate {np.mean([r.success for r in res]):.0%} over {len(res)} episodes")
    if a.video and frames:
        imageio.mimsave(a.video, frames[::2], fps=15, codec="libx264", macro_block_size=1)
        print("wrote", a.video)


if __name__ == "__main__":
    main()
