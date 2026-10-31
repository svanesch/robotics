"""DAgger collection: the policy drives, the expert labels. pixi run python scripts/collect_dagger.py --checkpoint runs/r4/best.pt --episodes 300 --out datasets/dagger_1"""
import argparse

import torch

from robotics.data.dagger import collect_dagger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--robot", default=None, help="default: inferred from the checkpoint")
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--policy-prob-full", type=float, default=0.3, help="fraction of episodes the policy drives from start to end")
    p.add_argument("--max-steps", type=int, default=480)
    p.add_argument("--ens-m", type=float, default=0.01)
    p.add_argument("--replan", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    s = collect_dagger(a.checkpoint, a.out, a.episodes, a.cells, a.seed, a.policy_prob_full, max_steps=a.max_steps,
                       device=a.device, ens_m=a.ens_m, replan=a.replan, robot=a.robot)
    print(f"saved {s['episodes']} episodes / {s['frames']} frames ({s['policy_frames']} policy-driven) to {a.out}; "
          f"expert success at the end {s['expert_success']}/{s['episodes']}; cut for no valid label {s['cut_unconfident']}; {s['seconds']:.0f}s")


if __name__ == "__main__":
    main()
