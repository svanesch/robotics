"""Generate a LeRobot-style dataset with the scripted expert: pixi run collect --robot so101 --episodes 50"""
import argparse

from robotics.data.collect import collect


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="so101")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--cells", type=int, default=4, help="parallel robot cells per worker process")
    p.add_argument("--workers", type=int, default=1, help="worker processes")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--noise", type=float, default=0.0, help="fraction of episodes with random joint perturbations (recovery data)")
    p.add_argument("--drop", type=float, default=0.0, help="fraction of episodes with a dropped block during the lift: the expert re-plans and picks it up again")
    p.add_argument("--task", default=None, choices=["lift", "place_left", "place_right", "stack"], help="restrict to one task (default: mix)")
    p.add_argument("--shadows", action="store_true", help="render shadows (nicer, but ~5x slower with many cells)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or f"datasets/{a.robot}_pick"
    r = collect(a.robot, out, a.episodes, a.cells, a.workers, a.image_size, a.seed, a.keep_failures, a.shadows, a.noise, a.task, a.drop)
    print(f"saved {r['saved']} episodes ({r['frames']} frames) to {out}; "
          f"expert success rate {r['saved'] / max(r['attempted'], 1):.0%}; {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()
