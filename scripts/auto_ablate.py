"""Unattended ablation: which training data hurts or helps? (pixi run python scripts/auto_ablate.py)

Fine-tunes the SAME start policy with different data mixes and evaluates every variant on the SAME validation scenes, so the
numbers are directly comparable. Results go to runs/auto/ablation.md and the dashboard job "overnight-ablation".
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from robotics.status import StatusReporter

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "auto" / "ablation.log"
OUT = ROOT / "runs" / "auto" / "ablation.md"


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(args: list[str], name: str) -> None:
    with open(ROOT / "runs" / "auto" / f"{name}.log", "w") as f:
        r = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{name} failed with exit code {r.returncode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-run", default="ur5e_lift_a")
    p.add_argument("--base-dataset", default="datasets/ur5e_lift_v1")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--val-episodes", type=int, default=100)
    p.add_argument("--cells", type=int, default=8)
    a = p.parse_args()
    variants = {  # name -> (extra datasets, their sampling weights)
        "C_expert_only": ([], []),
        "A_dagger_only": (["datasets/ur5e_dagger_1"], [3.0]),
        "B_repick_only": (["datasets/ur5e_drop_v1"], [1.5]),
    }
    rep = StatusReporter("auto", "overnight-ablation", total=len(variants) * 2, meta={"start": a.start_run})
    j = json.loads((ROOT / "runs" / a.start_run / "eval_val_val100.json").read_text())
    s0, n0 = j["closed_loop"]["all"]
    rows = ["# Ablation on the validation scenes", "", f"All variants fine-tune {a.start_run} for {a.steps} steps and are evaluated on the same {a.val_episodes} validation scenes.", "",
            "| variant | data | val success |", "|---|---|---|", f"| {a.start_run} (start) | - | {s0}/{n0} = {100 * s0 / n0:.0f}% |"]
    OUT.write_text("\n".join(rows) + "\n")
    done = 0
    for name, (extra, weights) in variants.items():
        run_name = f"ur5e_abl_{name[0]}"
        cfg = yaml.safe_load((ROOT / "configs/train/ur5e_lift.yaml").read_text())
        cfg.update(dataset=[a.base_dataset, *extra], root_weights=[1.0, *weights], run_name=run_name, steps=a.steps, lr=5.0e-5,
                   backbone_lr=1.0e-5, warmup_steps=200, seed=3, eval_every=1000)
        path = ROOT / "runs" / "auto" / f"{run_name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        rep.update(progress=done, stage=f"training {name}")
        say(f"{name}: fine-tuning {a.start_run} on {[a.base_dataset, *extra]}")
        run(["scripts/train.py", "--config", str(path.relative_to(ROOT)), "--init-from", f"runs/{a.start_run}/best.pt"], f"train_{run_name}")
        done += 1
        rep.update(progress=done, stage=f"evaluating {name}")
        say(f"{name}: evaluating on {a.val_episodes} validation scenes")
        run(["scripts/evaluate.py", "--checkpoint", f"runs/{run_name}/best.pt", "--split", "val", "--episodes", str(a.val_episodes),
             "--cells", str(a.cells), "--tag", "_val100"], f"eval_{run_name}")
        done += 1
        s, n = json.loads((ROOT / "runs" / run_name / "eval_val_val100.json").read_text())["closed_loop"]["all"]
        say(f"{name}: val success {s}/{n} = {100 * s / n:.0f}%")
        with open(OUT, "a") as f:
            f.write(f"| {run_name} ({name}) | {', '.join([a.base_dataset, *extra])} | {s}/{n} = {100 * s / n:.0f}% |\n")
        rep.update(progress=done, last_result=f"{name}: {100 * s / n:.0f}%")
    rep.finish("done")
    say("ablation finished")


if __name__ == "__main__":
    main()
