"""Unattended DAgger loop with model selection on validation scenes (pixi run python scripts/auto_dagger.py).

  baseline = closed-loop success of --start-run on the VALIDATION scenes (waits for runs/<run>/eval_val_val100.json)
  for each round: collect DAgger data with the best policy so far -> fine-tune on expert + all DAgger data -> evaluate on val
  keep the best round (by val success); stop when the target is reached, or when 2 rounds in a row fail to beat the best
  finally: evaluate the start run and the best run ONCE on the TEST scenes, record test videos, write the summary.

Everything is logged to runs/auto/ and shown on the dashboard as the job "overnight-dagger".
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
LOG = ROOT / "runs" / "auto" / "auto_dagger.log"
SUMMARY = ROOT / "runs" / "auto_dagger_summary.md"


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(args: list[str], name: str) -> None:
    log = ROOT / "runs" / "auto" / f"{name}.log"
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{name} failed with exit code {r.returncode}; see {log}")


def eval_result(run_name: str, split: str, tag: str) -> tuple[int, int]:
    j = json.loads((ROOT / "runs" / run_name / f"eval_{split}{tag}.json").read_text())
    s, n = j["closed_loop"]["all"]
    return s, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-run", default="ur5e_lift_a")
    p.add_argument("--base-dataset", default="datasets/ur5e_lift_v1")
    p.add_argument("--extra-dataset", nargs="*", default=[], help="more expert data mixed into every round (e.g. dropped-block / re-pick episodes)")
    p.add_argument("--extra-weight", type=float, default=1.5)
    p.add_argument("--base-config", default="configs/train/ur5e_lift.yaml")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=200, help="DAgger episodes per round")
    p.add_argument("--steps", type=int, default=4000, help="fine-tuning steps per round")
    p.add_argument("--val-episodes", type=int, default=100)
    p.add_argument("--test-episodes", type=int, default=150)
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--target", type=float, default=0.90, help="stop when the validation success reaches this")
    p.add_argument("--wait-hours", type=float, default=3.0, help="how long to wait for the baseline evaluation")
    a = p.parse_args()

    total = a.rounds * 4 + 4
    rep = StatusReporter("auto", "overnight-dagger", total=total, meta={"start": a.start_run, "rounds": a.rounds, "episodes": a.episodes})
    done_steps = 0

    def stage(text: str, **m):
        nonlocal done_steps
        rep.update(progress=done_steps, stage=text, **m)
        say(text)

    summary = ["# Overnight DAgger loop", "", "| round | run | val success | note |", "|---|---|---|---|"]
    rows = []

    def write_summary(extra: list[str] | None = None):
        SUMMARY.write_text("\n".join(summary + (extra or [])) + "\n")

    try:
        # ---- baseline on the validation scenes
        base_json = ROOT / "runs" / a.start_run / "eval_val_val100.json"
        t0 = time.time()
        stage(f"waiting for the baseline evaluation of {a.start_run}")
        while not base_json.exists():
            if time.time() - t0 > a.wait_hours * 3600:
                raise RuntimeError("baseline evaluation never appeared")
            time.sleep(30)
        s, n = eval_result(a.start_run, "val", "_val100")
        best_run, best = a.start_run, s / n
        rows.append(("0", a.start_run, f"{s}/{n} = {100 * s / n:.0f}%", "baseline"))
        summary.append(f"| 0 | {a.start_run} | {s}/{n} = {100 * s / n:.0f}% | baseline |")
        write_summary()
        done_steps += 1
        dagger_sets, bad = [], 0
        for i in range(1, a.rounds + 1):
            if best >= a.target:
                say(f"target {a.target:.0%} reached by {best_run}; no more DAgger rounds")
                break
            if bad >= 2:
                say("two rounds in a row without improvement: stopping the DAgger rounds")
                break
            out = f"datasets/ur5e_dagger_{i}"
            stage(f"round {i}: collecting {a.episodes} DAgger episodes with {best_run}", round=i, best_val_success=best)
            if (ROOT / out / "meta" / "info.json").exists():
                say(f"round {i}: reusing the existing DAgger data in {out}")
            else:
                run(["scripts/collect_dagger.py", "--checkpoint", f"runs/{best_run}/best.pt", "--out", out, "--episodes", str(a.episodes),
                     "--cells", str(a.cells), "--seed", str(7000 + i), "--max-steps", "480"], f"collect_round{i}")
            dagger_sets.append(out)
            done_steps += 1
            cfg = yaml.safe_load((ROOT / a.base_config).read_text())
            cfg.update(dataset=[a.base_dataset, *a.extra_dataset, *dagger_sets],
                       root_weights=[1.0] + [a.extra_weight] * len(a.extra_dataset) + [3.0] * len(dagger_sets), run_name=f"ur5e_d{i}",
                       steps=a.steps, lr=5.0e-5, backbone_lr=1.0e-5, warmup_steps=200, seed=i, eval_every=1000)
            cfg_path = ROOT / "runs" / "auto" / f"ur5e_d{i}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            stage(f"round {i}: fine-tuning from {best_run} on expert{" + re-pick" if a.extra_dataset else ""} + {len(dagger_sets)} DAgger sets", round=i, best_val_success=best)
            run(["scripts/train.py", "--config", str(cfg_path.relative_to(ROOT)), "--init-from", f"runs/{best_run}/best.pt"], f"train_round{i}")
            done_steps += 1
            stage(f"round {i}: evaluating ur5e_d{i} on {a.val_episodes} validation scenes", round=i, best_val_success=best)
            run(["scripts/evaluate.py", "--checkpoint", f"runs/ur5e_d{i}/best.pt", "--split", "val", "--episodes", str(a.val_episodes),
                 "--cells", str(a.cells), "--tag", "_val100"], f"eval_round{i}")
            done_steps += 1
            s, n = eval_result(f"ur5e_d{i}", "val", "_val100")
            succ = s / n
            if succ > best:
                note, best, best_run, bad = f"new best (was {100 * best:.0f}%)", succ, f"ur5e_d{i}", 0
            else:
                note, bad = f"not better than {best_run} ({100 * best:.0f}%)", bad + 1
            summary.append(f"| {i} | ur5e_d{i} | {s}/{n} = {100 * succ:.0f}% | {note} |")
            write_summary()
            say(f"round {i}: val success {s}/{n} = {100 * succ:.0f}%  ({note})")

        # ---- final: one test evaluation each for the start run and the selected run, videos, views
        finals = [best_run] + ([a.start_run] if best_run != a.start_run else [])
        for r in finals:
            stage(f"final: test evaluation of {r} on {a.test_episodes} scenes", best_run=best_run)
            run(["scripts/evaluate.py", "--checkpoint", f"runs/{r}/best.pt", "--split", "test", "--episodes", str(a.test_episodes),
                 "--cells", str(a.cells), "--tag", "_150"], f"eval_test_{r}")
            s, n = eval_result(r, "test", "_150")
            summary.append(f"| test | {r} | {s}/{n} = {100 * s / n:.0f}% | single test run |")
            write_summary()
            run(["scripts/make_views.py", "--run", r], f"views_{r}")
            stage(f"final: recording test videos of {r}", best_run=best_run)
            run(["scripts/record_videos.py", "--run", r, "--episodes", "8"], f"videos_{r}")
        done_steps = total
        write_summary([f"", f"Selected on validation scenes: **{best_run}** (val success {100 * best:.0f}%)."])
        rep.finish("done", best_run=best_run, best_val_success=best)
        say(f"finished: selected {best_run}")
    except Exception as e:  # leave a readable trail for the morning
        say(f"STOPPED: {e}")
        write_summary(["", f"Stopped early: {e}"])
        rep.finish("stopped", error=str(e)[:200])
        raise


if __name__ == "__main__":
    main()
