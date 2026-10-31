# Plan: stacking on the UR5e (written Oct 31, overnight)

Goal: from three blocks on the table, build a tower of three: **green on red, then blue on top of green**. That is two picks and two places,
the red block is the base and does not move. No text prompt; the order is a fixed rule the policy can read from the image
(colours red / green / blue, and what is already stacked). No special stacking pad.

Starting point: pick-and-lift model `ur5e_lift_e1` (binary open/close gripper, BCE on the gripper decision, one DAgger round):
79% validation, 78% test (117/150). Gate for stacking was relaxed by the owner to about 75%, so the gate is passed.

Rules for the night
- Disk: when less than 15 GB is free, delete only old datasets inside the repo (`datasets/`), nothing else.
- Model selection on validation scenes, one test run per milestone for the reported number.
- A second DAgger round that does not help is no reason to stop; carry on with the best model.
- Every long job reports to the dashboard; results go to `docs/results_stack.md` and to the "Stacking" chapter of the artifact.
- No push to `main`; commits go to the `test` branch.

## S0 - Close the pick-and-lift chapter (about 1 h)
Artifact update with the final lift numbers, tests for the binary gripper and the BCE loss, full test suite, commit and push to `test`.

## S1 - Stack scene and expert (about 2 h)
- Task `stack` in the scene sampler: three blocks, fixed colours by index (0 red = base, 1 green, 2 blue), random positions and yaws.
- `PickExpert` with stages: stage 0 picks green and puts it on red, stage 1 picks blue and puts it on green. Place height comes from the
  measured block heights, the arm retracts upward after each release, `sync()` infers the stage from the block poses (needed for DAgger and re-picks).
- Success: green rests on red and blue rests on green (xy within 2.5 cm, heights as for resting blocks), gripper open, held for 12 steps.
- Checks: expert succeeds >= 90% over 40+ scenes (parallel cells), unit test, camera audit (all three blocks and the tower visible in the base camera,
  gripper in the wrist camera).

## S2 - Expert data and first policy (about 3 h)
- 800 expert episodes (25% with joint noise, some with a dropped block that is picked up again), 4 workers.
- Fine-tune from `ur5e_lift_e1` (same architecture), 10k steps, binary gripper with BCE.
- Evaluation: success of the full tower and per stage (green on red, then blue on green), funnel (reach, grasp, lift, place), failure classification.

## S3 - DAgger rounds (about 4 h)
- Two to three rounds of about 200 episodes with the best policy driving and the expert labelling, fine-tune on all data, evaluate on the same validation scenes.
- Keep the best model by validation success. If a round does not help, continue with the best model.

## S4 - Final evaluation and write-up (about 1.5 h)
- One test run on 150 scenes, per-stage success, failure classes.
- Test videos for the dashboard, views, `docs/results_stack.md`, the stacking chapter in the artifact, commit and push to `test`.

## Risks
- A tower of three is a precision task; the second place (blue on green, 8 cm high) may fail more often than the first.
- The policy must pick the right colour and the right target without a prompt, from the image alone.
- Dropped or toppled blocks produce states the scripted expert cannot label; those are cut and counted.
