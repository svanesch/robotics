# MuJoCo robot-arm data generation

Scripted-expert demonstrations for robot arms in MuJoCo, written as **LeRobot-v2.1-style datasets**
(parquet + mp4) with two cameras (wrist/end-effector and external), proprioception, actions, and a
natural-language instruction per episode. Built to feed neural-network / VLA training later.

Robots: **SO-101** (5 joints + gripper) and **UR5e + Robotiq 2F-85** (6 DoF). Both models are
third-party git submodules (`third_party/`), never copied or edited.

## Setup
```bash
git clone --recurse-submodules <this repo>   # or: git submodule update --init --depth 1
pixi install
```

## Use
```bash
pixi run test                                              # unit + end-to-end checks
pixi run view --robot so101                                # interactive viewer, expert running (needs a display)
pixi run collect --robot so101 --episodes 1500 --cells 2 --workers 4 --noise 0.25   # -> datasets/so101_pick
pixi run inspect-data datasets/so101_v1                    # validate + counts + contact_sheet.png
pixi run python scripts/audit_cameras.py                   # are the blocks/arm/target in view of both cameras?
pixi run train                                             # configs/train/small_vla.yaml -> runs/<run_name>/{best,last}.pt
pixi run evaluate --checkpoint runs/d2/best.pt --split test --episodes 150   # offline error + closed-loop success
pixi run run-policy --checkpoint runs/d2/best.pt --prompt "put the blue block on the left" --video demo/blue_left.mp4
pixi run demo --checkpoint runs/d2/best.pt                 # live viewer; type prompts in the terminal
```

## Dataset (per frame, 30 fps)
Tasks (SO-101): `lift` ("pick up the red block"), `place_left` / `place_right` ("put the blue cube on the left pad").
Per episode: 2-4 blocks (9 colours, jittered sizes), random layout, lighting, external-camera pose, home-pose noise,
expert speed and one of ~10 phrasings per task. Colour x task combinations in `HELD_OUT` appear only in the test split.
`observation.images.{wrist,external}` (mp4, 128x128), `observation.state` (joint positions + gripper in [0,1]),
`action` (absolute joint targets + gripper close command; with `--noise` the executed motion is perturbed while the
recorded label stays the clean expert action, so the data contains recoveries), `observation.ee_pose` and
`observation.environment_state` (privileged block poses; **never given to the policy**), `task_index` (+ text in
`meta/tasks.jsonl`), `next.done`. `meta/episodes.jsonl` holds `task`, `split` (train/val/test, 80/10/10),
`target_color`, `noise`, `success`. Only successful episodes are kept unless `--keep-failures`.

## Small VLA (`src/robotics/policy/`)
ResNet18 (ImageNet weights, stem + layer1 frozen, features at stride 16 -> 8x8 tokens per camera) shared by both cameras + a
small text encoder (closed 36-word vocabulary) whose pooled prompt vector drives FiLM layers in the last two ResNet stages
and a few prompt-conditioned spatial-softmax readouts, + joint state -> transformer encoder -> decoder with 16 learned
queries -> a 16-step action chunk (12.4M parameters). L1 loss on normalised actions, random shifts + colour jitter on the
GPU. At run time every step's chunk is blended with earlier ones (temporal ensembling). `forward(wrist, external, state, tokens)`
cannot take simulator state, and `tests/test_policy.py` asserts that the loader only yields whitelisted keys.

## DAgger (fixing compounding errors)
A policy trained on expert states only drifts into states it never saw. `scripts/collect_dagger.py` lets the current policy
drive the arm for a random part of each episode (or all of it), while the expert re-plans from the actual state
(`PickExpert.sync()`) and labels every step; the expert then takes over to finish. Fine-tune on the original data plus the
DAgger data (`configs/train/dagger_ft.yaml`, `train.py --init-from`), repeat with the new policy.

```bash
pixi run python scripts/collect_dagger.py --checkpoint runs/r4/best.pt --episodes 220 --cells 10 --out datasets/dagger_1
pixi run train --config configs/train/dagger_ft.yaml --init-from runs/r4/best.pt
pixi run evaluate --checkpoint runs/d1/best.pt --split test --episodes 150
pixi run python scripts/language_probe.py --checkpoint runs/d1/best.pt --scenes 30   # does the arm follow the colour in the prompt?
```

## Results (details and caveats in `docs/results_phase1.md`)
Closed-loop success on 150 held-out test scenes, SO-101 in simulation, 128 px images:

| policy | success | lift | place-left | place-right |
|---|---|---|---|---|
| r4: trained on 1500 expert episodes | 10% | 23% | 3% | 5% |
| d1: + 229 DAgger episodes | 22% | 32% | 19% | 16% |
| d2: + 159 more DAgger episodes (policy d1) | 25% | 38% | 27% | 9% |

Known limits: the policy does **not** follow the colour in the prompt (matched-scene probe: 47% vs 49% chance; it
picks a block by geometry), and absolute success is low (25% at best). See the docs for the analysis and the recommended next steps.

## Design
* **Robot = YAML** (`configs/robots/*.yaml`): MJCF path, joints/actuators, gripper (optionally attached
  from another model), grasp-point offset, wrist-camera mount, workspace. Code never hardcodes joint names.
  Add a robot: add the submodule/model + a YAML; `--robot <name>` works everywhere.
* **Multi-robot scenes**: `MjSpec.attach` builds N independent cells (robot + table + cubes + cameras)
  in one model, stepped and rendered together (`--cells`); `--workers` adds process-level parallelism
  with shards merged into one dataset. Rendering dominates cost (about 1000 frames/s per process at 128px on the dev machine's Quadro P1000).
* **Expert**: waypoint state machine (approach, descend, grasp, lift) using privileged cube pose and
  damped-least-squares IK (`src/robotics/sim/kinematics.py`), with joint-rate limiting.

## Notes
* The SO-101 only has 5 arm joints, so IK tracks position plus a partial orientation target.
* Gripper friction is set high on purpose: the stock jaw geometry does not hold blocks at realistic values.

## Adding a robot
1. **Get the model** as a submodule (keeps third-party files untouched), e.g.
   `git submodule add --depth 1 <url> third_party/<name>`. Menagerie already has xArm, Franka, Kinova, ...
2. **Copy a config** (`configs/robots/ur5e.yaml` for a robot + separate gripper, `so101.yaml` for a robot with a built-in gripper) to `configs/robots/<name>.yaml` and fill in:
   * `mjcf`, `arm_joints`, `arm_actuators`, `home_qpos`. Names are the MJCF's own, un-prefixed.
   * `gripper`: actuator, open/closed ctrl values, a joint to read the gripper state from. For a separate
     gripper also `mjcf`, `attach_site` (on the arm) and `attach_body` (in the gripper file). Its names get `attach_prefix`.
   * `ee_site` (a site between the fingertips), `tool_axis` (axis pointing along the fingers) and `grasp_axis`
     (axis the fingers close along), both in the site frame; `grasp_offset` shifts the grasp point.
   * `wrist_cam` (body, pose and fov), `external_cam_pos`/`external_cam_lookat`, `cube_half_size`,
     `workspace_x/y` (where cubes spawn, relative to the base), `table_half`, `table_center` and `table_color` (fixed per robot), `cell_spacing`.
   * `ready_pos` (start pose via IK); `gripper_bodies` + `gripper_friction` if the grasp slips.
3. **Check it**: add the name to `ROBOTS` in `tests/test_sim.py`, run `pixi run test`,
   `pixi run python scripts/demo_video.py --robot <name>` and `pixi run view --robot <name>`.
   The usual tuning is `grasp_offset`, `grasp_height`, `approach_height` and the friction values. Tracing the expert's
   phases and checking the wrist camera in the demo video shows what is off.

## Performance note
Shadows add one render pass per cell, so they are **off by default** everywhere (about 5x faster with 8 cells).
Use `--shadows` on `collect`, `view` or `benchmark` to enable them. If the laptop lags, lower `--cells` or `--workers`: rendering uses the GPU, not just the CPU.
