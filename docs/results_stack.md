# Stacking on the UR5e (night of Oct 31 / Nov 1)

Task: green block on red, then blue on green (three blocks, no text prompt, no stacking pad). Simulation only.

## Result
Best model `ur5e_stack_e` (chosen on 100 validation scenes): **test, 150 scenes: first block on the base 98/150 (65%), tower of three 21/150 (14%)**.
Grasped the target in 148/150, within 4 cm in 149/150, never touched a wrong block.

## Runs (100 validation scenes, `--task stack`)
| Run | What changed | First block | Tower |
|---|---|---|---|
| stack_a | 800 expert episodes, fine-tuned from ur5e_lift_e1 | 44% | 14% |
| stack_b | + DAgger round 1 (207 episodes) | 57% | 14% |
| stack_c | + DAgger round 2 | 36% | 3% |
| stack_d | stack_b, 12k steps, lr 5e-5 | 54% | 13% |
| stack_e | stack_b + 5x sampling weight around the release (`release_weight`) | 70% | 14% |
| stack_f | stack_e + DAgger round 3 | 71% | 6% |

## Findings
- Grasping and reaching work; placing does not. stack_b (48 scenes): 42% lifted green but never placed it on red, 25% had green on red but never picked up blue.
- Expert success is only ~62-72% (tower collapse on release); only successful episodes are stored. `noslip_iterations: 10` in `configs/robots/ur5e.yaml` raised stack success (9/16 to 13/16) but also changes the physics of the lift task (old lift checkpoints were trained without it).
- DAgger is weak here: the expert could finish only 10-19 of 207 policy-driven episodes, 90-107 were cut for lack of a valid label.
- The release weighting helped the first place (57% to 70%), not the second.

## Next
Expert data focused on the placement phases, a sturdier expert, labels for toppled-block states, a stage signal or longer chunk for the policy.

Videos: `runs/ur5e_stack_e/videos/` (side view; the old view looked through the cell's back wall). Tools added: `scripts/diagnose_stack.py`, `--task` for `collect.py`, `collect_dagger.py`, `evaluate.py`, `record_videos.py`; `release_weight` in `scripts/train.py`.
