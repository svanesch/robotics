# UR5e pick-and-lift without a text prompt (written overnight, Oct 30-31)

Setup: UR5e + Robotiq 2F-85 in MuJoCo, base-mounted external camera + wrist camera, 128 px, one 4 cm cube per scene (3 colours), ACT-style policy
(ResNet18 + transformer, 16-step action chunks), 800 expert episodes (`datasets/ur5e_lift_v1`). Success = block lifted 4 cm and held for 12 steps.
Model selection on 100 validation scenes, one run on 150 test scenes at the end. About +-10 points of noise per validation number.

## Results

| run | what | val (100) | test (150) |
|---|---|---|---|
| ur5e_lift_a | 10k steps on the expert data | 61% | - |
| ur5e_abl_C | + 3k steps at low learning rate, expert data only | 68% | 65% |
| ur5e_d1 | + corrected DAgger round 1 (207 episodes) | **69%** | **69%** (104/150) |
| ur5e_d2 / d3 | DAgger rounds 2 and 3 | 56% / 53% | - |
| ur5e_lift_b | 30k steps from scratch | 55% | - |
| ur5e_abl_B | + 400 dropped-block / re-pick episodes | 62% | - |
| ur5e_abl_A | + DAgger data with the label defect (below) | 46% | - |
| ur5e_lift_d | binary open/close gripper command (new data `ur5e_lift_v3`, 800 episodes, 99% expert success, 10k steps) | 48% | - |
| ur5e_lift_e | binary command + BCE loss on the gripper (`grip_bce`), same data v3 | 54% | - |
| ur5e_lift_e1 | + one DAgger round on ur5e_lift_e (207 episodes, binary gripper), 4k steps | **79%** | **78%** (117/150) |
| ur5e_lift_c | NEW expert data with a short grasp wait (grip_rate 0.08, settle 4; episodes 93 instead of 165 steps) | 33% | - |

The 80% gate of the plan was NOT reached; stacking was therefore not started.

## What the nights' experiments showed
1. **A label defect made the first DAgger round harmful.** In the DAgger data the expert said "open the gripper" in 48% of the frames where the block was held in the air
   (the "block is held" test needed the gripper to be more than half closed; a policy often closes it less). After the fix 0% of such labels. Rounds 1 to 3 with correct labels did not help:
   69%, 56%, 53% (all within the noise of each other and of the 61-68% of the models without DAgger).
2. **More training did not help.** 30k steps gave a lower validation loss (0.054 vs 0.059) but 55% success. The validation loss does not predict closed-loop success.
3. **Where the failures are (ur5e_d1, test):** the gripper gets within 4 cm of the block in 146/150 scenes, closes on it in 142/150, but the block is lifted in only 106/150.
   Playing 48 test scenes and classifying them: 33 success, **13 "gripper closed but the arm never rose"**, 1 lost the block after lifting, 1 never closed.
4. The mean gripper closure while holding the 4 cm cube is about 0.6, the policy often closes less.

## Experiment: shorten the wait after grasping (ur5e_lift_c, Oct 31)
Hypothesis: the expert waits ~76 steps after the gripper closes (fixed timer, nearly identical images), the policy cannot tell when the wait is over and stalls.
Change: the expert closes the gripper in ~12 steps and lifts 4 steps later; 800 new episodes (expert success 97%), same training recipe as ur5e_lift_a.
Result: **worse, 33% on validation (ur5e_lift_a: 61%)**.
- The stall class disappeared (0 of 48 classified test scenes, before: 13), so the stall WAS tied to the wait.
- But two new failures appeared: 40% "arm rose, block stayed" (the policy lifts a few steps after closing, before the grip holds, or closes on the edge) and 15% never closed;
  the gripper got within 4 cm of the block in only 61/100 validation scenes (before 97%) and "closed on nothing" went from 8 to 51: the policy now closes the gripper too early, while still approaching.
- Reading: removing the long wait over-corrected. The wait does a job (it lets the grip build up) and the shorter, faster closing makes the close/lift timing much harder to learn.
  A middle setting (e.g. grip_rate 0.03, settle 10) or giving the policy memory (previous action / two frames) are the untested options. Not proven which.

## Experiment: binary open/close gripper (ur5e_lift_d, Oct 31)
Change: the gripper command is only open or closed (`gripper_binary`); the expert lifts when the jaws have stopped moving instead of after a timer.
Result: **48% on validation (ur5e_lift_a 61%, ur5e_d1 69%)**, within 4 cm of the block 76/100, grasped 65, lifted 51.
- Stalls ("closed, arm never rose") are gone: 0 of 48 classified scenes (before 13).
- But 14/48 "never closed the gripper" (the policy hovers near the block without issuing close) and 11/48 "arm rose, block stayed".
- Reading: removing the hidden timer removed the stall, but a hard 0/1 command is harder to learn at the right moment (closing decision is a sharp switch, L1 on a step function). Not yet proven; the policy output is thresholded at 0.5, so near-0.5 outputs may flip.
- Untested options: classification (BCE) loss on the gripper dimension, previous action / two frames as input, middle wait.

## Experiment: BCE on the binary gripper (ur5e_lift_e, Oct 31)
Same data as ur5e_lift_d; the gripper output is a logit trained with binary cross-entropy (no value in between), joints stay L1.
Result: **54% on validation** (d: 48%, a: 61%, d1: 69%; all within noise of each other except d1). Within 4 cm 88/100, grasped 58, lifted 55.
Classified 48 scenes: 27 success, 11 never closed, 8 arm rose but block stayed, 2 lost after lifting. Stalls remain gone.
Reading: the stall is solved but the policy still hovers near the block without issuing "close" in about a quarter of the scenes. These hovering states are rare in expert data, which is what DAgger addresses (next: DAgger round on ur5e_lift_e).

## Experiment: DAgger on the binary gripper (ur5e_lift_e1, Oct 31)
The policy ur5e_lift_e drove 207 episodes (29.5k policy-driven frames, expert success at the end 179/207, 12 episodes cut for no valid label); fine-tuned on v3 + these (weight 3), 4k steps.
Result: **79% on validation** (e: 54%), within 4 cm 89/100, grasped 87, lifted 83. Classified 48 scenes: 39 success, 6 never closed, 1 reopened, 1 slipped, 1 lost.
Test (one run, 150 scenes): **78% (117/150)**; within 4 cm 131/150, grasped 125, lifted 122. Above the relaxed ~75% gate, so stacking starts (docs/plan_stacking.md).

## Recommendations
- The grasp-then-lift timing is the weak point (stalls with the long wait, early lift and early closing with the short one). Next options: a middle setting, or give the policy its previous action / two frames so it knows how long it has been waiting.
- Keep checking labels, not only losses: the DAgger defect was visible only in the labels.
- Evaluate with 150+ scenes and compare models on the same scenes; 100 scenes cannot separate 60-70% models.
