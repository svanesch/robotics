# Phase 1 results: small VLA on datasets/so101_v1 (1500 episodes, 3 tasks, 2-4 blocks)

All closed-loop numbers: 60 test-split episodes in the simulator (95% interval about +-10 points, so 23% vs 12% is not a significant difference).

| run | architecture | train | val L1 (norm.) | closed loop (test) | prompt sensitivity* |
|---|---|---|---|---|---|
| r1 | ResNet18 stride 32 + raw text tokens in the transformer | 12k steps | 0.059 | 14/60 = 23% (lift 29, place-left 8, place-right 40) | 0.002 rad |
| r2 | + text encoder + FiLM (stride 32) | stopped at 3.2k | 0.091 | not evaluated | 0.003 rad |
| r3 | stride-16 features (8x8 tokens), FiLM | stopped at 2.5k | 0.103 | not evaluated | 0.002 rad |
| r4 | r3 + prompt-driven spatial readout, 3x language LR, 3x early-frame sampling | 12k steps | 0.061 | 7/60 = 12% (lift 4/21, place-left 2/24, place-right 1/15) | 0.003 rad |

*Prompt-sensitivity column: mean change of the predicted joint targets when the prompt's target colour is swapped on identical images.
**This test is flawed and should not be read as "the policy ignores the prompt":** the swapped colour was drawn from all 9 colours, so it was
usually not present in the scene (2-4 blocks) and there was nothing for the policy to switch to. A valid test must swap to a colour that is visible.
Rounds 2 and 3 were stopped early on the strength of this test; that decision was not justified.

## Closed-loop behaviour of r4 (60 test episodes) - what the policy actually does
- gripper's closest approach to the requested block: median 1.6 cm; to the nearest other block: median 10.7 cm
- within 4 cm of the requested block: 39/60; of some other block: 13/60
- closed the gripper on the requested block: 29/60; on another block: 9/60; on neither: 22/60
- lifted the requested block: 12/60; lifted another block: 5/60; full task success: 7/60
- ended closer to the requested block than to the others in 41/60 (chance about 21/60)
=> grasp/lift execution is a bottleneck (only 12 of 29 grasps on the right block turn into a lift). **The inference that the prompt steers the block choice was WRONG**, see the matched-scene probe below: the 'closer to the requested block' result is explained by a layout bias (the target is placed first inside the reachable workspace, distractors may lie up to 3 cm outside it, so the requested block is on average the most central one).

r4 failure breakdown (53 failures): 43 never lifted the target (no wrong block lifted), 5 lifted a wrong block, 5 lifted the target but did not finish.
r4 success by number of blocks: 2 blocks 3/17, 3 blocks 0/22, 4 blocks 4/21.

## Phase 2: execution quality (DAgger)
Same 150 test scenes (seed 1234), closed loop, all other settings identical:

| run | data | success | lift | place-left | place-right | closed on requested / other / neither | lifted requested |
|---|---|---|---|---|---|---|---|
| r4 | 1500 expert episodes | 15/150 = 10% | 11/47 | 2/59 | 2/44 | 74 / 20 / 56 | 32 |
| d1 | + 229 DAgger episodes (policy-driven states, expert labels), fine-tuned 4000 steps from r4 | 33/150 = 22% | 15/47 | 11/59 | 7/44 | 93 / 27 / 31 | 62 |

d1 by number of blocks: 2 blocks 7/36, 3 blocks 13/64, 4 blocks 13/50; seen colour x task combos 21/74, held-out combos 12/76.
Inference ablation on r4 (90 scenes): default ensembling 13/90, executing 8 steps of each chunk open-loop 5/90.
r1 (old architecture) cannot be loaded by the current model code; its 60-episode result (14/60 = 23%) was measured on a different sample.
Note: the choice among runs is made on the test scenes, so the best number is slightly optimistic.

## Matched-scene language probe (d1, 30 scenes, 61 episodes)
Same scene and images, prompted once per reachable block ("pick up the <colour> block", only colours that are in the scene):
- closest approach went to the requested block in 46% of episodes; chance is 49%
- both prompts of a scene sent the gripper to their own block in 1 of 30 scenes (3%)
- open-loop shift of the predicted action chunk when only the prompt changes: 0.046 rad (small next to the 0.3 rad action error on early frames)
- grasped the requested block in 38%, lifted it in 23%
=> with this architecture and data the policy has NOT learned to ground colour words; it picks a block by geometry. The execution gains from DAgger are real, the language is not there.

### Round 2 (d2)
d1 -> collect 159 DAgger episodes with d1 as the driving policy -> fine-tune 3000 steps on expert + dagger_1 + dagger_2 (sampling weights 1 / 3 / 3).

| run | success (150 test scenes) | lift | place-left | place-right | closed on requested / other / neither | lifted requested |
|---|---|---|---|---|---|---|
| r4 | 15/150 = 10% | 11/47 | 2/59 | 2/44 | 74 / 20 / 56 | 32 |
| d1 | 33/150 = 22% | 15/47 | 11/59 | 7/44 | 93 / 27 / 31 | 62 |
| d2 | 38/150 = 25% | 18/47 | 16/59 | 4/44 | 101 / 27 / 23 | 69 |

d2: 2 blocks 9/36, 3 blocks 13/64, 4 blocks 16/50; seen combos 22/74, held-out combos 16/76; touched a wrong block in 12% of episodes.
The r4 -> d1 gain is clear (95% intervals about +-7 points on the difference); d1 -> d2 (+5 points) is within noise.
d2 matched-scene language probe: follow 47.5% (chance 49%), grasped the requested block 44%, 1 of 30 scenes controllable -> still no colour grounding.

## Limitations (what did not work and why)
1. **No language grounding.** Across r1..d2 the policy chooses a block by geometry, not by the colour word (matched-scene follow rate = chance). Likely causes:
   a text side learned from scratch from ~1200 episodes with a 36-word vocabulary; blocks only 3-4 px wide in the external view at 128 px;
   only ~12k steps on a 4 GB Pascal GPU; a layout bias in the data generator (the target is placed first inside the reachable workspace,
   distractors may lie slightly outside) that lets the policy score above chance on "approached the requested block" just by going to the most central block.
2. **Low absolute success (25%).** Grasp/lift remains the weakest step: 101 of 150 episodes close the gripper on the requested block, 69 lift it, 38 finish the task. DAgger fixed a large part of the "never lifts" failures; place-right (9%) and wrong-block lifts (18/150) remain.
3. **Small, simulated, single robot.** 128 px images, one arm, one table, simulation only; no claim about the real SO-101.
4. **Statistics.** 150 episodes per evaluation (about +-7 points); the run choice was made on the same test scenes, so the best number is slightly optimistic.
5. **Early decisions made on a flawed test.** Rounds r2/r3 were stopped after ~3k steps because of a prompt-sensitivity test that swapped to colours that were not in the scene.

## Recommendations
1. **Fix the data generator bias first**: draw every block position from the same distribution and choose the target uniformly, otherwise "go to the most central block" is a shortcut. Cheap and necessary before any language work.
2. **Make language learnable**: (a) a matched-contrast dataset: the same scene and images with different prompts and different expert actions (the DAgger labels already do this implicitly when the policy goes to the wrong block); (b) a pretrained image-text model (CLIP text embeddings, then a CLIP image encoder) instead of a text side learned from 1.2k episodes - not tested here, memory/speed on 4 GB unknown; (c) higher resolution (192-224 px) so colours are resolved.
3. **More DAgger rounds** are the cheapest known gain for execution (10% -> 22% -> 25%); gains per round are shrinking, so combine with more varied initial states, not just more rounds.
4. **Fairer evaluation**: 300+ episodes and a separate model-selection split.
5. **If free-form language stays out of reach**: condition on a discrete task id (lift / left / right) and select the target block by a simple colour detector or rule - a clearly labelled modular system that would likely work much better than the end-to-end version today.
6. **A bigger GPU** would allow longer training and a pretrained vision-language backbone (SmolVLA-class models need far more than 4 GB).
