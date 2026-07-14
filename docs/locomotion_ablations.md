# Locomotion Ablation Pipeline

Use `experiments/train/locomotion_clean.yaml` as the control condition for all
new locomotion experiments. It deliberately removes gait phase from the policy,
all gait/contact/clearance/air-time/swing rewards, pose and height targets,
observation noise, reset perturbations, pushes, and command curriculum. The
remaining reward answers only three questions: does the pelvis track the
command, does the body remain upright without vertical or roll/pitch motion, and
does it do so without violent or invalid controls?

The control starts with a fixed `0.3 m/s` forward command. Its linear tracking
term uses `sigma: 0.05` and weight `2.0`, and termination costs `-5.0`. This
makes moving toward the command materially better than standing still without
making one exploratory fall more expensive than all of the tracking reward in
an episode. The asset uses spherical end effectors on both hands. PPO starts at
a constant `1e-4` learning rate and log standard deviation `-1.5`, capped at
`-0.5`. Policy joints use conservative explicit residual scales, and MJX-Warp
reserves `njmax: 2048` constraints per environment. The clean experiment clips
normalized actions to `[-1, 1]`; the repository default remains `[-5, 5]` for
existing recipes.

The policy controls the 13 leg and waist-yaw joints inherited from
`locomotion.yaml`; the upper body remains held. Actor and critic both observe
pelvis-local linear/angular velocity, pelvis gravity, command, joint state, and
last action. Keeping their observation sets equal makes the first baseline
easier to interpret.

## Run Structure

Every candidate uses the same two budgets:

```text
screen:  25,165,824 steps = 512 updates  x 2,048 envs x 24 steps
full:   150,994,944 steps = 3,072 updates x 2,048 envs x 24 steps
```

Run seeds `0`, `1`, and `2` from scratch. Do not select a reward from one lucky
run. The training command for a local screen is:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion_clean \
  config.seed=0 \
  config.total_timesteps=25165824 \
  checkpoint_config.name=locomotion_clean_screen_seed0 \
  wandb_config.run_name=locomotion_clean_screen_seed0
```

Use the same overrides after `--` with `bello-vast-train` for a Vast run. Change
the seed in the PPO, checkpoint, and W&B names together.

## Experiment Order

1. Forward-discovery control: run the clean config unchanged. This establishes
   whether corrected velocity measurements can learn balance and forward
   tracking without gait shaping.
2. Command controls: run zero-only and ranged-forward controls independently:

   ```text
   zero-only:     command_config.lin_vel_y=[0.0,0.0]
                  command_config.standing_probability=1.0
   ranged-forward: command_config.lin_vel_y=[0.15,0.35]
                  command_config.standing_probability=0.0
   ```

3. Run the two prepared single reward ablations against the unchanged control.
   Do not stack terms during this phase:

   ```text
   locomotion_clean_slip:      foot_slip only, weight -0.10
   locomotion_clean_air_time:  feet_air_time only, weight 0.25
   ```

   The air-time term continuously rewards exactly one foot in contact using the
   shorter of the current stance-contact and swing-air durations, capped at
   `0.40s`. It does not require a clock or prescribe which foot should swing.
4. Combine only terms that improve the median result across all three seeds.
   `locomotion_clean_feet` is the prepared combined config; run it from scratch
   against a fresh three-seed control only after both single-term gates pass.
   Keep gait-contact and clearance shaping as later, independent ablations if
   the minimal foot terms are insufficient.
5. Expand commands one axis at a time: forward range, then yaw, then lateral.
   Curriculum success must be computed from moving episodes separately from
   exact-zero episodes before enabling automatic range expansion.
6. Add robustness last, in this order: observation noise, reset joint noise,
   reset yaw, then pushes. Each addition remains a separate three-seed ablation.

Do not re-enable the current constant `swing_foot_velocity` target in this
matrix. A future swing-velocity experiment should differentiate a phase-shaped
foot trajectory so target velocity approaches zero at liftoff and touchdown.

## Promotion Gates

At updates 512, 1,536, and 3,072, evaluate deterministic policies in both CPU
MuJoCo and single-environment MJX-Warp. Launch each case five times to expose
Warp contact variation:

```text
zero:       [0.0, 0.0, 0.0]
forward:    [0.0, 0.3, 0.0]
lateral:    [0.2, 0.0, 0.0]  after lateral training begins
yaw:        [0.0, 0.0, 0.3]  after yaw training begins
```

A screen advances only when every seed is finite, median zero/forward survival
is at least 900 of 1,000 steps, and forward tracking does not come from sliding
or near-constant double support. Rank surviving candidates first by survival,
then velocity tracking, then foot slip/contact behavior, and finally control
cost. Never promote on total reward alone because reward definitions differ
between ablations.

The existing Rerun evaluator provides the required velocity, contact, action,
and done traces. A headless numeric evaluator should be added before scaling the
matrix beyond these initial terms; until then, save the five rollout summaries
and videos with each W&B run.
