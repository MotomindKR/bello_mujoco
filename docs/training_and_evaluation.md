# Training and evaluation

## Profiles

`bello-train` exposes four profiles:

```text
smoke
locomotion
hardware_locomotion
motion
```

`smoke` runs a real, minimal MJX-Warp PPO update. `locomotion` is the canonical
gait baseline. `hardware_locomotion` uses the same reward and policy contract
with the longer credit horizon used by the physical policy. `motion` is bounded
BeyondMimic tracking.

Common public overrides are seed, total timesteps, environment count, motion
path, resume path, checkpoint destination/name, W&B enablement/run name, and
video enablement. Edit a maintained profile in `configs/profiles/train/` when
changing task numerics; do not add a parallel recipe.

## Locomotion baseline

The physical profile is a 60 kg sphere-hand robot with 13 policy joints and 12
held arm joints. Commands contain forward, lateral, and yaw velocity. Ten
percent of sampled commands are exactly zero; moving samples are conditioned
outside the shared standing deadbands.

The strong-stand reward baseline keeps these principal weights:

```text
linear velocity tracking       1.0
yaw velocity tracking          1.0
vertical velocity             -2.0
flat orientation              -2.0
base height                  -10.0
normalized torque L1          -1.2
over-peak normalized torque  -12.0
standing foot motion          -1.0
contact timing                -5.0
touchdown air time             0.5
termination                 -100.0
```

Torque is normalized by each policy joint's available peak torque. The
over-peak term is quadratic. Training intentionally leaves policy effort
unclipped so the penalty observes excess demand; HIL and hardware enforce
physical limits.

## Evaluation

`bello-eval` always uses native MuJoCo. Locomotion streams the scene, commands,
actions, rewards, and termination diagnostics to Rerun. Motion evaluation
renders a deterministic video and reports completion and RMS tracking errors.

There is no public MJX evaluation backend. This keeps evaluation behavior
aligned with HIL and physical deployment.

## Checkpoints

Schema-7 checkpoints store:

- resolved task and training metadata;
- policy architecture/distribution;
- actor, critic, and action layouts;
- task and robot profile identifiers;
- source/prepared/simulation/viewer asset hashes;
- motion content and schema hashes when applicable;
- normalization, optimizer, environment continuation, RNG, and sampler state.

Evaluation validates only the semantic deployment contract. Resume additionally
requires the training and continuation contract, but permits a larger
`total_timesteps` value and a different output destination. Old schemas fail
with an instruction to use their archived Git revision.

## Validation

```bash
uv run bello-train experiment=smoke
uv run pytest -q tests/test_commands.py tests/test_rewards.py tests/test_checkpoint.py
```
