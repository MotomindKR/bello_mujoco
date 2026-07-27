# Training and evaluation

The complete public fields are in the
[command reference](commands.md#bello-train). Nested PPO, environment, reward,
asset, and output fields are in the
[training configuration reference](training_configuration.md).

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

The public command can override seed, total timesteps, environment count,
motion path, resume path, checkpoint destination/name, W&B enablement/run name,
and video enablement. Other task numerics are maintained in
`configs/profiles/train/`; arbitrary nested profile keys are not part of the
public command.

## Locomotion baseline

The physical profile is a 60 kg sphere-hand robot with 13 policy joints and 12
held arm joints. Commands contain forward, lateral, and yaw velocity. Ten
percent of sampled commands are exactly zero; moving samples are conditioned
outside the shared standing deadbands.

The canonical reward combines command tracking, base stability, normalized
torque costs, standing foot motion, contact timing, touchdown air time, and
termination cost. Exact terms and weights are defined in
[`configs/train/default.yaml`](../configs/train/default.yaml) and explained in
the [training configuration reference](training_configuration.md#locomotion-reward-reward_config).
Training leaves policy effort unclipped so the over-peak torque penalty observes
excess demand; the emulator and hardware enforce physical limits.

## Evaluation

`bello-eval` always uses native MuJoCo. Locomotion streams the scene, commands,
actions, rewards, and termination diagnostics to Rerun. Motion evaluation
renders a deterministic video and reports completion and RMS tracking errors.

Locomotion gamepad input is enabled by default. With a connected controller it
replaces the configured fixed command; with a disconnected controller the
command is zero. Disable gamepad input to apply a fixed command. An unbounded
`steps=null` rollout continues until controller exit or interruption, and
completed episodes reset unless `stop_on_done=true`.

Motion evaluation rejects non-default locomotion command/gamepad/stochastic
fields. Its default step horizon is the checkpoint environment's maximum
episode length.

There is no public MJX evaluation backend. This keeps evaluation behavior
aligned with the native emulator and physical deployment.

## Checkpoints

Schema-7 checkpoints store:

- resolved task and training metadata;
- policy architecture/distribution;
- actor, critic, and action layouts;
- task and robot profile identifiers;
- source/prepared/simulation/viewer asset hashes;
- motion content and schema hashes when applicable;
- normalization, optimizer, environment continuation, RNG, and sampler state.

Evaluation validates document integrity, metadata/config hashes,
policy/value/optimizer state layout, normalization, saved update, semantic task
contract, and runtime assets. Resume additionally requires environment
continuation, observation stacks, RNG, current learning rate, sampler state,
environment layout, and the full resume contract. A revised `total_timesteps`
value is accepted when its resulting update budget still covers the saved
update; output destinations may change. Old schemas fail with an instruction to
use their archived Git revision.

Locomotion final checkpoints are timestamped checkpoint roots. Motion runs also
maintain a `final` child below their timestamped root.
