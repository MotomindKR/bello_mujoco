# Command reference

The repository exposes eight public commands. Each command has a strict Hydra
key set: keys not declared by that command fail during decoding, and the
resulting mapping is loaded into an immutable dataclass before the workflow
starts. The command YAML is the sole owner of defaults; every schema field must
be present, and decoding never fills a missing command field from Python.
The tables below define the supported value domains. Workflow validators also
fail early on invalid ranges and task-specific fields.

## Invocation and inspection

Use the same override form for every command:

```text
uv run <command> field=value
```

Hydra accepts booleans as `true` or `false`, optional values as `null`, and
lists or tuples in bracket form such as `[0.5,0.0,0.0]`. Shell quoting may be
needed around brackets or values containing spaces. The public command
dataclasses are flat; nested PPO, reward, and environment fields are maintained
in the training profile YAML and are not arbitrary CLI overrides.

Every command supports:

| Option | Result |
| --- | --- |
| `--help` | Show the command's composed Hydra configuration and basic help. |
| `--hydra-help` | Show Hydra's own flags and override grammar. |
| `--cfg job` | Print the fully composed job configuration without running it. |

Hydra does not change the process working directory. Its diagnostic run
directories are written below `outputs/hydra/<command>/`.

## `bello-prepare-assets`

Build the canonical Bello URDF, runtime MJCF, and viewer MJCF from
`deps/bello-stack/descriptions/bello/`. Installed wheels carry the pinned URDF
and meshes, so standalone asset preparation does not require a source checkout.
Meshes and XML are validated before the prepared asset is returned.

Configuration source: `configs/commands/prepare_assets.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `profile` | `physical` or `visualizer`; `physical` | Select the canonical physical recipe or the lower-resolution persistent viewer recipe. |
| `output_root` | path or `null`; `null` | Override the generated-asset root. `~` is expanded. With the physical profile, `null` uses the content-addressed user cache at `$XDG_CACHE_HOME/bello-mujoco/assets` (or `~/.cache/bello-mujoco/assets`) and retains the eight most recently used variants. The visualizer profile defaults to `assets/generated/visualizer`. |

Both profiles use the 60 kg, box-collision, sphere-hand morphology with active
waist yaw and fixed waist roll/pitch, neck yaw, and head pitch. The physical
profile keeps a 200,000-face visual budget; the visualizer profile uses 20,000.
The collision budget remains 1,000 faces. Both standalone profiles omit
generated IMU sensors and enforce policy effort limits. The emulator and
checkpoint-bound runtimes prepare their own explicitly configured variants
rather than assuming this standalone output is interchangeable.

## `bello-train`

Compose a maintained experiment profile and execute its PPO training loop.
Locomotion and motion tracking use different environment contracts but share
the PPO, checkpoint, W&B, and Warp infrastructure.

Configuration source: `configs/commands/train.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `experiment` | `smoke`, `locomotion`, `hardware_locomotion`, `motion`, `twist_teacher`, `twist_student`, or `twist2`; `smoke` | Select the maintained profile stack. |
| `seed` | integer or `null`; `null` | Override the profile's PPO and rollout seed. |
| `total_timesteps` | integer or `null`; `null` | Override the transition budget. It must cover at least one rollout batch; a remainder smaller than one batch is not sampled. |
| `num_envs` | integer or `null`; `null` | Override the batched environment count. It must be positive and compatible with the minibatch count. |
| `contact_budget_per_env` | integer or `null`; `null` | Override the profile's positive Warp active-contact capacity per environment. Use the recorded value when resuming a checkpoint and raise it only when capacity telemetry reports overflow. |
| `motion_path` | path or `null`; `null` | For `experiment=motion`, select one canonical `.npz` clip by replacing the profile's motion directory and sequence. Checkpointed training requires the clip to be inside the repository and tracked in `HEAD`. Do not supply this field for locomotion profiles. |
| `dataset_path` | path or `null`; `null` | Override the packed `index.json` for a TWIST or TWIST2 run. Other experiments reject it. |
| `teacher_checkpoint_path` | path or `null`; `null` | Required for `twist_student` and rejected by every other experiment, including `twist_teacher`. |
| `resume_path` | path or `null`; `null` | Restore a continuation checkpoint. The current checkout must be clean and normally match its recorded Git commit; resume then validates the full training and environment state. |
| `allow_git_commit_mismatch` | boolean; `false` | Permit `resume_path` to come from a different commit. Requires `resume_path`; it waives source-revision identity while metadata semantics, layouts, state, and content checks still apply. |
| `checkpoint_directory` | path or `null`; `null` | Override the profile's checkpoint parent directory. It must be outside the source repository or covered by a Git ignore rule. |
| `checkpoint_name` | string or `null`; `null` | Override the checkpoint run name below the selected directory. It must be one path component. |
| `wandb_enabled` | boolean or `null`; `null` | Enable or disable W&B independently of the profile default. |
| `run_name` | string or `null`; `null` | Override the W&B run name. |
| `video_enabled` | boolean or `null`; `null` | Enable or disable scheduled native-MuJoCo evaluation videos. |

Profile composition is deterministic:

| Experiment | Configuration layers, lowest to highest precedence |
| --- | --- |
| `smoke` | `configs/train/default.yaml`, then `configs/profiles/train/smoke.yaml` |
| `locomotion` | `configs/train/default.yaml`, then `configs/profiles/train/locomotion.yaml` |
| `hardware_locomotion` | locomotion layers, then `configs/profiles/train/hardware_locomotion.yaml` |
| `motion` | `configs/train_motion/default.yaml`, then `configs/profiles/train/motion.yaml` |
| `twist_teacher` | `configs/train_twist/default.yaml`, then `configs/profiles/train/twist_teacher.yaml` |
| `twist_student` | `configs/train_twist/default.yaml`, then `configs/profiles/train/twist_student.yaml` |
| `twist2` | `configs/train_twist2/default.yaml`, then `configs/profiles/train/twist2.yaml` |

The public fields above are applied after profile composition. See
[Training configuration](training_configuration.md) for every nested section
and its contract.

## `bello-eval`

Restore a checkpoint and evaluate it with native MuJoCo. The recorded Git
commit is required by default. Locomotion opens a Rerun recording; motion
tracking renders a video and prints tracking statistics.

Configuration source: `configs/commands/eval.yaml`.

| Field | Type and default | Applies to | Meaning |
| --- | --- | --- | --- |
| `task` | `locomotion`, `motion`, `twist_student`, or `twist2`; `locomotion` | all | Select checkpoint validation and evaluation mode. |
| `checkpoint_path` | path or `null`; `null` | all | Required checkpoint directory. `~` is expanded. |
| `allow_git_commit_mismatch` | boolean; `false` | all | Permit a different clean checkout than the checkpoint's recorded commit. Other checkpoint checks still apply. |
| `output_path` | path; `outputs/evaluation/motion.mp4` | motion, twist_student, twist2 | Video destination for motion. TWIST and TWIST2 write JSON and select task-specific `.json` defaults when this field was not explicitly overridden. |
| `video_path` | path or `null`; `null` | twist2 | TWIST2 representative-video destination. When omitted, TWIST2 writes `outputs/evaluation/twist2.mp4`; explicitly set `video_path=null` to score without rendering. |
| `command` | three floats; `[0,0,0]` | locomotion | Fixed `+X` lateral/right, `+Y` forward, and `+Z` yaw velocity command. |
| `steps` | positive integer or `null`; `null` | locomotion, motion, twist2 | Bound rollout steps. It limits only the representative video for TWIST2 and is rejected for `twist_student`. |
| `stop_on_done` | boolean; `false` | locomotion, motion, twist_student | Stop at the first done/terminal state. TWIST2 always stops terminal clips so its held-out metrics cannot cross an automatic reset, and rejects this override. |
| `stochastic_actions` | boolean; `false` | locomotion | Sample from the checkpoint policy instead of using its deterministic mean. |
| `action_seed` | nonnegative integer; `0` | locomotion | PRNG seed used only for stochastic actions. |
| `gamepad_enabled` | boolean; `true` | locomotion | Read commands from controller zero. When disconnected, the command is zero. When false, `command` is used continuously. |

Fields that apply to only one task must not be explicitly supplied when the
other task is selected, even if the supplied value equals the documented
default.
Likewise, supplying `command` requires `gamepad_enabled=false`, and supplying
`action_seed` requires `stochastic_actions=true`; otherwise those values would
have no effect.

The locomotion gamepad maps left-stick vertical to forward velocity,
left-stick horizontal to lateral velocity, and right-stick horizontal to yaw.
A/Cross requests an episode reset and Start/Options exits. Stick values are
dead-zoned at the fixed value 0.1. Neutral remains the exact standing command;
each supported nonzero direction starts at the checkpoint's minimum moving
speed and scales to its command bound. The controller index is fixed at zero. A
fixed `command` uses `[lin_vel_x, lin_vel_y, ang_vel_yaw]`, with `+X`
lateral/right and `+Y` forward. It is passed through without clipping or range
validation, so it should remain within the checkpoint's trained bounds.

Evaluation validates the checkpoint task, policy and critic observation
sizes/layouts, action size/layout, asset configuration, and environment
metadata. It does not accept runtime overrides for the trained robot or reward
formulation.

Motion-policy evaluation renders at 640 by 480 pixels and 50 fps. The `width`,
`height`, and `fps` fields under `bello-motion` configure reference rendering.
TWIST evaluation accepts only student checkpoints and deterministically scores
every held-out clip. TWIST2 likewise requires its own typed one-stage checkpoint,
scores every held-out clip, and renders the first held-out clip by default;
`steps` bounds only that representative video. Both tasks reject locomotion
commands, stochastic actions, and gamepad fields. TWIST2 also rejects
`stop_on_done` because terminal stopping is mandatory.

## `bello-motion`

Operate on canonical motion archives without training a policy.

Configuration source: `configs/commands/motion.yaml`.

| Field | Type and default | Applies to | Meaning |
| --- | --- | --- | --- |
| `operation` | `import`, `import_dataset`, `inspect`, or `render`; `inspect` | all | Select the offline motion operation. |
| `source` | path or `null`; `null` | all | Required input file or directory. |
| `output` | path or `null`; `null` | import, import_dataset, render; optional for inspect | Destination archive, dataset directory, report file, or video. |
| `jobs` | positive integer; `8` | import_dataset | Maximum parallel legacy imports. |
| `overwrite` | boolean; `false` | import_dataset | Rebuild existing output clips instead of reporting them from the current dataset. |
| `reject_self_collisions` | boolean; `true` | import | Reject a converted clip if a per-frame native forward/collision check finds self-collision. |
| `width` | positive integer; `640` | render | Output video width in pixels. |
| `height` | positive integer; `480` | render | Output video height in pixels. |
| `fps` | positive integer; `30` | render | Output video frame rate. |
| `require_complete` | boolean; `false` | inspect | Fail the command when feasibility replay terminates before the clip completes. |

Operation-specific fields must not be explicitly supplied for another
operation, even if the supplied value equals the documented default.

`import` and `import_dataset` read trusted local GMR pickle data and emit the
canonical NPZ schema. Dataset import also writes `manifest.json`; any failed
sequence makes the overall command fail after the manifest is recorded.
Single-file import obeys `reject_self_collisions` using a per-frame native
forward/collision check. Dataset import always records self-collision
diagnostics instead of rejecting a sequence on that basis.

`inspect` replays the archive through the canonical native model and reports
completion/failure reason, root and joint tracking errors, peak actuator-force
ratio, actuator saturation, contact fraction, stance-foot slip, and nonfinite
state. `render` visualizes the reference itself, not a policy.

Single-file import, reference render, and an optional inspect report overwrite
their destination. The `overwrite` field controls dataset import only.
Single-file import records the default 10 tracked bodies; dataset import uses
the fixed 14-body BeyondMimic set. That body schema is not selectable from the
CLI and becomes part of downstream motion/checkpoint contracts.

For inspect, incomplete replay means nonfinite state or a hard-coded feasibility
failure: root-height error above 0.35 m, root-orientation error above 1.2 rad, or
end-effector position error above 0.65 m. `require_complete=true` turns that
report outcome into command failure.

## `bello-twist-data`

Validate operator-supplied licensed sources or pack already-retargeted clips.
It never downloads datasets or body models. Configuration source:
`configs/commands/twist_data.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `operation` | `scan_amass`, `validate`, or `build`; `validate` | Generate an AMASS manifest, validate the source boundary, or build split files and an index. |
| `manifest` | path or `null`; `null` | Output path for `scan_amass`; required input manifest otherwise. |
| `source_root` | path or `null`; `null` | Aggregated AMASS root for `scan_amass`; only `*_stageii.npz` motion archives are included. |
| `dataset_release` | string or `null`; `null` | Required operator-supplied AMASS release identifier for `scan_amass`. |
| `license_id` | string or `null`; `null` | Required operator-supplied license identifier for `scan_amass`. |
| `split_seed` | integer; `20251009` | Fixed seed recorded in a generated manifest. |
| `sampling_weight` | positive float; `1.0` | Explicit weight assigned to every scanned sequence. |
| `overwrite` | boolean; `false` | Permit replacement of an existing generated manifest. |
| `body_model_root` | path or `null`; `null` | Required for SMPL-H/SMPL-X and checked for the three SMPL-X NPZ model files. |
| `retargeted_root` | path or `null`; `null` | Required only for build; clips use `<source_id>__<sequence_id>.npz`. |
| `output_root` | path or `null`; `null` | Required only for build and must be empty. |
| `robot_model_path` | canonical Bello URDF path | Robot content used in provenance validation and the dataset fingerprint. |
| `gmr_root` | `deps/GMR` | Clean pinned GMR checkout whose exact commit enters the fingerprint. |
| `quality_config` | `configs/twist/quality.yaml` | Versioned rejection and quarantine thresholds. |
| `curation_config` | `configs/twist/amass_flat_ground.yaml` | Versioned pre-split exclusions for generated outputs and motions requiring unsupported terrain; used only by `scan_amass`. |

See [TWIST teacher and student](twist.md) for acquisition, conversion,
splitting, licensing, provenance, and output contracts.

## `bello-vast-train`

Launch one branch-pinned Vast.ai lifecycle. The launcher selects an offer,
records durable rental state, provisions the repository, starts detached
training, monitors progress, relies on W&B for configured checkpoint upload,
and attempts verified teardown on every terminal path. If destruction cannot be
confirmed, it retains the durable state and fails closed.

Configuration source: `configs/commands/vast_train.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `experiment` | `smoke`, `locomotion`, `hardware_locomotion`, `motion`, `twist_teacher`, `twist_student`, or `twist2`; `hardware_locomotion` | Select the same maintained profile used by `bello-train`. |
| `motion_path` | path or `null`; `null` | Required for motion jobs. Do not supply this field for other jobs. The clip must exist in the selected remote commit. |
| `dataset_path` | path or `null`; `null` | Required local schema-3 `index.json` for TWIST or TWIST2. Its five portable dataset files are transferred with resumable `rsync` and verified remotely. |
| `teacher_checkpoint_path` | path or `null`; `null` | Required local typed teacher checkpoint for `twist_student`; rejected by every other stage. |
| `allow_git_commit_mismatch` | boolean; `false` | Permit a staged student teacher checkpoint from a different clean source commit. Dataset, robot, observation, and layout checks remain enforced. |
| `max_price` | positive float or `null`; `null` | Maximum total hourly rental price. A positive value is required, including for dry-run offer queries. |
| `branch` | string or `null`; `null` | Required remote `origin` branch. A null value is rejected before rental. |
| `run_name` | string or `null`; `null` | Explicit lifecycle/W&B run label. Otherwise a unique name is generated from the profile, branch, commit, and current UTC timestamp. |
| `duration_hours` | positive float; `6.0` | Hard job timeout. The progress timeout is capped at 30 minutes and never exceeds this duration. |
| `skip_confirmation` | boolean; `false` | Skip the final rental confirmation for unattended operation. |
| `dry_run` | boolean; `false` | Resolve Git state and print the offer query without reading credentials or renting. |
| `preferred_countries` | ISO country-code list; `[KR, JP, TW, HK, SG]` | Rank nearby offers ahead of globally better marketplace value. |
| `min_upload_mib_per_second` | positive float; `5.0` | Reject and destroy a TWIST instance whose direct-SSH upload probe falls below this rate. |
| `route_probe_mib` | positive integer; `16` | Payload size for the pre-transfer direct-SSH upload probe. |

The selected profile, `pyproject.toml`, `uv.lock`, and any motion clip are
verified in the exact `origin/<branch>` commit before rental. Ignored local
motions are never uploaded implicitly. TWIST transfers only the five files
named by its validated dataset contract and, for a student, the selected
checkpoint directory. Credentials come from `VAST_API_KEY` or
keyring `vast/api_key`, `WANDB_API_KEY` or keyring `wandb/api_key`, and
optionally `GITHUB_TOKEN` or keyring `github/bello_mujoco_read_token`.

The maintained offer policy requests one RTX 5090, at least 64 GB CPU RAM and
80 GB disk, reliability of at least 0.98, CUDA 13 compatibility, verification,
rentability, and direct SSH. The marketplace search is global. Non-dry
execution requires Vast and W&B credentials; GitHub
credentials are optional for public repositories.

Durable lifecycle state lives under `.vast/`. Do not remove it while a rental
may exist; rerunning the launcher allows reconciliation to inspect and destroy a
recorded instance.

## `bello-run-emulator`

Run the headless native-MuJoCo RobStride hardware-interface emulator. It always
exposes the configured SocketCAN buses, pelvis EBIMU stream, and PoseService
while stepping the canonical model in real time. It opens no graphical windows.

Configuration source: `configs/commands/emulator.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `bus_interfaces` | list of strings; `[can0=vcan0,can1=vcan1]` | Map configured logical bus names to Linux SocketCAN interfaces. Names must be unique and use `name=interface` form. |
| `command_latency_seconds` | nonnegative float; `0.01` | Delay MIT motor-control frames before applying them to MuJoCo. Lifecycle, parameter, and stop traffic remains immediate; use `0` to disable the delay. |

The emulator is checkout-only. It selects the canonical `lower_body` and
`upper_body` groups from `deps/bello-stack/descriptions/bello/hardware.yaml`,
uses `deps/bello-stack/descriptions/robstride/models.yaml` for protocol and
hardware limits, and derives direct and differential transmissions from the
canonical URDF. Only the logical-to-SocketCAN interface mapping is local. It
applies MIT impedance control, publishes the pelvis IMU serial stream and
PoseService, and paces simulation against wall-clock time. Its torso guide is
always engaged at startup; Enter releases it, and another Enter resets the robot
and re-engages the guide.

## `bello-run-hardware`

Validate or execute a policy through the `bello-hardware` Python binding. It is
installed from the pinned `deps/bello-stack` uv workspace member:

```text
git submodule update --init --recursive
uv sync --frozen
uv run bello-run-hardware checkpoint_path=... \
  remote_endpoint=http://visualizer-host:50054
```

Physical hardware execution is checkout-only. A wheel installation does not
contain the binding or hardware description.

The command always selects JAX's CPU backend before loading either hardware
runner. Inherited `JAX_PLATFORMS` and `JAX_PLATFORM_NAME` values are ignored;
there is no hardware backend configuration field.

Configuration source: `configs/commands/hardware.yaml`.

| Field | Type and default | Applies to | Meaning |
| --- | --- | --- | --- |
| `task` | `locomotion` or `motion`; `locomotion` | both | Select the checkpoint and observation contract. |
| `checkpoint_path` | path or `null`; `null` | both | Required locomotion or motion checkpoint directory. Its recorded Git commit normally equals the current clean checkout. |
| `standing_checkpoint_path` | path or `null`; `null` | motion | Required locomotion checkpoint used before and after motion tracking. |
| `allow_git_commit_mismatch` | boolean; `false` | both | Permit a different clean checkout. This does not disable task, shape, asset, motion-content, or hardware preflight checks. |
| `endpoint` | string; `http://127.0.0.1:50051` | both | `bello-hardwared` endpoint. |
| `remote_endpoint` | string or `null`; `null` | locomotion | Required plaintext URL of the visualizer's RemoteCommandService, for example `http://visualizer-host:50054`. |
| `remote_staleness_ms` | positive integer; `200` | locomotion | Maximum age of the latest remote-command sample before the command becomes zero and the stream reconnects. |
| `motion_path` | path or `null`; `null` | motion | Override the checkpoint's bound motion location. Its schema and content hash must still match. |
| `duration_seconds` | positive float or `null`; `null` | both | Maximum post-`RUN` policy duration. `null` runs until operator exit, interruption, or a safety failure. |
| `ramp_seconds` | positive float or `null`; `null` | both | Override nominal-pose ramp time. `null` uses 10 seconds for both tasks. |
| `policy_transition_seconds` | positive float or `null`; `null` | motion | Duration of each standing-to-motion and motion-to-standing target transition. `null` uses 0.2 seconds. |
| `max_motion_tilt_radians` | positive float or `null`; `null` | motion | Maximum robot tilt during transitions into and out of motion tracking and while motion tracking is active. `null` uses 70 degrees (approximately 1.222 radians); standing locomotion keeps its 0.70-radian limit. |
| `status_interval_seconds` | positive float or `null`; `null` | both | Emit periodic policy-state diagnostics at this interval. `null` prints only preflight, phase changes, and summaries. |
| `policy_log_path` | path or `null`; `null` | both | JSONL destination for every executed policy observation and action. `null` selects `policy_steps.jsonl` inside the unique Hydra run directory. An explicit path must not already exist. |
| `arm` | boolean; `false` | both | `false` performs read-only preflight and sends no command. |

Fields that apply to only one task must not be explicitly supplied when the
other task is selected, even if the supplied value equals the documented
default.
Locomotion requires `remote_endpoint`. Its visualizer-hosted physical controller
uses the same stick mapping as `bello-eval`; R1 aborts. Controller or
service loss commands zero, reconnects automatically, and requires a neutral
controller before accepting nonzero intent again. The visualizer service must
produce a fresh sample before the hardware daemon is contacted. Hardware
locomotion has no native-controller or fixed-command mode.

Motion additionally requires a locomotion checkpoint selected by
`standing_checkpoint_path` and absolute pelvis pose from
`bello-state-estimatord` PoseService at `127.0.0.1:50053`. The standing and
motion checkpoints must resolve to the same ordered physical-coordinate
mapping.

Both armed tasks require the operator to type exact interactive confirmations:
`HOLD` before the nominal-pose ramp and `RUN` before policy control. Motion mode
ramps to the standing checkpoint, starts its zero-command locomotion policy
after `RUN`, and requires exact `MOTION` confirmation while that policy remains
active. It then transitions to motion tracking and automatically transitions
back to the standing policy when the reference completes or the operator presses
Ctrl-C during active motion tracking. A subsequent Ctrl-C while standing enters
damped shutdown. While waiting for `RUN`, the runner continues refreshing the
hold target and checking telemetry and measured nominal-pose error bounds:
0.25 rad for direct coordinates and 1.00 rad for the load-bearing differential
ankle motor coordinates.
`duration_seconds` measures only the post-`RUN` policy flow; the ramp and
`HOLD`/`RUN` confirmation waits are outside that duration.

Armed runs print the resolved policy-log path before acquiring control. The
first JSONL record describes checkpoint paths, observation layouts, stack
sizes, action names, and physical target names. Every following record contains
one executed policy step with wall and monotonic timestamps, phase, step,
unstacked observation, normalized stacked policy observation, raw and applied
action, and physical target. Locomotion
records include the velocity command. Motion records cover the standing
policy, both policy transitions, and motion tracking; transition records contain
both the policy target and the blended target actually sent. File encoding and
flushing run on a dedicated background thread so normal disk I/O does not block
the control loop. These complete 50 Hz logs can grow by hundreds of megabytes
per hour, depending on the checkpoint observation-stack size.

Hardware targets use the proportional and derivative gains stored in checkpoint
metadata.

Preflight validates the checkpoint task, sphere-hand/fixed-joint morphology,
policy layouts, generated transmission, motor inventory, IMU presence,
complete state-frame freshness, and initial state. Armed loops consume the latest
complete daemon state frame through one client-paced RPC per policy step.
The complete RPC round-trip counts toward motor, IMU, and group-state age.
Samples outside the configured freshness limits or spanning more than 40 ms fail
closed.
Armed loops continue checking group state, sample age, tilt, acceleration,
angular velocity, loop lag, policy target bounds, and asynchronous command
receipts. The runner does not clip per-step targets; the pinned hardware daemon
rejects motor target steps above 2.0 rad. A safety failure is terminal and
requests damping. Damped groups remain damped until the operator supports the
robot and confirms `IDLE`; unexpected cleanup releases leases without issuing
an automatic disable.

## What is intentionally not public

The seven command documents expose workflow choices and operator controls.
Robot topology, coordinate ownership, motor identities, checkpoint integrity
rules, telemetry limits, control validity windows, pose frames, and other
safety guardrails remain validated implementation constants. Changing those
contracts requires a reviewed source/config change and corresponding tests
rather than an ad hoc command-line override.
