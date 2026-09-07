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

Build the selected robot's canonical URDF, runtime MJCF, and viewer MJCF from
the pinned descriptions in `deps/bello-stack`. Installed wheels carry the
registered URDFs and meshes, so standalone asset preparation does not require a
source checkout. Meshes and XML are validated before the prepared asset is
returned.

Configuration source: `configs/commands/prepare_assets.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `profile` | `physical` or `visualizer`; `physical` | Select the canonical physical recipe or the lower-resolution persistent viewer recipe. |
| `asset` | `bello` or `bello_mini`; `bello` | Select the registered robot contract to prepare. |
| `output_root` | path or `null`; `null` | Override the generated-asset root. `~` is expanded. With the physical profile, `null` uses the content-addressed user cache at `$XDG_CACHE_HOME/bello-mujoco/assets` (or `~/.cache/bello-mujoco/assets`) and retains the eight most recently used variants. The visualizer profile defaults to `assets/generated/visualizer`. |

The selected robot contract determines morphology, mass, joint partition, and
payload attachment points. `bello` remains the maintained 55.068 kg default;
`bello_mini` prepares its reviewed 43.503 kg static morphology. The physical
profile keeps a 200,000-face visual budget; the visualizer profile uses 20,000.
The collision budget remains 1,000 faces. Both standalone profiles omit generated
IMU sensors and enforce policy effort limits. The emulator and checkpoint-bound
runtimes prepare their own explicitly configured variants rather than assuming
this standalone output is interchangeable.

## `bello-train`

Compose a maintained experiment profile and execute its PPO training loop.
Locomotion and motion tracking use different environment contracts but share
the PPO, checkpoint, W&B, and Warp infrastructure.

Configuration source: `configs/commands/train.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `experiment` | `smoke`, `bello_locomotion`, `bello_mini_locomotion`, `bello_mini_standing`, `motion`, `twist_teacher`, `twist_student`, or `twist2`; `smoke` | Select the maintained profile stack. The shared locomotion recipe is internal; `locomotion` remains a runtime task, not a training experiment. |
| `seed` | integer or `null`; `null` | Override the profile's PPO and rollout seed. |
| `total_timesteps` | integer or `null`; `null` | Override the transition budget. It must cover at least one rollout batch; a remainder smaller than one batch is not sampled. |
| `num_envs` | integer or `null`; `null` | Override the batched environment count. It must be positive and compatible with the minibatch count. |
| `learning_rate` | positive float or `null`; `null` | Override the profile's initial learning rate. |
| `contact_budget_per_env` | integer or `null`; `null` | Override the profile's positive Warp active-contact capacity per environment. Use the recorded value when resuming a checkpoint and raise it only when capacity telemetry reports overflow. |
| `motion_path` | path or `null`; `null` | For `experiment=motion`, select one canonical `.npz` clip by replacing the profile's motion directory and sequence. Checkpointed training requires the clip to be inside the repository and tracked in `HEAD`. Do not supply this field for locomotion profiles. |
| `dataset_path` | path or `null`; `null` | Override the packed `index.json` for a TWIST or TWIST2 run. Other experiments reject it. |
| `teacher_checkpoint_path` | path or `null`; `null` | Required for `twist_student` and rejected by every other experiment, including `twist_teacher`. |
| `end_effector_payload_mass_kg` | nonnegative float or `null`; `null` | Motion only. Add this many kilograms to each sphere hand. `null` keeps the profile value, normally zero. |
| `resume_path` | path or `null`; `null` | Restore a continuation checkpoint. The current checkout must be clean and normally match its recorded Git commit; resume then validates the full training and environment state. |
| `warm_start_path` | path or `null`; `null` | TWIST2 only: initialize policy and normalization from a compatible checkpoint trained on a different dataset. Optimizer, update counter, curriculum, rollout state, RNG, and learning rate start fresh. Mutually exclusive with `resume_path`. |
| `allow_git_commit_mismatch` | boolean; `false` | Permit `resume_path` or `warm_start_path` to come from a different commit. It waives source-revision identity while task semantics and runtime layouts remain enforced. |
| `checkpoint_directory` | path or `null`; `null` | Override the profile's checkpoint parent directory. It must be outside the source repository or covered by a Git ignore rule. |
| `checkpoint_name` | string or `null`; `null` | Override the checkpoint run name below the selected directory. It must be one path component. |
| `wandb_enabled` | boolean or `null`; `null` | Enable or disable W&B independently of the profile default. |
| `run_name` | string or `null`; `null` | Override the W&B run name. |
| `video_enabled` | boolean or `null`; `null` | Enable or disable scheduled native-MuJoCo evaluation videos. |

Profile composition is deterministic:

| Experiment | Configuration layers, lowest to highest precedence |
| --- | --- |
| `smoke` | `configs/train/default.yaml`, then `configs/profiles/train/smoke.yaml` |
| `bello_locomotion` | `configs/train/default.yaml`, `configs/train/locomotion.yaml`, then `configs/profiles/train/bello_locomotion.yaml` |
| `bello_mini_locomotion` | `configs/train/default.yaml`, `configs/train/locomotion.yaml`, then `configs/profiles/train/bello_mini_locomotion.yaml` |
| `bello_mini_standing` | `configs/train/default.yaml`, `configs/train/locomotion.yaml`, `configs/profiles/train/bello_mini_locomotion.yaml`, then `configs/profiles/train/bello_mini_standing.yaml` |
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
| `end_effector_payload_mass_kg` | nonnegative float or `null`; `null` | motion | Explicit native-simulation load ablation, in kilograms per hand. `null` uses the checkpoint's trained payload. |
| `output_path` | path; `outputs/evaluation/motion.mp4` | motion, twist_student, twist2 | Video destination for motion. TWIST and TWIST2 write JSON and select task-specific `.json` defaults when this field was not explicitly overridden. |
| `video_path` | path or `null`; `null` | twist2 | TWIST2 representative-video destination. When omitted, TWIST2 writes `outputs/evaluation/twist2.mp4`; explicitly set `video_path=null` to score without rendering. |
| `command` | three floats; `[0,0,0]` | locomotion | Fixed pelvis-frame `[lin_vel_x, lin_vel_y, ang_vel_yaw]` command. Bello forward is `+Y`; Bello Mini forward is `+X`. |
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
dead-zoned at the fixed value 0.1. Every evaluation rollout initializes the gait
clock at phase zero. Neutral supplies a zero standing request, but the effective
policy command remains moving until the next scheduled half-cycle boundary,
even if neutral is requested during double support. The boundary step remains a
gait step; standing begins on the following step. Each supported nonzero
direction starts at the checkpoint's minimum moving speed and scales to its
command bound. The controller index is fixed at zero. A fixed `command` uses
pelvis-frame `[lin_vel_x, lin_vel_y, ang_vel_yaw]`. Bello forward/right is
`+Y`/`+X`, while Bello Mini forward/right is `+X`/`-Y`; use `[0, +v, 0]` and
`[+v, 0, 0]`, respectively, for straight forward evaluation. The command is
passed through without clipping or range validation, so it should remain within
the checkpoint's trained bounds. Gamepad motion remains semantic
forward/right/yaw and is projected into the selected robot's pelvis axes.

Evaluation validates the checkpoint task, policy and critic observation
sizes/layouts, action size/layout, asset configuration, and environment
metadata. The explicit motion payload field is the only physical-model
ablation; no reward or other trained-robot overrides are accepted.

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
| `asset` | `bello` or `bello_mini`; `bello` | all | Select the robot contract. Import operations require its motion/GMR profile; both registered assets provide one. |
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

For direct SMPL-X conversion, use the repository wrappers documented under
[Clean GMR conversion](twist.md#clean-gmr-conversion). Those wrappers prepare
and inject the selected generated robot model; standalone GMR robot selection
does not include Bello Mini.

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
| `asset` | robot asset; `bello` | Select the robot and its motion/GMR profile for provenance validation and packing. |
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
| `experiment` | `smoke`, `bello_locomotion`, `bello_mini_locomotion`, `bello_mini_standing`, `motion`, `twist_teacher`, `twist_student`, or `twist2`; `bello_locomotion` | Select the same maintained profile used by `bello-train`. |
| `motion_path` | path or `null`; `null` | Required for motion jobs. Do not supply this field for other jobs. The clip must exist in the selected remote commit. |
| `dataset_path` | path or `null`; `null` | Required local schema-3 `index.json` for TWIST or TWIST2. Its five portable dataset files are transferred with resumable `rsync` and verified remotely. |
| `teacher_checkpoint_path` | path or `null`; `null` | Required local typed teacher checkpoint for `twist_student`; rejected by every other stage. |
| `warm_start_checkpoint_path` | path or `null`; `null` | Optional local typed TWIST2 checkpoint to transfer. Its dataset fingerprint must differ while its robot, GMR, joint/body ordering, task, and model contract remain compatible. |
| `allow_git_commit_mismatch` | boolean; `false` | Permit a staged student teacher or TWIST2 warm-start checkpoint from a different clean source commit. Remaining compatibility checks stay enforced. |
| `total_timesteps` | positive integer or `null`; `null` | Override the selected remote profile's transition budget. |
| `learning_rate` | positive float or `null`; `null` | Override the selected remote profile's initial learning rate. |
| `end_effector_payload_mass_kg` | nonnegative float or `null`; `null` | Motion only. Forward a symmetric per-hand payload mass to both the remote validation run and full training run. |
| `max_price` | positive float or `null`; `null` | Maximum total hourly rental price. A positive value is required, including for dry-run offer queries. |
| `branch` | string or `null`; `null` | Required remote `origin` branch. A null value is rejected before rental. |
| `run_name` | string or `null`; `null` | Explicit lifecycle/W&B run label. Otherwise a unique name is generated from the profile, branch, commit, and current UTC timestamp. |
| `duration_hours` | positive float; `6.0` | Hard job timeout. The progress timeout is capped at 30 minutes and never exceeds this duration. |
| `ssh_timeout_seconds` | positive integer; `900` | Maximum time to wait for a newly running offer to accept the registered SSH key. An unreachable offer is destroyed and the next candidate is tried. |
| `skip_confirmation` | boolean; `false` | Skip the final rental confirmation for unattended operation. |
| `dry_run` | boolean; `false` | Resolve Git state and print the offer query without reading credentials or renting. |
| `preferred_countries` | ISO country-code list; `[KR, JP, TW, HK, SG]` | Rank nearby offers ahead of globally better marketplace value. |
| `min_upload_mib_per_second` | positive float; `5.0` | Reject and destroy a TWIST instance whose direct-SSH upload probe falls below this rate. |
| `route_probe_mib` | positive integer; `16` | Payload size for the pre-transfer direct-SSH upload probe. |

The selected profile, `pyproject.toml`, `uv.lock`, and any motion clip are
verified in the exact `origin/<branch>` commit before rental. Ignored local
motions are never uploaded implicitly. TWIST transfers only the five files
named by its validated dataset contract and any explicitly selected teacher or
TWIST2 warm-start checkpoint directory. Credentials come from `VAST_API_KEY` or
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

Run the headless native-MuJoCo RobStride hardware-interface emulator. It exposes
the selected robot's configured SocketCAN buses, EBIMU stream, and PoseService
while stepping the canonical model in real time. It opens no graphical windows.

Configuration source: `configs/commands/emulator.yaml`.

| Field | Type and default | Meaning |
| --- | --- | --- |
| `asset` | `bello` or `bello_mini`; `bello` | Select the reviewed robot contract. Mini uses its own 28-motor physical/vCAN description. |
| `bus_interfaces` | list of strings; `[can0=vcan0,can1=vcan1]` | Map configured logical bus names to Linux SocketCAN interfaces. Names must be unique and use `name=interface` form. |
| `command_latency_seconds` | nonnegative float; `0.01` | Delay MIT motor-control frames before applying them to MuJoCo. Lifecycle, parameter, and stop traffic remains immediate; use `0` to disable the delay. |
| `motor_kp_scale` | positive float; `1` | Emulator-only multiplier on decoded MIT position gains. Use controlled sweeps to diagnose effective stiffness mismatch; `1` preserves the canonical plant. |
| `motor_kd_scale` | nonnegative float; `1` | Emulator-only multiplier on decoded MIT velocity gains. Use controlled sweeps to diagnose effective damping mismatch; `1` preserves the canonical plant. |
| `video_path` | path or `null`; `null` | Capture free-torso emulator state at 30 FPS and render it to MP4 when the emulator exits. Guided setup is omitted so recording does not disturb the real-time loop. |
| `end_effector_payload_mass_kg` | nonnegative float; `0` | Add this many kilograms to each sphere hand. Use the payload recorded by the motion-policy checkpoint when emulating a weighted policy. |

The emulator is checkout-only. It selects the canonical `lower_body` and
`upper_body` groups from the chosen asset's hardware description, uses
`deps/bello-stack/descriptions/robstride/models.yaml` for protocol and hardware
limits when present, and derives direct and differential transmissions from the
canonical URDF. ROBSTRIDE_02 electrical telemetry that is absent from the
catalog is reported as zero. Only the logical-to-SocketCAN interface mapping is
local. The emulator applies MIT impedance control, publishes the configured IMU
serial stream and PoseService, and paces simulation against wall-clock time. Its
torso guide is always engaged at startup; Enter releases it, and another Enter
resets the robot and re-engages the guide.

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

The command always selects JAX's CPU backend before loading a hardware
runner. Inherited `JAX_PLATFORMS` and `JAX_PLATFORM_NAME` values are ignored;
there is no hardware backend configuration field.

Configuration source: `configs/commands/hardware.yaml`.

| Field | Type and default | Applies to | Meaning |
| --- | --- | --- | --- |
| `task` | `locomotion`, `motion`, or `twist2`; `locomotion` | all | Select the checkpoint and observation contract. |
| `checkpoint_path` | path or `null`; `null` | all | Required checkpoint directory. Its recorded Git commit normally equals the current clean checkout. |
| `standing_checkpoint_path` | path or `null`; `null` | motion | Required locomotion checkpoint used before and after motion tracking. |
| `allow_git_commit_mismatch` | boolean; `false` | all | Permit a different clean checkout. Evaluation/hardware loading removes retired generated-asset fields replaced by the canonical URDF, rejects retired joint selectors that were enabled, and supplies explicit historical PPO and no-op reward/randomization defaults for older locomotion metadata. This does not disable other task, shape, asset, motion-content, source-service, or hardware preflight checks. |
| `endpoint` | string; `http://127.0.0.1:50051` | all | `bello-hardwared` endpoint. |
| `remote_endpoint` | string or `null`; `null` | locomotion | Required plaintext URL of the visualizer's `RemoteCommandService`, for example `http://visualizer-host:50054`. |
| `remote_staleness_ms` | positive integer; `200` | locomotion | Maximum age of the latest remote-command sample before the command becomes zero and the stream reconnects. |
| `motion_path` | path or `null`; `null` | motion | Override the checkpoint's bound motion location. Its schema and content hash must still match. |
| `end_effector_payload_mass_kg` | nonnegative float or `null`; `null` | motion | Declare the physical mass carried by each hand. `null` uses the checkpoint value; an explicit mismatch is rejected before control is acquired. |
| `duration_seconds` | positive float or `null`; `null` | all | Maximum post-`RUN` policy duration. `null` runs until operator exit, interruption, or a safety failure. |
| `ramp_seconds` | positive float or `null`; `null` | all | Override the initial supported-pose ramp time. `null` uses 10 seconds. Locomotion always uses the checkpoint's nominal arm pose during ramp and hold; TWIST2 combines the repository-owned relaxed-standing SMPL-X retarget with its canonical Bello stance. |
| `gain_scale` | float in `(0, 1]`; `1.0` | all | Multiply every commanded motor `kp` and `kd` by this value during start, ramp, and policy control. Fault damping is unchanged. |
| `max_linear_acceleration` | positive float or `null`; `null` | all | Override the IMU acceleration-norm safety limit in m/s². `null` uses 100 m/s². Exceeding the limit immediately enters damped shutdown. |
| `policy_transition_seconds` | positive float or `null`; `null` | motion, twist2 | Duration of each policy-target transition, including the TWIST2 normal-hold-to-standing-policy handoff. `null` uses 0.2 seconds. |
| `max_motion_tilt_radians` | positive float or `null`; `null` | motion, twist2 | Maximum robot tilt during transitions and active motion. `null` uses 70 degrees (approximately 1.222 radians); standing keeps its 0.70-radian limit. |
| `action_lpf_cutoff_hz` | positive float or `null`; `null` | motion | First-order low-pass cutoff for the motion action sent to the motors and fed back as `last_action`. `null` uses 15 Hz. |
| `status_interval_seconds` | positive float or `null`; `null` | all | Emit periodic policy-state diagnostics at this interval. Locomotion status reports the effective `command` and `standing_transition_pending`. `null` prints only preflight, phase changes, and summaries. |
| `policy_log_path` | path or `null`; `null` | all | JSONL destination for every executed policy observation and action. `null` selects `policy_steps.jsonl` inside the unique Hydra run directory. An explicit path must not already exist. |
| `arm` | boolean; `false` | all | `false` performs read-only preflight and sends no command. |
| `allow_provisional_hardware` | boolean; `false` | locomotion | Explicit acknowledgement for a robot contract marked provisional. It is rejected for the reviewed Bello and Bello Mini contracts. |
| `upper_body_tracking_enabled` | boolean; `true` | locomotion | Live RTMW/GMR targets every held arm coordinate on Bello or Bello Mini. `false` skips RTMW, SMPL-X, and GMR and actively holds those joints at their nominal targets. |
| `rtmw_endpoint` | gRPC host:port; `127.0.0.1:50055` | locomotion | Pinned RTMW-SMPLX-Teleop `SkeletonService` endpoint. |
| `rtmw_tls_root_certificate_path` | path or `null`; `null` | locomotion | CA certificate for one-way server TLS. `null` uses plaintext and is appropriate only on loopback or a trusted LAN. |
| `rtmw_tls_server_name_override` | string or `null`; `null` | locomotion | TLS certificate name override. It is rejected without a CA certificate. |
| `rtmw_connect_timeout_seconds` | positive float; `5.0` | locomotion | Deadline for establishing the RTMW gRPC channel. |
| `rtmw_rpc_timeout_seconds` | positive float; `2.0` | locomotion | Deadline for stream-info and status RPCs. |
| `rtmw_reference_qualification_frames` | positive integer; `5` | locomotion | Distinct successive safe references required after `RUN` before starting or resuming live arm tracking. Until then, the arms remain at their nominal pose. |
| `rtmw_min_joint_confidence` | float in `[0,1]`; `0.25` | locomotion | Minimum position confidence for every GMR target joint. |
| `rtmw_min_rotation_confidence` | float in `[0,1]`; `0.10` | locomotion | Minimum derived-rotation confidence for the pelvis, spine3, and palm-constrained wrist rotations used by live GMR. |
| `rtmw_max_required_rotation_residual_degrees` | float in `(0,180]`; `30.0` | locomotion | Maximum usable rotation-fit residual. Pelvis, spine3, and palm-constrained wrists must also have rank 2; rank-one shoulder and elbow frames retain their transported twist and remain usable. |
| `rtmw_max_pipeline_latency_ms` | positive float; `250.0` | locomotion | Maximum accepted RTMW capture-to-output processing latency. |
| `rtmw_transport_latency_budget_ms` | nonnegative float; `10.0` | locomotion | Conservative transport contribution added to reference age because producer and robot hosts need not share a monotonic clock. |
| `rtmw_max_source_interval_seconds` | positive float; `0.50` | locomotion | Maximum interval between successive accepted RTMW source frames before live GMR is re-anchored. This tolerates brief confidence-gated gaps while remaining inside the separate 1-second reference-age envelope. |
| `rtmw_camera_to_world_quaternion_xyzw` | normalized four-float tuple; `[0.5,-0.5,0.5,-0.5]` | locomotion | Calibrated camera-to-world rotation. The default maps OpenCV forward/right/down to Bello forward/right/down. |
| `rtmw_align_initial_heading` | boolean; `true` | locomotion | Rotate the acquired target's initial anatomical heading to world forward, relative to the canonical SMPL-X-to-world pelvis basis. |
| `rtmw_align_initial_floor` | boolean; `true` | locomotion | Use the acquired target's initial lower foot as the fixed world floor. |
| `rtmw_human_height_m` | positive float; `1.75` | locomotion | Subject height used and advertised by RTMW's metric depth reconstruction; it must equal RTMW's `reconstruction.subject_height_m`. Published upper-body articulation is normalized to the neutral SMPL-X model rather than this height. |
| `rtmw_gmr_task_profile` | string; `live_upper_body` | locomotion | Orientation-led GMR task profile for RTMW arm tracking. This is currently the only accepted value; Mini uses shoulder, elbow, and palm frames while fixed robot links determine arm positions. |
| `fastsam_endpoint` | gRPC host:port; `127.0.0.1:50055` | twist2 | Corrected `fastsam3dbody-serve` endpoint, after the local hardware, emulator, pose, and visualization service ports. |
| `fastsam_tls_root_certificate_path` | path or `null`; `null` | twist2 | CA certificate for one-way server TLS. `null` uses plaintext and is appropriate only on loopback or a trusted LAN. |
| `fastsam_tls_server_name_override` | string or `null`; `null` | twist2 | TLS certificate name override. It is rejected without a CA certificate. |
| `fastsam_connect_timeout_seconds` | positive float; `5.0` | twist2 | Deadline for establishing the gRPC channel. |
| `fastsam_rpc_timeout_seconds` | positive float; `2.0` | twist2 | Deadline for stream-info and status RPCs. |
| `fastsam_reference_ready_timeout_seconds` | positive float; `15.0` | twist2 | Time allowed after each `MOTION` confirmation while normal-standing commands continue. |
| `fastsam_reference_qualification_frames` | positive integer; `5` | twist2 | Number of distinct, successively received safe references required for live tracking. |
| `fastsam_min_tracking_confidence` | float in `[0,1]`; `0.5` | twist2 | Minimum whole-person confidence. |
| `fastsam_min_joint_confidence` | float in `[0,1]`; `0.25` | twist2 | Minimum confidence for every GMR target joint. |
| `fastsam_max_pipeline_latency_ms` | positive float; `120.0` | twist2 | Maximum server-reported submission-to-output latency. |
| `fastsam_transport_latency_budget_ms` | nonnegative float; `10.0` | twist2 | Conservative transport contribution added to every reference age because server and robot hosts need not share a monotonic clock. |
| `fastsam_camera_to_world_quaternion_xyzw` | normalized four-float tuple; `[0.5,-0.5,0.5,-0.5]` | twist2 | Calibrated camera-to-world rotation. The default maps OpenCV forward/right/down to Bello forward/right/down. |
| `fastsam_align_initial_heading` | boolean; `true` | twist2 | Rotate the acquired target's initial pelvis heading to world forward. |
| `fastsam_align_initial_floor` | boolean; `true` | twist2 | Use the acquired target's initial lower foot as the fixed world floor. |
| `smplx_neutral_model_path` | path or `null`; `null` | locomotion, twist2 | Required licensed `SMPLX_NEUTRAL.npz`, used locally to validate the body-joint contract and pose its skeleton with the repository-owned relaxed-standing sample. |
| `smplx_human_height_m` | positive float; `1.66` | locomotion, twist2 | Canonical SMPL-X source height supplied to the pinned GMR scale table. For RTMW this stays independent of the measured subject height. |
| `left_elbow_yaw_offset_radians` | finite float; `0.0` | locomotion | Bello-only offset added to the left GMR elbow-yaw target before clipping. Set `1.57079632679` for positive 90 degrees. Mini rejects nonzero elbow-yaw offsets. |
| `right_elbow_yaw_offset_radians` | finite float; `0.0` | locomotion | Bello-only offset added to the right GMR elbow-yaw target before clipping. Set `1.57079632679` for positive 90 degrees. Mini rejects nonzero elbow-yaw offsets. |

Task-specific fields must not be explicitly supplied to an inapplicable task,
even if the supplied value equals the documented default.
Locomotion always requires `remote_endpoint`. When
`upper_body_tracking_enabled=true`, it also requires
`smplx_neutral_model_path`. Its
visualizer-hosted physical controller uses the same stick mapping as
`bello-eval`; R1 aborts. Controller or service loss commands zero, reconnects
automatically, and requires a neutral controller before accepting nonzero
intent again. The visualizer service must produce a fresh sample before the
hardware daemon is contacted. The joystick commands Bello's 13-coordinate or
Mini's 14-coordinate locomotion policy. The selected robot's RTMW/GMR profile
targets the other 12 Bello or 14 Mini arm coordinates. Hardware locomotion has
no native-controller or fixed-command mode.

With `upper_body_tracking_enabled=false`, explicit RTMW and SMPL-X/GMR
overrides must not be supplied. The runner uses the checkpoint's nominal
targets for every held arm coordinate (12 for Bello, 14 for Mini) while the
joystick continues to command locomotion:

```text
uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_locomotion \
  remote_endpoint=http://visualizer-host:50054 \
  upper_body_tracking_enabled=false
```

Motion additionally requires a locomotion checkpoint selected by
`standing_checkpoint_path` and absolute pelvis pose from
`bello-state-estimatord` PoseService at `127.0.0.1:50053`. The standing and
motion checkpoints must resolve to the same ordered physical-coordinate
mapping.

All armed tasks require the operator to type exact interactive confirmations:
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
TWIST2 uses the same standing-to-motion routine, but normal standing is a
repository-owned relaxed SMPL-X sample retargeted through pinned GMR and live
`MOTION` input is the latest valid Fast-SAM SMPL-X frame. Five distinct safe
frames qualify a new target by default. Its complete setup and loss behavior
are documented in [Physical hardware](hardware.md#twist2-mode).
`duration_seconds` measures only the post-`RUN` policy flow; the ramp and
`HOLD`/`RUN` confirmation waits are outside that duration.

Armed runs print the resolved policy-log path before acquiring control. The
first JSONL record describes checkpoint paths, observation layouts, stack
sizes, action names, and physical target names. Every following record contains
one executed policy step with wall and monotonic timestamps, phase, step,
unstacked observation, normalized stacked policy observation, raw and applied
action, and physical target. Locomotion
records include the effective velocity command. Motion records cover the standing
policy, both policy transitions, and motion tracking; transition records contain
both the policy target and the blended target actually sent. File encoding and
flushing run on a dedicated background thread so normal disk I/O does not block
the control loop. These complete 50 Hz logs can grow by hundreds of megabytes
per hour, depending on the checkpoint observation-stack size.

Hardware targets use the proportional and derivative gains stored in checkpoint
metadata.

Motor velocity observations come from `bello-hardwared`, whose deployment YAML
sets `runtime.motor_velocity_filter.cutoff_hz` (10 Hz by default). This filter
cannot be changed by `bello-run-hardware`; `null` and zero are invalid daemon
settings. For experiments that need effectively unfiltered observations, use a
high positive cutoff such as 1000 Hz and restart `bello-hardwared`. This only
changes published motor velocity telemetry; daemon health and safety checks
continue to use raw protocol feedback.

IMU state frames are filtered inside `bello-hardwared` before either hardware
runner consumes them. The deployment YAML's `runtime.imu_filter` section sets
independent orientation, gyro, and acceleration cutoffs, all 10 Hz by default.
The filter advances on every 200 Hz EBIMU device timestamp before the runner
selects the latest sample at 50 Hz; it cannot be configured through
`bello-run-hardware`.

Motion-policy base angular velocity comes from the filtered root-local gyro field
published by `bello-state-estimatord` through `PoseService`. The
hardwared-filtered gyro in the hardware state frame remains independently
validated against the configured hardware angular-velocity safety limit. The
canonical estimator configuration applies an additional 15 Hz gyro low-pass
cutoff for its PoseService output.

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
