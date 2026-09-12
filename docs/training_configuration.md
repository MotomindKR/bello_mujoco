# Configuration

Values belong in YAML, not duplicated documentation tables.

| Location | What to edit |
| --- | --- |
| `configs/commands/*.yaml` | Public CLI fields |
| `configs/profiles/train/*.yaml` | Experiment selection and overrides |
| `configs/train/default.yaml` | Locomotion PPO, environment, rewards, logging |
| `configs/train/locomotion.yaml` | Shared production locomotion recipe |
| `configs/train_motion/default.yaml` | BeyondMimic training |
| `src/bello_mujoco/robots/*.yaml` | Topology, nominal pose, motor gains, tracked bodies |
| `deps/bello-stack/descriptions/` | Canonical geometry and hardware configuration |

Maintained experiments: `smoke`, `bello_locomotion`, `bello_mini_locomotion`,
`bello_mini_standing`, `motion`. The CLI accepts its public top-level overrides;
edit the selected recipe for nested reward/physics changes.

```bash
uv run bello-train --cfg job
uv run --group training bello-train experiment=bello_mini_locomotion seed=1 num_envs=4096
```

Hydra documents decode into frozen dataclasses, validating fields, dimensions,
ranges, and incompatible combinations before simulation. Add settings to their
owning dataclass and YAML together, without aliases or parallel CLI defaults.

## Locomotion components

- `gait.py`: cadence, directional start, command slew, delayed stop, standing blend.
- `actions.py`: motor targets and calibration offsets.
- `observations.py`: actor/critic assembly, noise, filtering.
- `randomization.py`: spawn and pushes.
- `rewards.py`: tracking, alignment, crossing, clearance, contact, force, smoothness.
- `locomotion_state.py` and `locomotion_config.py`: rollout state and typed settings.

The motor-coordinate action convention requires fresh training. Keep training,
native evaluation, and hardware observation/filter conventions aligned.

Runtime initialization validates the selected robot, including default-pose
collisions. Unit tests exercise algorithms and invalid-input handling with
synthetic inputs. Do not test a robot's YAML values, joint count, nominal angles,
or geometry inventory as fixed expectations.
