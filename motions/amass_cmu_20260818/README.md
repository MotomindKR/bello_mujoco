# CMU deep-lunge motion

`cmu_144_17_deep_lunge.npz` is a canonical Bello motion clip derived from
AMASS CMU subject 144, trial 17 (`Lunges`). The tracked interval is source time
5.3--8.7 seconds: standing, one deep forward lunge, and a return to standing.

The source SMPL-X archive is not redistributed. Its SHA-256 is
`4235a859742495c0b0c20dbd4b25bf728c925a6f6bd225da751e34394170a72d`.
The clip was retargeted at 30 Hz with MotomindKR/GMR revision `741d521` and the
motion-tracking viewer model generated from the repository's pinned
`bello-stack` revision `4c9b724`.

The canonical lower-body and root trajectory is unchanged. Both arm
trajectories were subsequently replaced with a constant carrying pose: 0.20 rad
shoulder roll, 0.15 rad elbow pitch, and zero shoulder pitch/yaw, elbow yaw,
and wrist pitch. This holds the arms downward with a small symmetric outward
offset. Derived velocities and tracked-body kinematics were regenerated. The
edited reference has zero box-model self-collision frames; edit provenance and
the pre-edit archive checksum are embedded in the NPZ.

The intended load-training configuration is `end_effector_payload_mass_kg=4.5359237`,
which represents 10 lb attached to each sphere hand (9.0718474 kg total
payload). The mass is a generated-asset setting rather than motion data, so the
same clip remains usable for explicit payload ablations.

Train the intended task with:

```bash
uv run bello-train \
  experiment=motion \
  motion_path=motions/amass_cmu_20260818/cmu_144_17_deep_lunge.npz \
  end_effector_payload_mass_kg=4.5359237 \
  checkpoint_name=motion_deep_lunge_10lb_per_hand_seed0
```

Native evaluation uses the checkpoint payload by default. Supplying
`end_effector_payload_mass_kg=<value>` to `bello-eval task=motion` performs an
explicit load ablation. The same field on `bello-run-hardware task=motion`
declares the actual per-hand load and must match the checkpoint before arming.
