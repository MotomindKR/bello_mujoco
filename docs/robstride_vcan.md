# RobStride HIL

The HIL runner presents the canonical 60 kg sphere-hand Bello morphology as
RobStride devices on Linux SocketCAN while native MuJoCo supplies dynamics.

```bash
./scripts/setup_hil.sh
uv run bello-run-robstride
```

The fixed inventory is:

- `configs/robstride/lower_body.yaml`
- `configs/robstride/upper_body_sphere_hands.yaml`
- `configs/robstride/motor_kinds.yaml`

Torque limits are always enforced from the motor catalog. Differential ankles
remain two coupled tendon motors per side. The runner also exposes the
pelvis-mounted IMU serial stream and world pelvis PoseService used by the
visualizer and motion hardware runner.

Useful public controls:

```bash
uv run bello-run-robstride headless=true real_time=false max_steps=1000
uv run bello-run-robstride release_after_seconds=2
uv run bello-run-robstride mode=prescribed_gains
```

`mode=prescribed_gains` runs the generated nominal targets and impedance gains
directly in MuJoCo without a CAN controller. It is intended for morphology and
gain commissioning, not policy evaluation.

The HIL path validates motor coverage, CAN identities, protocol ranges,
transmission ownership, torque enforcement, IMU data, PoseService frames, and
torso-guide behavior at initialization.
