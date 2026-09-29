# SONIC transfer validation: 2026-09-28

Validation used the RTX 5090 at `gmmyung-ubuntu` in an isolated checkout under
`outputs/sonic-validation-20260928/source`; the host's existing source edits were
left untouched. The tested source snapshot is
`b20217f21ddc9d722a8ed90279732facbf09fdfc`, retained locally as
`sonic-validation/validated`. It records the required Any2Any submodule pin.
The implementation remains on `feat/sonic-transfer` in the Mac worktree.

The host initially had NVIDIA kernel driver 595.84 with 595.91.07 userspace
libraries. Initial tests used matching, process-local Nix driver libraries.
After CPU validation, the user-approved reboot loaded 595.91.07; `nvidia-smi`
and CUDA-enabled JAX worked without that override. The existing Mac public key
was registered and public-key-only SSH login was verified after reboot.

## Inputs and results

The original NVIDIA checkpoint was downloaded at Hugging Face revision
`6733128a3d8a523b1418b06bca3cdf61c8b0987f`; its SHA-256 is
`e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
The four retained clips were `balance`, `lunge`, `randb`, and `squat` (535,
975, 1231, and 455 frames). `randb` was excluded from the training split only
to exercise evaluation plumbing, not to establish held-out generalization.

- Default CPU suite: 731 passed; 279 opt-in integration cases deselected.
- Native Rust consumer integration: 70 passed, covering locomotion,
  BeyondMimic, SONIC, and bundle validation.
- Rust unit tests: 21 passed; all targets compiled.
- Post-reboot GPU regressions: BeyondMimic sampling/noise/push batched stepping
  and locomotion finite batched reset/step both passed. Real-checkpoint GPU
  parity was also rerun successfully without a driver-library override.
- Lockfile, generated protobuf, Ruff, formatting, and diff checks passed.
- Real-checkpoint PyTorch/JAX GPU parity: encoder maximum absolute error
  `5.96e-7`, FSQ `0`, decoder `4.29e-6`, critic `5.07e-7`, exploration adapter
  `5.96e-8`. Zero-initialized LoRA was included in these comparisons.
- GPU training: 8 environments, 24 steps, two PPO updates, five epochs and four
  minibatches per update. Invalid/nonfinite guards passed; adapters changed and
  frozen backbone weights remained identical.
- Checkpoint restore: all runner-state leaves and restored actor outputs were
  bitwise identical, including optimizer, random keys, histories and sampling.
- JAX peak live allocation was 342,464,256 bytes (about 327 MiB); allocator pool
  peak was 534,773,760 bytes (510 MiB). These are not whole-process VRAM peaks;
  Warp, driver and desktop allocations are additional. Large batches were not
  tested.
- CPU-only Mesa llvmpipe rendering produced 50 frames without termination or
  nonfinite values. First, middle and final frames were visually reviewed.
- JAX, Python ONNX Runtime and Rust playback passed 32 real-clip feedback
  samples covering starts, interior motion, history advances, resets and final
  reference-window clamping. Maximum observation error was `7.23e-7`, raw
  action error `7.16e-6`, and motor-target error `3.14e-6`.

The smoke used the training APIs directly with W&B disabled and no artifact
identity. It did not exercise a production corpus artifact or a Vast rental.
Logs, harnesses, smoke checkpoints, exported bundles and the video are retained
outside Git in `outputs/sonic-validation/` on the Mac. The final snapshot's
artifacts are in its `runtime-artifacts/` subdirectory; earlier diagnostic
attempts are retained separately.
The transferred `policy-interior.zip` SHA-256 was verified on both machines:
`522c65dab6c9d74790eabe65b9836b585558fbc0ce2113377c03c4db27df1ebf`.

## Findings

The real release contains trainer metadata requiring an explicit, inert
allowlist in the restricted PyTorch loader. Unknown pickle globals continue to
fail. The default GPU matrix precision also failed float32 network parity;
SONIC now requests `HIGHEST` precision in backbone, adapter and weight-merging
products, with a regression test independent of process precision defaults.
The native BeyondMimic fixture was updated for the shared reference interface.
Real mid-motion playback also exposed reference-clock differences. SONIC Rust
now uses JAX-compatible float32 frame arithmetic and ties-to-even rounded future
frame offsets. New tests cover long clips and 25 Hz clips; BeyondMimic keeps
its original float64 time sampling.

Do not require bitwise equality after another GPU physics rollout. In the
final smoke, restored-versus-uninterrupted continuation differed by at most
`3.56e-4` in trainable parameters, `0.00109` in qpos and `0.235` in qvel.
Repeated uninterrupted runs also differed, but by less in this run; an earlier
run showed nearly identical restored and repeated-run drift. RNG, reference
time and sampling state matched. Exact continuation-trajectory equivalence is
not established; this diagnostic is distinct from the exact checkpoint
restoration gate and should be revisited in longer stability testing.

## Remaining gates

This is an integration smoke, not a trained or commissioned policy. Still
required are an immutable W&B corpus artifact, a representative train/held-out
corpus, longer stability and throughput testing, full Mini joint-sweep/FK
review, Lima HIL for the operator lifecycle, and later trained-policy hardware
commissioning. SONIC physical arming remains rejected. No robot was armed and
no full transfer training rental was started.

## Subsequent recipe audit and scaling checks

Later worktree changes are **not** the `b20217f` snapshot above. They correct
effective reference PPO behavior: direct-standard-deviation optimization,
trajectory minibatches, sample-variance advantage normalization, per-minibatch
adaptive learning rate, value-loss scaling, and the separately updated pretrained
critic normalizer. The SONIC profile now defaults to 2,048 environments. The
earlier exact restore/export results must be rerun against this updated recipe
before calling the production workflow fully validated.

The RTX 5090 has 32,607 MiB visible device memory. Measurements below include
the desktop, driver and Warp allocations, with 250 ms `nvidia-smi` sampling;
they are observed peaks, not guaranteed memory upper bounds. All runs still
used only the four retained smoke clips, not the future eight-hour AMASS bank.

| Environments | Updates | Whole-device peak | Median warm update | Result |
| --- | --- | --- | --- | --- |
| 1,024 | 31 | 9,379 MiB | 0.737 s | Passed finite/invalid guards |
| 2,048 | 101 attempted | Not recorded on failure | Not finalized | Raw gradient-norm guard failed at update 86 |
| 2,048 | 101, repeat | 16,049 MiB | 1.157 s | Passed; final raw gradient norm about 5,606 |

The repeat is not evidence that the first stability failure is resolved. The
raw-norm guard remains enabled; a representative-corpus run and investigation
are required. The median 2,048-environment update corresponds to about 42,500
environment steps/s on this smoke corpus. First-update compilation took about
106 seconds, with another approximately 13-second second-update compilation
spike. Host peak RSS was about 6.8 GiB. Do not extrapolate these measurements into
a guaranteed convergence time or full-corpus VRAM requirement.

The smoke corpus was uploaded, byte-verified, to the existing private project as
`gyungmin/bello-mujoco/mini-sonic-integration-smoke-20260928:v0`.
Its manifest SHA-256 is
`43a12f967171e10eaaee3ff00d94fabae568da3230cc48181686898fdb10a927`.
This establishes the artifact path, not a held-out generalization result.

### AMASS preparation pilot

The user selected AMASS instead of BONES-SEED and confirmed non-commercial
research use. The corrected offline pilot accepted 24 clips out of 62 candidates:
191.0 seconds of training motion and 5.78 seconds of evaluation motion. It checks
the actual Mini transmission limits, joint speed, floor clearance and collision
penetration. An initial pilot incorrectly lacked terrain; it was superseded,
not incorporated into the full corpus. Three corrected reference clips were
rendered on the Ubuntu CPU renderer and inspected visually. These were reference
renders, not trained-policy rollouts.

The full preparation uses the subsequently tightened subject grouping (capture
session aliases and shared MPI IDs), so its split need not match the pilot.
The eight-hour corpus, its final artifact, representative GPU stability checks,
updated production CLI save/resume/export checks, full geometry review and HIL
remain completion gates. No AMASS policy has been trained or commissioned yet.

After adding the preparation and corpus validation tests, the default CPU suite
passed **740 tests**, with 2 skipped and 279 opt-in cases deselected. Ruff,
formatting, lockfile and diff checks passed. The two skipped tests remain skips,
not claimed passes or substitutes for the outstanding integration gates.
