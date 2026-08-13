# AMASS demo training clips (2026-08-13)

These canonical Bello motion archives were selected for the four requested
Vast.ai training runs. Each motion was retargeted from the named AMASS
stage-II sequence with the local GMR Bello mapping, resampled to approximately
30 Hz, imported with the 14-body BeyondMimic tracking profile, and validated
against the fixed-waist, fixed-head motion-training model.

Only `SMPLX_NEUTRAL.npz` was available locally, so the retarget step uses that
body model while preserving the source AMASS poses, translations, shape
coefficients, and timing.

| Canonical clip | AMASS stage-II source | Raw source SHA-256 | Collision audit |
| --- | --- | --- | --- |
| `kit_chacha.npz` | `KIT/572/dance_chacha01_stageii.npz` | `48375c3e1cb88a85ab62ea161cacd7e9dd53d7efe2e0f61b38aac42fa61ee50f` | 17 / 206 frames; 55.35 mm maximum penetration |
| `kit_wave_right.npz` | `KIT/291/wave01_stageii.npz` | `a56a3ee848a82a418637444a0bae0ca5ef24db44200caed275f4cc95f5c8ce3b` | 39 / 93 frames; 10.70 mm maximum penetration |
| `accad_block_left_high.npz` | `ACCAD/MartialArtsWalksTurns_c3d/E11_-_block_left_high_stageii.npz` | `4e2ff16534e01363a108b5855729a4bf51fb5a67ddc80e020764cb52b2d7cd61` | 1 / 80 frames; 4.33 mm maximum penetration |
| `kit_point_forward.npz` | `KIT/291/point_at01_stageii.npz` | `00c423321be55b04f5ba659dd4c73484dc7d5e72a014e29d4eb1ccfbe50f1073` | collision-clean |

The collision audit is retained rather than filtered. These are training
inputs, not evidence that a resulting policy is hardware-ready.
