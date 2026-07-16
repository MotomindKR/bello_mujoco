# ASAP Single-Foot Balance

This directory contains fixed-head Bello retargets of ASAP single-foot balance
motions used by reproducible Vast.ai training runs.

- Source repository: https://github.com/LeCAR-Lab/ASAP
- Source revision: `df5320cc47dd8cad97961bdfabfe402dd62ad999`
- GMR configuration: fixed waist roll/pitch, fixed neck yaw/head pitch
- Active Bello joints: 25
- Tracked Bello bodies: 14

| Source clip | Frames | Primitive-box self-collision frames | Archive SHA-256 |
| --- | ---: | ---: | --- |
| `video_single_foot_balance_level1_filter_amass.npz` | 218 at 30 FPS | 0 | `869df4c491f19e3aee8d3bdbcabd99cedfd29fe889d152ea93d9e890936e3029` |
| `video_single_foot_balance_level4_filter_amass.npz` | 265 at 30 FPS | 0 | `ccc079a5fb25a0275d53e152118664f3e3083e5d5385d12c9c40a332be979b08` |

The canonical archive was generated from the local ASAP checkout with:

```bash
cd /home/gmmyung/GMR
nix develop -c python scripts/prepare_asap_bello_dataset.py --jobs 2

cd /home/gmmyung/bello_mujoco_wholebody
nix develop -c uv run bello-import-gmr-motion-dataset \
  /home/gmmyung/GMR/motion_data/asap_bello_fixed_head \
  motions/local/asap_fixed_head \
  --jobs 2
```
