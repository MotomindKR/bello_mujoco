# ASAP Single-Foot Balance

This directory contains the fixed-head Bello retarget of ASAP's level-1
single-foot balance motion used by the reproducible Vast.ai training run.

- Source repository: https://github.com/LeCAR-Lab/ASAP
- Source revision: `df5320cc47dd8cad97961bdfabfe402dd62ad999`
- Source clip: `video_single_foot_balance_level1_filter_amass.npz`
- GMR configuration: fixed waist roll/pitch, fixed neck yaw/head pitch
- Frames: 218 at 30 FPS
- Active Bello joints: 25
- Tracked Bello bodies: 14
- Primitive-box self-collision frames: 0
- Archive SHA-256: `869df4c491f19e3aee8d3bdbcabd99cedfd29fe889d152ea93d9e890936e3029`

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
