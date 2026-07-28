# ASAP Single-Foot Balance

This directory contains fixed-head Bello retargets of ASAP single-foot balance
motions used by reproducible Vast.ai training runs.

- Source repository: https://github.com/LeCAR-Lab/ASAP
- Source revision: `df5320cc47dd8cad97961bdfabfe402dd62ad999`
- Source license: [MIT](https://github.com/LeCAR-Lab/ASAP/blob/df5320cc47dd8cad97961bdfabfe402dd62ad999/LICENSE)
- GMR configuration: fixed waist roll/pitch, fixed neck yaw/head pitch
- Active Bello joints: 25
- Tracked Bello bodies: 14

| Source clip | Frames | Primitive-box self-collision frames | Archive SHA-256 |
| --- | ---: | ---: | --- |
| `video_single_foot_balance_level1_filter_amass.npz` | 218 at 30 FPS | 0 | `869df4c491f19e3aee8d3bdbcabd99cedfd29fe889d152ea93d9e890936e3029` |
| `video_single_foot_balance_level3_filter_amass.npz` | 271 at 30 FPS | 0 | `1d1456b5fb61a5e2037ffdf7c93506915635d8227824061a689612a3b3022621` |
| `video_single_foot_balance_level4_filter_amass.npz` | 265 at 30 FPS | 0 | `ccc079a5fb25a0275d53e152118664f3e3083e5d5385d12c9c40a332be979b08` |

The archives were generated from the pinned ASAP checkout through GMR's
fixed-head Bello pipeline. Set `GMR_REPO` to that checkout and run the first
command there, then run the importer from this repository's root:

```bash
export BELLO_MUJOCO_REPO=$PWD
export GMR_REPO=/path/to/GMR

cd "$GMR_REPO"
nix develop -c python scripts/prepare_asap_bello_dataset.py --jobs 2

cd "$BELLO_MUJOCO_REPO"
nix develop -c uv run bello-motion \
  operation=import_dataset \
  source="$GMR_REPO/motion_data/asap_bello_fixed_head" \
  output=motions/local/asap_fixed_head \
  jobs=2
```

The complete 51-clip import remains ignored under `motions/local/`. These
tracked examples are byte-identical to the corresponding canonical imports;
the table hashes are the distribution integrity check.
