# Custom motion references

`output_bello_beyondmimic.npz` is a private BeyondMimic reference generated
from the local SOMA `output.bvh` clip. It contains 298 frames at 30 Hz and uses
the fixed-waist, fixed-head, sphere-hand 25-joint Bello contract with the
14-body BeyondMimic tracking profile.

Provenance:

- source BVH SHA-256:
  `b2fe34bcafcc1bf8d3272827c0a75224947e8089fc39bd55f55928e1259aa4f1`
- GMR Bello pickle SHA-256:
  `d9333d94c2799958d8c62d6ccde565aba1d47c6bf70bae9288d155f36b1975f0`
- canonical NPZ SHA-256:
  `9198968c16a98d72424cd64f80120b1b53179cdc71949a37611e70355767f88f`

GMR's generated Bello asset uses ankle-pitch coordinates rebased by -15
degrees. The current `bello_mujoco` source model retains the original hardware
coordinate. Before canonical import, both ankle-pitch coordinates were
therefore increased by exactly 15 degrees. This maps the GMR range
`[-45 deg, +45 deg]` to the source-model range `[-30 deg, +60 deg]` without
changing the physical pose. The strict importer then verified joint limits and
rejected no self-collision frames.

Open-loop physical feasibility replay terminates on root-height error at 0.28
seconds. The clip therefore requires learned balance feedback and should not be
treated as a feedforward-safe hardware trajectory.
