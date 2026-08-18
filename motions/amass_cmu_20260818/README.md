# CMU deep-lunge motion

`cmu_144_17_deep_lunge.npz` is a canonical Bello motion clip derived from
AMASS CMU subject 144, trial 17 (`Lunges`). The tracked interval is source time
5.3--8.7 seconds: standing, one deep forward lunge, and a return to standing.

The source SMPL-X archive is not redistributed. Its SHA-256 is
`4235a859742495c0b0c20dbd4b25bf728c925a6f6bd225da751e34394170a72d`.
The clip was retargeted at 30 Hz with MotomindKR/GMR revision `741d521` and the
motion-tracking viewer model generated from the repository's pinned
`bello-stack` revision `4c9b724`.

Canonical import records 35 self-collision frames, with a maximum 0.020158 m
box overlap between the right wrist and right hip. This is an upper-body proxy
collision, not a leg collision; the diagnostic remains embedded in the NPZ.
