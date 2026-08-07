# Curated localized ASAP motions

These ten Bello retargets were selected for compact, leg-active demonstration
training. They remain within roughly 0.13--0.53 m of planar span and finish
close to their initial position.

- Source corpus: the pinned ASAP SMPL motions documented in
  `motions/asap_fixed_head/README.md`
- Retargeter: GMR commit `741d521686f47c6b0710997d1fe903a6cc5c2f5e`
- Bello limits: synchronized with `deps/bello-stack` commit
  `5c56030b757c48513afef16c1d5950756e84c87e`
- Format: canonical `bello-motion` version 1, 30 FPS, fixed head and waist
  roll/pitch, 25 active joints, 14 tracked bodies

The batch contains two step-and-return clips, two squats, two kicks, two
single-foot balance motions, the compact Bolt pose, and the Spider-Man crouch.

All archives pass schema, finite-value, and current hard joint-limit
validation. They were imported with self-collision rejection disabled because
the fresh GMR arm solution intersects Bello's primitive box model in portions
of every clip. Training therefore retains the maintained undesired-contact
penalty; these runs are exploratory and should not be promoted directly to
hardware without policy evaluation and collision review.
