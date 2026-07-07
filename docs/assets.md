# Asset Pipeline

## Source Assets

The repo only reads robot descriptions from tracked local source assets:

```text
assets/source/bello/full_body/
assets/source/bello/lower_body/
assets/source/bello/upper_body/
```

Valid asset names:

```text
full_body
lower_body
upper_body
```

## Prepare Assets

Default:

```bash
bello-prepare-assets --asset full_body
```

Manual output root:

```bash
bello-prepare-assets \
  --asset full_body \
  --output-root /tmp/my-bello-assets
```

Default output:

```text
/tmp/bello-mujoco-assets/<config-hash>/<asset>/
```

Generated files:

```text
<asset>/urdf/bello_<asset>.urdf
<asset>/mjcf/bello_<asset>.xml
<asset>/mjcf/bello_<asset>_viewer.xml
<asset>/meshes/
<asset>/visual_meshes/
<asset>/asset_manifest.json
```

The training MJCF is collision-only. The viewer MJCF includes visual geoms.

## Cache Key And Locking

The deterministic asset cache key includes:

- cache version
- asset name
- collision mesh face budget
- foot collision mesh face budget
- visual mesh face budget
- collision mode
- end-effector sphere mode
- neighbor exclusion depth
- actuator config

Asset generation uses a lock next to the cache directory so concurrent processes do not write the same generated asset at the same time.

## Mesh Simplification

Collision and visual mesh budgets are independent:

```bash
bello-prepare-assets \
  --mesh-face-budget 1000 \
  --foot-collision-mesh-face-budget 1000 \
  --visual-mesh-face-budget 200000
```

The simplifier may stop above the target if another simplification pass does not reduce face count.

## Collision Modes

```bash
bello-prepare-assets --collision-mode mesh
bello-prepare-assets --collision-mode foot-mesh
bello-prepare-assets --collision-mode box
```

Modes:

- `mesh`: simplified mesh collision for robot collision geoms.
- `foot-mesh`: foot collision geoms remain mesh; other robot mesh geoms become boxes from mesh bounds.
- `box`: robot mesh collision geoms become boxes from mesh bounds.

Foot mesh names:

```text
left_ankle_roll_link
right_ankle_roll_link
```

## Viewer MJCF

The viewer MJCF is generated alongside the training MJCF:

```text
<asset>/mjcf/bello_<asset>_viewer.xml
```

It includes non-colliding visual geoms from `visual_meshes`. The training env does not load these visual geoms by default.

## End-Effector Spheres

Arm end effectors can be replaced with 65 mm diameter spheres, attached 20 mm below the selected wrist yaw joint:

```bash
bello-prepare-assets --end-effector-sphere-mode both
```

Valid modes:

```text
none
both
right
left
```

Replacement removes the selected wrist yaw joint and all child links/joints below it, then attaches a fixed sphere link to the wrist parent.

This option is only valid for assets containing arm joints. It is rejected for `lower_body`.

## Neighbor Collision Excludes

Default:

```bash
bello-prepare-assets --neighbor-exclusion-depth 2
```

Depth 2 excludes:

- parent/child collision
- grandparent/grandchild collision
- sibling body collision

This removes most mechanically-neighboring self-collision while preserving non-neighbor body contact.

## Terrain Is Runtime-Only

Source URDFs and generated robot MJCFs do not include ground. Terrain is composed at model load time through `TerrainConfig`.
