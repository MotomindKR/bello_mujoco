# Robot description audit tools

These tools extract reviewable physical data from a CAD assembly before any
robot description is written. Generated inventories, converted CAD files,
meshes, and browser assets belong under `artifacts/` and are not committed.

Run the STEP inventory extractor with FreeCAD's Python executable and module
directory:

```bash
PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib \
  /Applications/FreeCAD.app/Contents/Resources/bin/python \
  tools/robot_description/extract_step_inventory.py \
  --source "/absolute/path/to/assembly.step" \
  --output artifacts/<robot>/inventory.json \
  --fcstd-output artifacts/<robot>/assembly.FCStd
```

The inventory is bound to the source bytes by SHA-256. It records the complete
FreeCAD object hierarchy, placements, shape geometry, mass-property geometry,
and source relationships without assigning materials, masses, links, or
joints. Those decisions belong to later reviewed manifests.

Build a source-bound browser review after creating the inventory and FCStd
cache:

```bash
uv run python tools/robot_description/build_review_manifest.py \
  --inventory artifacts/<robot>/inventory.json \
  --output artifacts/<robot>/review_manifest.json \
  --mesh-url-prefix /artifacts/<robot>/meshes

PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib \
  /Applications/FreeCAD.app/Contents/Resources/bin/python \
  tools/robot_description/export_review_geometry.py \
  --fcstd artifacts/<robot>/assembly.FCStd \
  --manifest artifacts/<robot>/review_manifest.json \
  --output-dir artifacts/<robot>/raw_review_geometry

uv run python tools/robot_description/pack_review_meshes.py \
  --manifest artifacts/<robot>/review_manifest.json \
  --raw-dir artifacts/<robot>/raw_review_geometry \
  --output-dir artifacts/<robot>/meshes

uv run python -m http.server 8765 --bind 127.0.0.1
```

Then open
`http://127.0.0.1:8765/tools/robot_description/review.html?manifest=/artifacts/<robot>/review_manifest.json`.
The page stores draft decisions in browser local storage and exports a JSON
review bound to the original STEP SHA-256. It never edits a robot description.

After resolving the Bello Mini material and catalog-mass review, build the
joint/link naming gate:

```bash
uv run python tools/robot_description/build_joint_review_manifest.py \
  --manifest artifacts/bello_mini_cad/review_manifest.json \
  --review /absolute/path/to/bello_mini_cad_review-resolved.json \
  --output artifacts/bello_mini_cad/joint_review_manifest.json
```

Open
`http://127.0.0.1:8765/tools/robot_description/joint_review.html?manifest=/artifacts/bello_mini_cad/joint_review_manifest.json`.
This second gate names all physical motors, movable and fixed joints, and links.
Its motor centers are visual evidence only: shaft origins, axis signs, limits,
encoder zeros, CAN IDs, and ankle-differential coefficients remain explicit
later review gates.

After exporting and fully approving the joint/link review, extract the physical
joint-axis lines from the cached CAD document:

```bash
PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib \
  /Applications/FreeCAD.app/Contents/Resources/bin/python \
  tools/robot_description/build_joint_frame_manifest.py \
  --fcstd artifacts/bello_mini_cad/bello_mini_import.FCStd \
  --manifest artifacts/bello_mini_cad/joint_review_manifest.json \
  --review /absolute/path/to/bello_mini_joint_link_review.json \
  --output artifacts/bello_mini_cad/joint_frame_review_manifest.json
```

Open
`http://127.0.0.1:8765/tools/robot_description/joint_review.html?manifest=/artifacts/bello_mini_cad/joint_frame_review_manifest.json`.
The frame gate independently approves each CAD-derived axis line and its
semantic positive direction. Direct-drive axes are extracted from dominant
clusters of concentric motor circles. The four ankle axes retain their lower
orthogonal-bearing evidence, while the RS02 motor shafts remain remote
transmission inputs and never become URDF joints.

Resolve the exported frame decisions before any link or URDF generation:

```bash
uv run python tools/robot_description/resolve_joint_frame_review.py \
  --manifest artifacts/bello_mini_cad/joint_frame_review_manifest.json \
  --review /absolute/path/to/bello_mini_joint_frame_review.json \
  --output artifacts/bello_mini_cad/resolved_joint_frames.json
```

The resolver rejects missing or unresolved rows and applies only explicit
`Flip + axis` decisions. The result records the frame-review checksum and is
the signed joint-frame input for link ownership and URDF generation.

Build the rigid-link ownership review from the approved materials and frames:

```bash
uv run python tools/robot_description/build_link_ownership_manifest.py \
  --manifest artifacts/bello_mini_cad/review_manifest.json \
  --classification-review /absolute/path/to/bello_mini_cad_review-resolved.json \
  --frames artifacts/bello_mini_cad/resolved_joint_frames.json \
  --output artifacts/bello_mini_cad/link_ownership_review_manifest.json
```

The ownership manifest covers every non-hand CAD occurrence exactly once. It
keeps whole catalog motors on their stator/parent links, puts both remote ankle
motors on the knee, and reserves each ankle-pitch link as a massless serial
transmission coordinate. Material-derived structural mass and measured torso
assemblies follow the link assignment that is confirmed in the 3D review.

Open
`http://127.0.0.1:8765/tools/robot_description/link_ownership_review.html?manifest=/artifacts/bello_mini_cad/link_ownership_review_manifest.json`.
The fixed 3D panel renders the selected rigid group in solid orange and all
other retained geometry translucently. It projects every approved joint axis
and child-link position over the CAD, with the selected owner link and semantic
positive axis emphasized. The pelvis root marker is the CAD assembly origin;
full link-frame rotations remain a later explicit convention. Export is
disabled until every group has an explicitly confirmed physical owner link.
The reviewer also pairs components with identical bilateral geometry evidence,
renders each pair together, and blocks export while a matched pair is assigned
to non-mirrored links. Groups without a high-confidence exact pair remain
available through the symmetry filter for manual inspection.

Resolve the approved ownership review and apply the explicit Bello Mini CAD
symmetry completion:

```bash
uv run python tools/robot_description/resolve_link_ownership_review.py \
  --manifest artifacts/bello_mini_cad/link_ownership_review_manifest.json \
  --review /absolute/path/to/bello_mini_link_ownership_review.json \
  --output artifacts/bello_mini_cad/resolved_link_ownership.json
```

The resolver rejects any remaining matched bilateral ownership disagreement.
It reflects `ownership-010` across CAD `x=0`, then copies `ownership-021`,
`022`, `023`, `024`, `028`, and `029` to the right leg using the reviewed
ankle-center translation while preserving each source transform's orientation
and handedness. These operations are recorded as synthetic, source-linked
groups rather than silently changing the STEP inventory.

Inspect the resolved result at
`http://127.0.0.1:8765/tools/robot_description/link_ownership_review.html?manifest=/artifacts/bello_mini_cad/resolved_link_ownership.json`.
This mode is read-only. Its symmetry-completion table compares each reviewed
source in orange with its generated counterpart in purple, while retaining the
joint-axis and link-position overlays.

Build mass properties, exterior-only visual selection, primitive collisions,
and decimated link-local meshes after the ownership gate is approved:

```bash
PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib \
  /Applications/FreeCAD.app/Contents/Resources/bin/python \
  tools/robot_description/build_inertial_manifest.py \
  --ownership artifacts/bello_mini_cad/resolved_link_ownership.json \
  --inventory artifacts/bello_mini_cad/inventory.json \
  --output artifacts/bello_mini_cad/inertial_manifest.json

uv run python tools/robot_description/build_geometry_manifest.py \
  --ownership artifacts/bello_mini_cad/resolved_link_ownership.json \
  --inertials artifacts/bello_mini_cad/inertial_manifest.json \
  --classification /absolute/path/to/bello_mini_cad_review-resolved.json \
  --output artifacts/bello_mini_cad/geometry_manifest.json

uv run python tools/robot_description/export_link_meshes.py \
  --geometry artifacts/bello_mini_cad/geometry_manifest.json \
  --inertials artifacts/bello_mini_cad/inertial_manifest.json \
  --output-dir artifacts/bello_mini_cad/generated/meshes \
  --manifest-output artifacts/bello_mini_cad/visual_mesh_manifest.json \
  --face-limit 16000
```

The visual policy omits measured battery/electronics internals, small hardware,
and sub-gram details while retaining their physical mass properties. Each
catalog motor contributes one exterior actuator envelope. Collision geometry
contains only URDF/MuJoCo boxes, cylinders, and spheres; no visual mesh is used
for collision.
The pelvis camera is classified as an Orbbec Astra Stereo S U3 catalog
component. Its 30 g nominal catalog mass is distributed over its exact
in-assembly CAD geometry and lumped into `pelvis_link`; it is not treated as a
homogeneous plastic enclosure or concentrated at the pelvis origin. The camera
CAD visual and its exact-bounds box collision are attached to the massless
`astra_mount_frame`, while the mass properties remain in `pelvis_link` at their
CAD pose. The fixed sensor tree follows the modern Orbbec convention:
`camera_link`, `camera_depth_frame`, and `camera_left_ir_frame` share the
camera-left IR/depth origin; the right IR origin is 40 mm toward camera-right;
and RGB is 10 mm toward camera-right and 2 mm farther forward. Every optical
frame uses `rpy="-1.5707963267948966 0 -1.5707963267948966"`.

The imported camera solid contains the exact 40 mm IR aperture-axis spacing and
10 mm RGB lateral offset, but it does not expose an unambiguous pair of 2.30 mm,
54.99 mm-spaced mounting holes or a calibrated pinhole datum. Consequently,
`astra_mount_frame` uses the Compound008 source datum at the exact IR-axis and
enclosure symmetry midpoint. The depth origin is hard-coded at 2 mm behind the
CAD frontmost plane (source-local Z = 4.30 mm), and RGB is hard-coded 2 mm
forward of depth. These are nominal mechanical approximations, not runtime
parameters or factory calibration. If the camera driver publishes factory
sensor extrinsics, disable the URDF sensor-frame broadcaster for that subtree;
the two sources must not publish duplicate TFs.
The massless `imu_link` is fixed 50 mm below the midpoint of the reviewed
left/right hip-pitch origins. Its exact pelvis translation is
`[-0.0099999999999894115, 0, -0.230000001143162]` m and its yaw is
`-1.5707963267948966` rad, giving IMU +X robot-right, +Y robot-forward, and +Z
up as in original Bello. No recognizable IMU component or mounting datum
exists in the reviewed STEP, so the 50 mm offset is a nominal placement; the
hip midpoint itself is derived and symmetry-validated from the reviewed CAD
joint origins.
The generated MuJoCo compiler extension disables geometry-derived inertia, so
the two fixed anonymous sphere links remain massless when the URDF is loaded
directly and cannot silently increase the robot mass.

Joint ranges come from the designer's 2026-09-03 values. Source values are in
degrees and the generated URDF stores radians. The left-side signs are
**provisionally** mapped to the current URDF positive axes pending user review;
right-side pitch ranges are copied and right-side roll/yaw ranges are reflected
across Mini `y=0`. Symmetric ranges are unchanged by that reflection.

| Station | Left / body range (deg) | Derived right range (deg) |
| --- | ---: | ---: |
| waist roll | -25 .. +25 | body joint |
| waist yaw | -90 .. +90 | body joint |
| hip pitch | -90 .. +180 | -90 .. +180 |
| hip roll | -121 .. +31 | -31 .. +121 |
| hip yaw | -90 .. +90 | -90 .. +90 |
| knee | 0 .. +145 | 0 .. +145 |
| ankle roll | -35 .. +35 | -35 .. +35 |
| ankle pitch | -45 .. +45 | -45 .. +45 |
| shoulder pitch | -225 .. +45 | -225 .. +45 |
| shoulder roll | -180 .. +15 | -15 .. +180 |
| shoulder yaw | -90 .. +90 | -90 .. +90 |
| elbow pitch | -15 .. +130 | -15 .. +130 |
| wrist yaw | -116.8 .. +116.8 | -116.8 .. +116.8 |
| wrist pitch | -90 .. +90 | -90 .. +90 |
| wrist roll | -90 .. +90 | -90 .. +90 |

These values still require user sign review and physical-stop validation before
motion. The axial arm coordinate between `elbow_link` and `lower_arm_link` is
named `wrist_yaw_joint`; its physical actuator is `wrist_yaw_motor`. This is a
naming correction only: chain placement is unchanged and the left/right CAN IDs
remain 35/45. Motor `command_direction` remains null because protocol polarity
is a separate hardware-calibration question.

Generate the source-bound URDF, calibration-gated motor topology, build report,
and visual confirmation assets:

```bash
uv run python tools/robot_description/build_robot_description.py \
  --ownership artifacts/bello_mini_cad/resolved_link_ownership.json \
  --inertials artifacts/bello_mini_cad/inertial_manifest.json \
  --geometry artifacts/bello_mini_cad/geometry_manifest.json \
  --meshes artifacts/bello_mini_cad/visual_mesh_manifest.json \
  --urdf-output artifacts/bello_mini_cad/generated/urdf/bello_mini.urdf \
  --motor-output artifacts/bello_mini_cad/generated/motor_topology.yaml \
  --report-output artifacts/bello_mini_cad/generated/build_report.json

uv run python tools/robot_description/render_geometry_review.py \
  --urdf artifacts/bello_mini_cad/generated/urdf/bello_mini.urdf \
  --mjcf-output artifacts/bello_mini_cad/generated/bello_mini_geometry_review.xml \
  --image-dir artifacts/bello_mini_cad/generated/review_images

uv run python tools/robot_description/audit_foot_collision.py \
  --geometry artifacts/bello_mini_cad/geometry_manifest.json \
  --meshes artifacts/bello_mini_cad/visual_mesh_manifest.json \
  --image-output artifacts/bello_mini_cad/generated/review_images/foot_bottom_collision_overlay.png \
  --report-output artifacts/bello_mini_cad/generated/foot_collision_audit.json
```

Open
`http://127.0.0.1:8765/tools/robot_description/geometry_review.html?base=/artifacts/bello_mini_cad`
for the final visual/collision gate. The fixed viewer can isolate a link, make
all other links translucent, toggle collision primitives, and display the
current positive joint axes. Sliders cover all 28 revolute joints using the
exact generated URDF tree and designer-provided, sign-review-pending limits;
the reset and left/right mirror controls support pose and symmetry review.
Mirroring is atomic: if any reflected target would exceed its derived target
limit, the viewer leaves the entire pose unchanged and lists every offending
target. This intentionally exposes asymmetric bilateral limits for review.
Link meshes, collision primitives, anonymous end effectors, joint axes, sensor
frames, motor markers, and camera FoV volumes all follow the same browser-side
forward kinematics. A separate link-frame overlay draws compact RGB axes for
all physical links and expands them with labels when one link is selected.
The motor audit overlays every CAD motor-frame `+Z` shaft axis at the
corresponding actuator location and lists the assigned CAN bus/ID beside the
CAD-only joint-axis alignment estimate. These estimates are geometric review
evidence, not calibrated RobStride wire polarity.

The CAN layout retains Bello's body ranges while increasing IDs outward along
each kinematic chain: left/right arms are `31..37` / `41..47`, waist yaw/roll
are `51,52`, and left/right legs are `61..66` / `71..76`. Legs use `can0`;
waist and arms use `can1`; both are nominally 1 Mbps. The generated topology
keeps every live `command_direction` and encoder zero null because RobStride's
published documentation does not tie protocol-positive rotation to STEP-frame
`+Z`. Direct-drive rows include the conditional placeholder
`sign(dot(CAD motor +Z, reviewed joint +axis))`; ankle rows expose only their
pitch-axis projection and retain null differential matrices. Validate raw
feedback and low-current motion on a restrained robot before creating a live
hardware description.

The viewer applies the build report's nominal root translation and draws the
ground plane at `z=0`, so both sole contact surfaces can be checked in their
final zero-joint deployment pose. The URDF keeps `pelvis_link` as its physical
floating root; the robot definition must use the report's exact
`grounding.nominal_root_translation_m` as its default root position rather than
adding a synthetic ground-root link. The dedicated
bottom-view audit checks each five-piece sole fit against the corresponding STL
slice, including footprint overlap and the exact lowest contact plane.

The camera overlay renders the published Astra Stereo S U3 depth FoV
(67.9 degrees horizontal by 45.3 degrees vertical, 0.25-2.5 m) and RGB FoV
(71.5 degrees horizontal by 56.7 degrees vertical). The frusta originate at the
corresponding optical frames and follow REP-103 optical axes. The depth volume
starts at its 0.25 m minimum range; RGB is shown to 2.5 m only to make its
angular coverage directly comparable to depth.
