# Bello URDF inertial overhaul report

Date: 2026-08-17

## Final result

The canonical physical model is **55.068358 kg** with 25 movable joints. It uses reviewed CAD material assignments, official RobStride masses, exact CAD placement/inertia geometry, bilateral averaging, a 50% effective solid-mass fraction for the TPU foot parts, and a 5 kg torso-electronics lump.

The maintained runtime uses these canonical inertials directly. The former uniform 60 kg rescaling is disabled.

## Evidence and scope

- FreeCAD source SHA-256: `ea35b91fda1fa03ade2be6fa82a2b6b0b0900b0aa16366a95e53301a1638905c`
- Canonical URDF SHA-256 after write: `16a25817373b1cc93b2472c7a486b12adea3a0c5444978f16707a94df9003d4f`
- Material review SHA-256: `49bef29ba05ef6ec5c72ffd9f2a1e0a648c77a96ca216c6fbacf564b1c4317b4`
- Link review SHA-256: `112c199b34870c2c3d76fa9c0ac08a2b1dbf70372e1cd8ba4e7ef95273b363a9`
- Reviewed structural occurrences: 203
- Material-excluded occurrences: 27
- Ownership assignments changed from geometric suggestion: 6
- CAD motors retained/excluded: 27/4
- Hands and grippers are absent. Each wrist terminates in a fixed, massless 32.5 mm collision sphere.
- The approved lower parallel linkage policy is `serial_lumped`.

## Canonical topology

The active chain contains six joints per leg, six joints per arm, and waist yaw: 25 movable joints total. Waist roll, waist pitch, neck yaw, and head pitch are fixed because their physical motor pairs are absent. The final arm coordinate is wrist pitch; no wrist-roll or wrist-yaw coordinate exists.

Six fixed links intentionally have no inertial element: the three camera frames, two endpoint spheres, and the waist-roll spacer. Camera proxy mass is already included in `head_link`.

## Mass closure

| Category | Mass | Share |
| --- | ---: | ---: |
| Density-derived structure | 25.680358 kg | 46.63% |
| RobStride actuators | 24.388000 kg | 44.29% |
| Torso electronics lump | 5.000000 kg | 9.08% |
| **Total** | **55.068358 kg** | **100%** |

| Region | Structure | Motors | Electronics | Total | COM in URDF world |
| --- | ---: | ---: | ---: | ---: | --- |
| Upper body | 11.124130 kg | 8.364000 kg | 5.000000 kg | **24.488130 kg** | [0.368763, -5.78625, 344.703] mm |
| Lower body | 14.556227 kg | 16.024000 kg | 0.000000 kg | **30.580227 kg** | [0.0390866, -6.11795, -309.211] mm |
| **Whole robot** | **25.680358 kg** | **24.388000 kg** | **5.000000 kg** | **55.068358 kg** | **[0.185689, -5.97045, -18.4247] mm** |

### Subassemblies

| Assembly | Structure | Motors | Electronics | Total |
| --- | ---: | ---: | ---: | ---: |
| Upper core | 5.513935 kg | 2.840000 kg | 5.000000 kg | **13.353935 kg** |
| Left arm | 2.805098 kg | 2.762000 kg | 0.000000 kg | **5.567098 kg** |
| Right arm | 2.805098 kg | 2.762000 kg | 0.000000 kg | **5.567098 kg** |
| Pelvis | 1.637671 kg | 4.260000 kg | 0.000000 kg | **5.897671 kg** |
| Left leg | 6.459278 kg | 5.882000 kg | 0.000000 kg | **12.341278 kg** |
| Right leg | 6.459278 kg | 5.882000 kg | 0.000000 kg | **12.341278 kg** |

## Material breakdown

| Material | Reviewed density | Effective mass | CAD source groups | Physical occurrences |
| --- | ---: | ---: | ---: | ---: |
| `aluminum_6061_t6` | 2700 kg/m^3 | 19.030891 kg | 55 | 119 |
| `abs` | 1040 kg/m^3 | 3.943955 kg | 27 | 42 |
| `steel_4140` | 7850 kg/m^3 | 1.377437 kg | 7 | 32 |
| `tpu` | 1200 kg/m^3 | 1.137941 kg | 1 | 2 |
| `stainless_304` | 8000 kg/m^3 | 0.190134 kg | 1 | 8 |

`Solid061` (`Part2.38`) is TPU with a user-approved 50% effective mass and inertia fraction relative to fully solid CAD. The pair contributes 1.137941 kg. This is a fabrication calibration, not a claim that slicer infill alone equals 50% effective density.

## Actuator breakdown

Vendor evidence: [RobStride Dynamics Co., Ltd.](https://github.com/RobStride/Product_Information), source revision `6ad12f50006273b7ea4eea88980f927d97c22f0d`, specification date 2026-07-13.

| Model | Count | Unit mass | Subtotal | Official STEP SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `ROBSTRIDE_00` | 4 | 0.310 kg | 1.240000 kg | `4133dea27ccff90d69588dbc37662c9eafeed0bf858fc0a4cee93417e508abfe` |
| `ROBSTRIDE_03` | 6 | 0.900 kg | 5.400000 kg | `8eccaf6291e395d8622300396289df898ab18a1648193a5cd327cfbd2bf72500` |
| `ROBSTRIDE_04` | 9 | 1.420 kg | 12.780000 kg | `841f5deaa15575e98bdca3a6d0b69f4207c305f56f1743f0a7122a42366996b1` |
| `ROBSTRIDE_06` | 8 | 0.621 kg | 4.968000 kg | `ee6af519b6ef8d4a22e7e1f3d2196e1d60af0780e2c7cb78ba6118dd9b185d0e` |

The two secondary knee RS04 motors are retained in the physical mass model and lumped under the serial/lumped linkage policy. They do not add URDF degrees of freedom or separate hardware coordinates; the maintained 25-coordinate hardware map still exposes one knee coordinate per side. Establishing a separate CAN/transmission mapping for the secondary pair would require measured coupling and wiring information and is outside this inertial overhaul. The waist roll/pitch RS04 pair and neck/head RS05 pair are excluded as absent. Motor intrinsic inertias use exact in-assembly CAD actuator geometry scaled to official vendor mass; these are not vendor-measured tensors.

## Torso electronics calibration

A 5 kg electronics contribution is attached to `torso_link` at local `[0, -0.025, 0.19749988]` m, the center of the torso visual-mesh axis-aligned bounding box. It is a provisional point mass: its parallel-axis effect is included, but its intrinsic package inertia is zero until package dimensions or layout are measured.

The observed stripped robot was approximately 54 kg after removing modeled leg ABS parts, head, neck, and `Solid198`. Those modeled removals total 1.640163 kg. The final model predicts 53.428194 kg in that condition, leaving about 0.571806 kg for distributed wiring, connectors, fasteners, measurement uncertainty, and remaining modeling error.

## Bilateral symmetry

All reviewed paired links have equal mass after assignment, and left/right source inventories match. The generator averages paired world-frame COM and inertia under the best reviewed bilateral transform.

| Left link | Right link | Transform | Raw mass difference | Raw COM residual | Raw inertia residual |
| --- | --- | --- | ---: | ---: | ---: |
| `l_elbow_link` | `r_elbow_link` | `rotate_xy` | 0 g | 3.58496e-05 mm | 1.4961e-05% |
| `l_lower_arm_link` | `r_lower_arm_link` | `rotate_xy` | 0 g | 7.28415e-05 mm | 1.41665e-05% |
| `l_shoulder_pitch_link` | `r_shoulder_pitch_link` | `rotate_xy` | 0 g | 6.35561e-05 mm | 4.32812e-06% |
| `l_shoulder_roll_link` | `r_shoulder_roll_link` | `rotate_xy` | 0 g | 4.45701e-05 mm | 9.47894e-06% |
| `l_upper_arm_link` | `r_upper_arm_link` | `rotate_xy` | 0 g | 7.80652e-06 mm | 2.16866e-05% |
| `l_wrist_link` | `r_wrist_link` | `rotate_xy` | 0 g | 8.51424e-05 mm | 9.81141e-06% |
| `left_ankle_pitch_link` | `right_ankle_pitch_link` | `reflect_x` | 0 g | 0.277013 mm | 0.0309916% |
| `left_ankle_roll_link` | `right_ankle_roll_link` | `reflect_x` | 0 g | 0.272896 mm | 0.0103165% |
| `left_hip_pitch_link` | `right_hip_pitch_link` | `rotate_xy` | 4.44089e-13 g | 0.500012 mm | 0.00048649% |
| `left_hip_roll_link` | `right_hip_roll_link` | `rotate_xy` | 0 g | 0.550863 mm | 0.326547% |
| `left_hip_yaw_link` | `right_hip_yaw_link` | `reflect_x` | 0 g | 0.495159 mm | 0.466868% |
| `left_knee_link` | `right_knee_link` | `reflect_x` | 0 g | 2.17916 mm | 5.65502% |

The largest raw discrepancy is the knee inertia, caused by lower-linkage CAD placement/orientation differences despite equal part inventories. The canonical tensors are intentionally bilateral averages.

## Canonical per-link inertials

COM is link-local. Inertia order is `ixx, iyy, izz, ixy, ixz, iyz` in kg m^2.

| Link | Mass | Structure | Motors | Electronics | COM m | Inertia kg m^2 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `head_link` | 0.603674856 | 0.603674856 | 0.000000000 | 0.000000000 | [-0.0636714, -0.00132237, 0.0419138] | [0.00269601, 0.00245731, 0.00319094, 3.1297e-05, -3.1968e-06, 4.95777e-05] |
| `l_elbow_link` | 0.715823771 | 0.405823771 | 0.310000000 | 0.000000000 | [-0.0158127, -0.0259655, -0.0539569] | [0.00231665, 0.00208124, 0.000946468, 2.07514e-06, -0.000373663, 9.34724e-06] |
| `l_lower_arm_link` | 0.536588500 | 0.226588500 | 0.310000000 | 0.000000000 | [0.000227289, -0.00388846, -0.0626326] | [0.000948927, 0.000813868, 0.000371007, 1.7421e-07, 2.93824e-06, -3.28237e-05] |
| `l_shoulder_pitch_link` | 1.503764860 | 0.603764860 | 0.900000000 | 0.000000000 | [-0.0785965, 0.00240127, 1.20702e-05] | [0.00200975, 0.00327912, 0.00278303, -3.78446e-05, -3.95663e-07, -5.90535e-06] |
| `l_shoulder_roll_link` | 1.374529267 | 0.753529267 | 0.621000000 | 0.000000000 | [-0.0121774, -0.0266984, -0.0650075] | [0.00510113, 0.00445788, 0.00284181, -1.65803e-05, -0.000494032, -6.05094e-05] |
| `l_upper_arm_link` | 1.313829660 | 0.692829660 | 0.621000000 | 0.000000000 | [0.0086503, 0.00528, -0.119379] | [0.00580679, 0.00591548, 0.00150619, 4.60131e-07, 0.000510659, 8.21562e-05] |
| `l_wrist_link` | 0.122561766 | 0.122561766 | 0.000000000 | 0.000000000 | [-9.17713e-07, 0.0247501, -0.0187826] | [0.00015856, 7.39281e-05, 0.00012366, 2.42249e-10, -1.52904e-09, 5.52234e-06] |
| `left_ankle_pitch_link` | 0.046365845 | 0.046365845 | 0.000000000 | 0.000000000 | [0.0128998, 2.83574e-08, -0.00817509] | [9.60355e-06, 8.6248e-06, 3.02578e-06, -2.82533e-07, 1.45201e-09, -6.23675e-11] |
| `left_ankle_roll_link` | 1.372305013 | 1.372305013 | 0.000000000 | 0.000000000 | [-7.2274e-05, -0.027664, -0.00509084] | [0.00699076, 0.00758371, 0.00202629, 1.55407e-07, -1.75483e-07, 0.000507187] |
| `left_hip_pitch_link` | 1.590197597 | 0.690197597 | 0.900000000 | 0.000000000 | [3.94635e-05, 0.00298891, 0.0581141] | [0.00310832, 0.00295901, 0.00245906, 1.41429e-05, 0.000267037, -5.91777e-05] |
| `left_hip_roll_link` | 1.488019341 | 0.588019341 | 0.900000000 | 0.000000000 | [0.000231238, -0.0521909, 0.0302062] | [0.00430649, 0.00201345, 0.00402388, 4.70117e-07, -5.31168e-07, -0.000260871] |
| `left_hip_yaw_link` | 4.937231750 | 2.097231750 | 2.840000000 | 0.000000000 | [-0.00017432, 0.0025829, 0.132024] | [0.0249308, 0.0255263, 0.0119327, -1.62156e-07, -2.86925e-06, -0.000734588] |
| `left_knee_link` | 2.907158477 | 1.665158477 | 1.242000000 | 0.000000000 | [-0.139841, -0.0139537, 0.00956892] | [0.00513594, 0.0219815, 0.0244169, -0.00541987, 9.93098e-05, 2.45345e-05] |
| `neck_link` | 0.066109188 | 0.066109188 | 0.000000000 | 0.000000000 | [0.0069897, -3.17984e-08, 0.0301398] | [3.9386e-05, 3.86881e-05, 1.89628e-05, 2.8489e-14, -9.06178e-06, -4.36925e-14] |
| `pelvis_link` | 5.897671077 | 1.637671077 | 4.260000000 | 0.000000000 | [0.000202669, 0.000321424, 0.0107295] | [0.0252237, 0.0312558, 0.0230781, 7.24387e-07, -6.69491e-05, 0.000113307] |
| `r_elbow_link` | 0.715823771 | 0.405823771 | 0.310000000 | 0.000000000 | [0.0158127, 0.0259655, -0.0539569] | [0.00231665, 0.00208124, 0.000946468, 2.07514e-06, 0.000373663, -9.34724e-06] |
| `r_lower_arm_link` | 0.536588500 | 0.226588500 | 0.310000000 | 0.000000000 | [-0.000227289, 0.00388846, -0.0626326] | [0.000948927, 0.000813868, 0.000371007, 1.7421e-07, -2.93824e-06, 3.28237e-05] |
| `r_shoulder_pitch_link` | 1.503764860 | 0.603764860 | 0.900000000 | 0.000000000 | [0.0785965, -0.00240127, 1.20702e-05] | [0.00200975, 0.00327912, 0.00278303, -3.78446e-05, 3.95663e-07, 5.90535e-06] |
| `r_shoulder_roll_link` | 1.374529267 | 0.753529267 | 0.621000000 | 0.000000000 | [0.0121774, 0.0266984, -0.0650075] | [0.00510113, 0.00445788, 0.00284181, -1.65803e-05, 0.000494032, 6.05094e-05] |
| `r_upper_arm_link` | 1.313829660 | 0.692829660 | 0.621000000 | 0.000000000 | [-0.0086503, -0.00528, -0.119379] | [0.00580679, 0.00591548, 0.00150619, 4.60131e-07, -0.000510659, -8.21562e-05] |
| `r_wrist_link` | 0.122561766 | 0.122561766 | 0.000000000 | 0.000000000 | [9.17713e-07, -0.0247501, -0.0187826] | [0.00015856, 7.39281e-05, 0.00012366, 2.42249e-10, 1.52904e-09, -5.52234e-06] |
| `right_ankle_pitch_link` | 0.046365845 | 0.046365845 | 0.000000000 | 0.000000000 | [0.0129049, 2.83574e-08, -0.0311229] | [9.60355e-06, 8.6248e-06, 3.02578e-06, -2.82533e-07, 1.45201e-09, -6.23676e-11] |
| `right_ankle_roll_link` | 1.372305013 | 1.372305013 | 0.000000000 | 0.000000000 | [-0.000378232, 0.0276639, -0.00509087] | [0.00699076, 0.00758371, 0.00202629, -9.7454e-08, 3.91771e-07, -0.000507187] |
| `right_hip_pitch_link` | 1.590197597 | 0.690197597 | 0.900000000 | 0.000000000 | [-3.94635e-05, -0.00298891, 0.0581141] | [0.00310832, 0.00295901, 0.00245906, 1.41429e-05, -0.000267037, 5.91777e-05] |
| `right_hip_roll_link` | 1.488019341 | 0.588019341 | 0.900000000 | 0.000000000 | [-0.000231238, 0.0521909, 0.0302062] | [0.00430649, 0.00201345, 0.00402388, 4.70117e-07, 5.31168e-07, 0.000260871] |
| `right_hip_yaw_link` | 4.937231750 | 2.097231750 | 2.840000000 | 0.000000000 | [-0.000264387, -0.0025829, 0.132024] | [0.0249308, 0.0255263, 0.0119327, -1.62156e-07, 2.86925e-06, 0.000734588] |
| `right_knee_link` | 2.907158477 | 1.665158477 | 1.242000000 | 0.000000000 | [-0.139841, -0.0139537, 0.00964864] | [0.00513586, 0.0219815, 0.0244169, -0.00541988, -9.07975e-05, -2.21417e-05] |
| `torso_link` | 12.123684262 | 4.283684262 | 2.840000000 | 5.000000000 | [0.000740166, -0.0124644, 0.221762] | [0.108015, 0.122393, 0.067263, 0.000148819, -0.00134792, -0.00479827] |
| `waist_yaw_link` | 0.560466460 | 0.560466460 | 0.000000000 | 0.000000000 | [-1.20986e-11, 0.00628053, 0.0506612] | [0.000973196, 0.000945393, 0.00187985, -9.065e-14, 4.93131e-14, 1.0012e-05] |

## Validation and limitations

- Every written inertia tensor is symmetric, positive-definite, and satisfies the principal-moment triangle inequality.
- Post-averaging left/right paired-link mass difference is zero.
- The canonical source mass is 55.068357541 kg; maintained configs do not rescale it.
- Native MuJoCo compilation preserves 25 actuators and the canonical mass to XML serialization precision. Collision geometry on inertial-free fixed frames is explicitly assigned zero mass so MuJoCo does not infer mass for the waist spacer or endpoint spheres.
- Runtime tests verify zero-pose world-frame bilateral symmetry, fixed absent joints, sphere geometry, actuator inventory, hardware mappings, and optional mass-scaling behavior.
- Final verification: 443 root tests passed with 2 skipped; 38 upper-body teleoperation tests passed; affected Python sources passed Ruff checks and formatting checks.
- No electronics other than the explicit 5 kg torso lump are included.
- Camera CAD envelopes are uniform ABS proxies rather than measured sensor assemblies.
- The electronics intrinsic inertia remains provisional.
- Material density remains sensitive to actual stock, fabrication, and infill. TPU has the explicit 50% calibration; other printed parts use reviewed bulk densities.
- The CAD file contained no native material designations; all material assignments were explicitly reviewed.

The final source of truth is `deps/bello-stack/descriptions/bello/urdf/bello.urdf`. Audit artifacts and one-off extraction tools are not runtime dependencies.

The temporary CAD extraction, browser-review, and URDF-writing tools were deleted after this report and the source URDF were finalized. The hardware Rust checks require the submodule's Linux-only Nix development shell and were not executable on the aarch64 macOS audit host; the Python teleoperation tests and linters were run locally.
