const frameName = "pico_native_y_up";
const bodyNames = [
  "Pelvis",
  "Left_Hip",
  "Right_Hip",
  "Spine1",
  "Left_Knee",
  "Right_Knee",
  "Spine2",
  "Left_Ankle",
  "Right_Ankle",
  "Spine3",
  "Left_Foot",
  "Right_Foot",
  "Neck",
  "Left_Collar",
  "Right_Collar",
  "Head",
  "Left_Shoulder",
  "Right_Shoulder",
  "Left_Elbow",
  "Right_Elbow",
  "Left_Wrist",
  "Right_Wrist",
  "Left_Hand",
  "Right_Hand",
];
const bodyParents = [
  -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20,
  21,
];
const handNames = [
  "palm",
  "wrist",
  "thumb_metacarpal",
  "thumb_proximal",
  "thumb_distal",
  "thumb_tip",
  ...["index", "middle", "ring", "little"].flatMap((finger) =>
    ["metacarpal", "proximal", "intermediate", "distal", "tip"].map(
      (joint) => `${finger}_${joint}`,
    ),
  ),
];
const handParents = [
  1, -1, 1, 2, 3, 4, 1, 6, 7, 8, 9, 1, 11, 12, 13, 14, 1, 16, 17, 18, 19, 1, 21,
  22, 23, 24,
];

function component(value, hand = false) {
  if (!value || !(hand ? value.active : value.valid)) return null;
  const names = hand ? handNames : bodyNames;
  const parents = hand ? handParents : bodyParents;
  const n = names.length;
  if (
    value.frame !== frameName ||
    value.jointNames?.length !== n ||
    new Set(value.jointNames).size !== n ||
    !names.every((name) => value.jointNames.includes(name)) ||
    value.positions?.length !== n * 3 ||
    value.orientations?.length !== n * 4 ||
    !Number.isFinite(value.ageSeconds) ||
    value.ageSeconds < 0 ||
    (hand &&
      (value.positionValid?.length !== n ||
        value.orientationValid?.length !== n))
  )
    throw new Error("Invalid XRoboToolkit joint layout or frame.");
  const joints = names.map((name) => {
    const index = value.jointNames.indexOf(name);
    const position = value.positions.slice(index * 3, index * 3 + 3);
    const wxyz = value.orientations.slice(index * 4, index * 4 + 4);
    const valid = !hand || value.positionValid[index];
    const orientationValid = !hand || value.orientationValid[index];
    if (
      (valid && !position.every(Number.isFinite)) ||
      (orientationValid &&
        (!wxyz.every(Number.isFinite) ||
          Math.abs(Math.hypot(...wxyz) - 1) > 0.01))
    )
      throw new Error("Invalid XRoboToolkit joint position or orientation.");
    return {
      name,
      position: valid ? position : null,
      orientation: orientationValid
        ? [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        : null,
    };
  });
  return { joints, parents, ageMs: value.ageSeconds * 1000 };
}
function trackingFrame(value) {
  const waiting =
    value.packetAgeSeconds === -1 && String(value.sequence) === "0";
  if (
    value.upstreamConnected &&
    !waiting &&
    (!Number.isFinite(value.packetAgeSeconds) || value.packetAgeSeconds < 0)
  )
    throw new Error("Invalid XRoboToolkit packet age.");
  return {
    sequence: value.sequence,
    deviceId: value.deviceId,
    frame: frameName,
    upstreamConnected: value.upstreamConnected,
    packetAgeMs:
      value.upstreamConnected && !waiting
        ? value.packetAgeSeconds * 1000
        : null,
    body: waiting ? null : component(value.xrobotBody),
    leftHand: waiting ? null : component(value.xrobotLeftHand, true),
    rightHand: waiting ? null : component(value.xrobotRightHand, true),
  };
}
module.exports = { trackingFrame, bodyNames, handNames };
