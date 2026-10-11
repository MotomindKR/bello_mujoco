import * as THREE from "three";

const axes = [
  { label: "X", direction: new THREE.Vector3(1, 0, 0), color: "#ee4b55" },
  { label: "Y", direction: new THREE.Vector3(0, 1, 0), color: "#42b960" },
  { label: "Z", direction: new THREE.Vector3(0, 0, 1), color: "#498bfa" },
];

export function drawAxisIndicator(canvas, camera) {
  const context = canvas.getContext("2d");
  if (!context) return;
  const size = 72;
  context.setTransform(canvas.width / size, 0, 0, canvas.height / size, 0, 0);
  context.clearRect(0, 0, size, size);
  const inverse = camera.quaternion.clone().invert();
  const projected = axes
    .map((axis) => ({
      ...axis,
      direction: axis.direction.clone().applyQuaternion(inverse),
    }))
    .sort((a, b) => a.direction.z - b.direction.z);
  for (const axis of projected) {
    const x = 36 + axis.direction.x * 20;
    const y = 36 - axis.direction.y * 20;
    context.beginPath();
    context.moveTo(36, 36);
    context.lineTo(x, y);
    context.lineWidth = 2;
    context.lineCap = "round";
    context.strokeStyle = axis.color;
    context.stroke();
    context.font = "bold 12px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = axis.color;
    context.fillText(
      axis.label,
      x + (axis.direction.x >= 0 ? 7 : -7),
      y + (axis.direction.y >= 0 ? -7 : 7),
    );
  }
}
