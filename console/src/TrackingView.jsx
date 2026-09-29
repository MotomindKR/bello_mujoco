import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Focus } from "lucide-react";
import { number } from "./format.mjs";
import { trackingStatus } from "./tracking-status.mjs";

const parts = [
  ["body", "Body"],
  ["leftHand", "Left hand"],
  ["rightHand", "Right hand"],
];

export function TrackingView({ state, dark }) {
  const host = useRef(null);
  const view = useRef(null);
  const latest = useRef(state.tracking);
  latest.current = state.tracking;
  const [error, setError] = useState("");
  useEffect(() => {
    const element = host.current;
    const tokens = getComputedStyle(document.documentElement);
    const color = (key) => new THREE.Color(tokens.getPropertyValue(key).trim());
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setError("3D rendering is unavailable on this device.");
      return;
    }
    setError("");
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = color("--model-background");
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(2, 1.8, 3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.enableDamping = false;
    const grid = new THREE.GridHelper(
      10,
      20,
      color("--border"),
      color("--border"),
    );
    scene.add(grid);
    const groups = Object.fromEntries(
      parts.map(([key]) => {
        const material = new THREE.LineBasicMaterial({
          color: color("--foreground"),
        });
        const line = new THREE.LineSegments(
          new THREE.BufferGeometry(),
          material,
        );
        const dots = new THREE.Points(
          new THREE.BufferGeometry(),
          new THREE.PointsMaterial({
            color: color("--foreground"),
            size: key === "body" ? 0.024 : 0.008,
          }),
        );
        scene.add(line, dots);
        return [key, { line, dots }];
      }),
    );
    let previous;
    let centered = false;
    const center = () => {
      const positions = parts.flatMap(
        ([key]) =>
          latest.current?.[key]?.joints.flatMap((joint) =>
            joint.position ? [new THREE.Vector3(...joint.position)] : [],
          ) || [],
      );
      if (!positions.length) return;
      const bounds = new THREE.Box3().setFromPoints(positions);
      bounds.getCenter(controls.target);
      const size = Math.max(bounds.getSize(new THREE.Vector3()).length(), 0.5);
      camera.position
        .copy(controls.target)
        .add(new THREE.Vector3(0.7, 0.35, 1).multiplyScalar(size));
      controls.update();
      centered = true;
    };
    view.current = { center };
    const resize = new ResizeObserver(() => {
      camera.aspect = element.clientWidth / Math.max(1, element.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(element.clientWidth, element.clientHeight);
    });
    resize.observe(element);
    renderer.setAnimationLoop(() => {
      const frame = latest.current;
      if (frame !== previous) {
        previous = frame;
        for (const [key] of parts) {
          const part = frame?.[key];
          const { line, dots } = groups[key];
          const points = [],
            edges = [];
          if (part)
            part.joints.forEach((joint, i) => {
              if (!joint.position) return;
              const p = new THREE.Vector3(...joint.position);
              points.push(p);
              const parent = part.joints[part.parents[i]];
              if (parent?.position)
                edges.push(p, new THREE.Vector3(...parent.position));
              // Short orientation rays use the native local +Z axis.
              if (joint.orientation) {
                const direction = new THREE.Vector3(
                  0,
                  0,
                  key === "body" ? 0.055 : 0.015,
                ).applyQuaternion(new THREE.Quaternion(...joint.orientation));
                edges.push(p, p.clone().add(direction));
              }
            });
          line.geometry.dispose();
          dots.geometry.dispose();
          line.geometry = new THREE.BufferGeometry().setFromPoints(edges);
          dots.geometry = new THREE.BufferGeometry().setFromPoints(points);
          line.material.color.copy(
            color(part?.fresh ? "--foreground" : "--caption"),
          );
          dots.material.color.copy(line.material.color);
        }
        if (!centered) center();
      }
      controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      view.current = null;
      resize.disconnect();
      controls.dispose();
      renderer.setAnimationLoop(null);
      for (const { line, dots } of Object.values(groups)) {
        for (const object of [line, dots]) {
          object.geometry.dispose();
          object.material.dispose();
        }
      }
      grid.geometry.dispose();
      grid.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [dark]);
  const service = state.services.tracking || {};
  const tracking = state.tracking;
  const status = trackingStatus(service, tracking);
  return (
    <section className="card tracking-view" aria-label="XRoboToolkit 3D pose">
      <div
        className="tracking-canvas"
        ref={host}
        aria-label="Native Y-up body and hand skeleton"
      />
      <div className="tracking-toolbar">
        <div
          className={"service " + status.tone}
          title={service.error || state.endpoints?.tracking}
        >
          <span className="service-dot" />
          <strong>XRoboToolkit</strong>
          <span>{status.label}</span>
        </div>
        <button
          className="icon-button"
          aria-label="Center tracking pose"
          title="Center pose"
          onClick={() => view.current?.center()}
        >
          <Focus size={18} />
        </button>
      </div>
      {(error || !tracking || !parts.some(([key]) => tracking[key])) && (
        <p className="tracking-empty">
          {error ||
            (service.error
              ? "XRoboToolkit is unavailable. Check its address in Settings."
              : "Waiting for body or hand tracking.")}
        </p>
      )}
      <aside className="tracking-feedback" aria-label="Tracking source status">
        {parts.map(([key, label]) => (
          <div key={key}>
            <span>{label}</span>
            <strong className={tracking?.[key]?.fresh ? "text-success" : ""}>
              {!tracking?.[key]
                ? "--"
                : `${tracking[key].fresh ? "Live" : "Stale"} · ${number(tracking[key].ageMs)} ms`}
            </strong>
          </div>
        ))}
        <small>{tracking?.deviceId || "--"} · Y-up · metres</small>
      </aside>
    </section>
  );
}
