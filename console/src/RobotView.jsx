import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import URDFLoader from "urdf-loader";
import { jointPositions, parseTransmissions } from "./transmission.mjs";
import { number } from "./format.mjs";

function disposeObject(root) {
  const resources = new Set();
  root.traverse((object) => {
    if (object.geometry) resources.add(object.geometry);
    const materials = Array.isArray(object.material)
      ? object.material
      : [object.material];
    for (const material of materials) {
      if (!material) continue;
      resources.add(material);
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) resources.add(value);
      }
    }
  });
  for (const resource of resources) resource.dispose();
}

function FeedbackOverlay({ state }) {
  const motors = Object.values(state.motors).sort((a, b) =>
    String(a.motorName).localeCompare(String(b.motorName)),
  );
  return (
    <aside className="model-feedback" aria-label="Robot feedback">
      <div className="feedback-line">
        <span>IMU</span>
        <strong className={state.imu && !state.imu.fresh ? "text-error" : ""}>
          {state.imu
            ? `${state.imu.imuName} · ${state.imu.sampleAgeMs} ms${state.imu.fresh ? "" : " · Stale"}`
            : "--"}
        </strong>
      </div>
      <div className="feedback-line">
        <span>Pose</span>
        <strong
          className={
            state.pose && !state.services.estimator.fresh ? "text-error" : ""
          }
        >
          {state.pose
            ? `${state.pose.trackedFrame} · ${state.pose.sampleAgeMs} ms${state.services.estimator.fresh ? "" : " · Stale"}`
            : "--"}
        </strong>
      </div>
      <div className="feedback-position">
        {["x", "y", "z"].map((axis) => (
          <span key={axis}>
            {axis.toUpperCase()}{" "}
            <strong>{number(state.pose?.translationMetres?.[axis])} m</strong>
          </span>
        ))}
      </div>
      <details className="feedback-motors">
        <summary>
          {motors.length} motor{motors.length === 1 ? "" : "s"}
          {motors.some((motor) => !motor.fresh) ? " · Stale feedback" : ""}
        </summary>
        <div className="motor-list">
          {motors.map((motor) => (
            <details
              key={motor.motorName}
              className={motor.fresh ? "" : "stale"}
            >
              <summary>
                <span>{motor.motorName}</span>
                <strong>{number(motor.position)} rad</strong>
                <small>{number(motor.temperature)} °C</small>
              </summary>
              <div className="stat-pair">
                <span>Velocity</span>
                <strong>{number(motor.velocity)} rad/s</strong>
              </div>
              <div className="stat-pair">
                <span>Torque</span>
                <strong>{number(motor.torque)} N m</strong>
              </div>
              <div className="stat-pair">
                <span>Fault bits</span>
                <strong>{motor.faultBits ?? "--"}</strong>
              </div>
            </details>
          ))}
        </div>
      </details>
    </aside>
  );
}

export function RobotView({ state, dark, visible }) {
  const active = useRef(visible);
  active.current = visible;
  const mount = useRef(null);
  const model = useRef(null);
  const transmissions = useRef([]);
  const [loadError, setLoadError] = useState("");
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setLoaded(false);
    setLoadError("");
    let disposed = false;
    const request = new AbortController();
    const host = mount.current;
    const scene = new THREE.Scene();
    const tokens = getComputedStyle(document.documentElement);
    const color = (name) => tokens.getPropertyValue(name).trim();
    scene.background = new THREE.Color(color("--model-background"));
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    camera.position.set(1.8, 1.25, 2.1);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.5, 0);
    controls.enableDamping = true;
    controls.update();
    const room = new RoomEnvironment();
    const generator = new THREE.PMREMGenerator(renderer);
    const environment = generator.fromScene(room, 0.04);
    generator.dispose();
    room.dispose();
    scene.environment = environment.texture;
    scene.add(
      new THREE.HemisphereLight(
        color("--model-light"),
        color("--model-ground"),
        1.1,
      ),
    );
    const key = new THREE.DirectionalLight(color("--model-light"), 2.2);
    key.position.set(3.5, 6, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(color("--model-light"), 1.0);
    rim.position.set(-4, 2.5, -4);
    scene.add(rim);
    const grid = new THREE.GridHelper(
      8,
      24,
      color("--model-grid"),
      color("--model-grid"),
    );
    scene.add(grid);
    const frame = new THREE.Group();
    frame.rotation.x = -Math.PI / 2;
    scene.add(frame);
    const loader = new URDFLoader(new THREE.LoadingManager());
    loader.parseCollision = false;
    const gltf = new GLTFLoader(loader.manager);
    loader.loadMeshCb = (url, _manager, _material, done) =>
      gltf.load(
        url,
        (result) => {
          if (disposed) disposeObject(result.scene);
          else done(result.scene);
        },
        undefined,
        (error) => done(new THREE.Object3D(), error),
      );
    loader.manager.onError = (url) => {
      if (!disposed)
        setLoadError(`Could not load model: ${url.split("/").pop()}.`);
    };
    loader.manager.onLoad = () => {
      if (disposed) return;
      setLoaded(true);
      const robot = model.current;
      if (robot) {
        const bounds = new THREE.Box3().setFromObject(robot);
        if (Number.isFinite(bounds.min.y)) {
          frame.position.y -= bounds.min.y;
          bounds.setFromObject(frame);
          const center = bounds.getCenter(new THREE.Vector3());
          const size = bounds.getSize(new THREE.Vector3());
          const extent = Math.max(size.y, size.x / camera.aspect);
          const distance =
            (extent /
              (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
            1.3;
          controls.target.copy(center);
          camera.position
            .copy(center)
            .add(
              new THREE.Vector3(1.8, 0.7, 2.1)
                .normalize()
                .multiplyScalar(distance),
            );
          controls.update();
        }
      }
    };
    fetch(window.bello.modelUrl, { signal: request.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Check the robot model path.");
        return response.text();
      })
      .then((xml) => {
        if (disposed) return;
        const document = new DOMParser().parseFromString(xml, "text/xml");
        transmissions.current = parseTransmissions(document);
        const robot = loader.parse(
          xml,
          new URL(".", window.bello.modelUrl).href,
        );
        frame.add(robot);
        model.current = robot;
      })
      .catch((error) => {
        if (!disposed) setLoadError(String(error));
      });
    const observer = new ResizeObserver(() => {
      const width = host.clientWidth,
        height = host.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    });
    observer.observe(host);
    let animation;
    const animate = () => {
      animation = requestAnimationFrame(animate);
      if (active.current) {
        controls.update();
        renderer.render(scene, camera);
      }
    };
    animate();
    return () => {
      disposed = true;
      request.abort();
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      disposeObject(scene);
      environment.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
      model.current = null;
    };
  }, [dark]);
  useEffect(() => {
    const robot = model.current;
    if (!robot) return;
    const coordinates = {};
    for (const [name, motor] of Object.entries(state.motors)) {
      const coordinate = state.inventory?.motors?.find(
        (item) => item.name === name,
      )?.coordinate;
      if (coordinate && motor.fresh && Number.isFinite(Number(motor.position)))
        coordinates[coordinate] = Number(motor.position);
    }
    for (const [name, position] of Object.entries(
      jointPositions(coordinates, transmissions.current),
    )) {
      if (robot.joints[name]) robot.joints[name].setJointValue(position);
    }
    if (
      state.services.estimator.fresh &&
      state.pose?.translationMetres &&
      state.pose?.orientationXyzw
    ) {
      const p = state.pose.translationMetres,
        q = state.pose.orientationXyzw;
      robot.parent.position.y = 0;
      robot.position.set(Number(p.x), Number(p.y), Number(p.z));
      robot.quaternion.set(Number(q.x), Number(q.y), Number(q.z), Number(q.w));
    }
  }, [state]);
  return (
    <div
      className="model"
      ref={mount}
      data-ready={loaded}
      data-error={loadError}
    >
      <span className="model-tag">
        {loadError ||
          (state.services.hardware.fresh
            ? "BELLO MINI · Live pose"
            : "BELLO MINI · Waiting for feedback")}
      </span>
      <FeedbackOverlay state={state} />
    </div>
  );
}
