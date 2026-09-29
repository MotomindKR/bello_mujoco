import React, { useState } from "react";
import { DirectionHint } from "./DirectionHint.jsx";
import { operatorGuide } from "./operator-guide.mjs";
import { AbortButton } from "./AbortButton.jsx";
import { MotionPicker } from "./MotionPicker.jsx";
import { StickIndicators } from "./StickIndicators.jsx";
import { number } from "./format.mjs";

import { GamepadButton as Key } from "./GamepadButton.jsx";

export function OperatorCard({ state, children }) {
  const [selectionError, setSelectionError] = useState("");
  async function selectChoice(kind, name) {
    try {
      const result = await window.bello.selectChoice(kind, name);
      setSelectionError(result.error || "");
    } catch {
      setSelectionError("Runner unavailable.");
    }
  }
  const operator = state.operator || {};
  const sample = state.controllerSample || {};
  const guide = operatorGuide(state);
  const live = state.services.runner.fresh;
  const choosing = live && operator.mode === "MODE_STAND";
  const progressing =
    operator.mode === "MODE_MOTION" ||
    (operator.mode === "MODE_HOLD" && operator.phase === "ramping");
  const progress = Math.min(
    100,
    Math.max(0, Number(operator.progress || 0) * 100),
  );
  return (
    <section className="card operator-card" aria-label="Controller and status">
      <div className="operator-guide">
        <div className="operator-guide-heading">
          <h3 title={guide.title}>{guide.title}</h3>
          <span
            className="counter"
            title={state.controller.name || state.controller.error}
          >
            {state.controller.connected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <div className="operator-transition">
          {guide.keys && (
            <div className="next-action">
              <div>
                {guide.keys.map((name, i) => (
                  <React.Fragment key={name}>
                    {i > 0 && <span> + </span>}
                    <Key
                      name={name}
                      sample={sample}
                      progress={
                        live
                          ? name === "A"
                            ? operator.confirmProgress
                            : operator.holdProgress
                          : 0
                      }
                    />
                  </React.Fragment>
                ))}
              </div>
              <strong>{guide.duration}</strong>
            </div>
          )}
          {progressing && (
            <div className="motion-status">
              <div className="stat-pair">
                <span>{operator.activeMotion || "Hold pose"}</span>
                <strong>
                  {operator.phase} · {number(progress)}%{!live && " · Stale"}
                </strong>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-label="Phase progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress}
              >
                <div style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="mode-heading">
        <span
          title={selectionError}
          role={selectionError ? "alert" : undefined}
        >
          {selectionError || (choosing ? "Mode" : "Modes · Stand only")}
        </span>
      </div>
      <div className="mode-picker">
        <div className="mode-row">
          <DirectionHint direction="Left" sample={sample} />
          <div className={"mode-list" + (choosing ? "" : " unavailable")}>
            {(operator.modes || []).map((name) => (
              <button
                type="button"
                disabled={!choosing}
                aria-pressed={operator.selected === name}
                title={name}
                onClick={() => selectChoice("mode", name)}
                className={
                  "mode-option" +
                  (operator.selected === name ? " selected" : "")
                }
                key={name}
              >
                <span>{name}</span>
              </button>
            ))}
          </div>
          <DirectionHint direction="Right" sample={sample} />
        </div>
        <div className="mode-detail">
          {operator.selected === "Tracking" && (
            <div
              className="tracking-toggle tracking-reference"
              role="status"
              aria-label="SONIC reference"
              title={operator.sonicStatus || "Waiting for reference"}
            >
              <span
                className={live && operator.sonicReady ? "text-success" : ""}
              >
                {!live
                  ? "Stale"
                  : operator.sonicReady
                    ? "Ready"
                    : operator.sonicStatus || "Waiting"}
              </span>
              <span className="tracking-toggle-state">
                <span>Buffer {number(operator.sonicBufferedMs)} ms</span>
                <span>Delay {number(operator.sonicDelayMs)} ms</span>
              </span>
            </div>
          )}
          {operator.selected === "Motion" && (
            <MotionPicker
              operator={operator}
              sample={sample}
              choosing={choosing}
              onSelect={(name) => selectChoice("motion", name)}
              onToggle={(open) => selectChoice("motionMenuOpen", open)}
            />
          )}
          {operator.selected === "Locomotion" && (
            <button
              type="button"
              role="switch"
              aria-label="Upper-body teleop"
              aria-checked={Boolean(operator.trackingEnabled)}
              className="tracking-toggle"
              disabled={
                !live ||
                operator.mode !== "MODE_LOCOMOTION" ||
                !operator.trackingAvailable ||
                !state.controller.connected ||
                operator.neutralRequired
              }
              title={
                operator.trackingAvailable
                  ? "Toggle in Locomotion."
                  : "Enable control.following in the runner configuration."
              }
              onClick={() =>
                selectChoice("trackingEnabled", !operator.trackingEnabled)
              }
            >
              <span>Upper-body teleop</span>
              <span className="tracking-toggle-state">
                <Key name="Y" sample={sample} />
                <span className={operator.trackingLive ? "text-success" : ""}>
                  {!live
                    ? "--"
                    : !operator.trackingAvailable
                      ? "Unavailable"
                      : !operator.trackingEnabled
                        ? "Off"
                        : operator.trackingLive
                          ? "Live"
                          : "Waiting"}
                </span>
                <span className="toggle-track">
                  <span />
                </span>
              </span>
            </button>
          )}
        </div>
      </div>
      <StickIndicators state={state} />
      <AbortButton
        available={live}
        sample={sample}
        progress={live ? operator.abortProgress : 0}
      />
      {children}
    </section>
  );
}
