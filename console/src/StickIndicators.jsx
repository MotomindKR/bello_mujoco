import React from "react";
import { number } from "./format.mjs";
import { GamepadButton } from "./GamepadButton.jsx";

function Stick({ label, x, y, connected, children }) {
  const bounded = (value) => Math.max(-1, Math.min(1, Number(value) || 0));
  return (
    <div className="stick-indicator">
      <strong className="stick-label">{label}</strong>
      <div
        className="stick-dial"
        role="img"
        aria-label={
          connected
            ? `${label}: horizontal ${number(x)}, vertical ${number(y)}`
            : `${label}: disconnected`
        }
      >
        <div className="stick-axis horizontal" />
        <div className="stick-axis vertical" />
        <div className="stick-center" />
        {connected && (
          <div
            className="stick-thumb"
            style={{
              left: `${50 + bounded(x) * 36}%`,
              top: `${50 + bounded(y) * 36}%`,
            }}
          />
        )}
      </div>
      <dl className="stick-velocity">{children}</dl>
    </div>
  );
}
function Metric({ label, value, unit }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        {number(value)} <span>{unit}</span>
      </dd>
    </div>
  );
}
export function StickIndicators({ state }) {
  const sample = state.controllerSample || {};
  const operator = state.operator || {};
  const active =
    state.services.runner.fresh && operator.mode === "MODE_LOCOMOTION";
  const velocity = active ? operator.velocityCommand : null;
  const straightening =
    operator.mode === "MODE_HOLD" && operator.phase === "straightening";
  const sitting =
    operator.mode === "MODE_SIT" &&
    ["upright", "crouching", "bent"].includes(operator.phase);
  const assisted = straightening || sitting;
  const acceptingInput =
    assisted &&
    state.services.runner.fresh &&
    !operator.neutralRequired &&
    operator.controllerConnected;
  if (!active && !assisted) return null;
  return (
    <section
      className="sticks"
      aria-label="Controller sticks and command velocity"
    >
      <div className="mode-heading">
        <span className="stick-actions">
          {sitting && (
            <span>
              <GamepadButton name="LeftStickDown" sample={sample} /> Bend
            </span>
          )}
          {assisted ? (
            <span>
              <GamepadButton name="LeftStickUp" sample={sample} /> Straighten
            </span>
          ) : (
            "Command velocity"
          )}
        </span>
        <span>{assisted ? "Release to pause" : "Active"}</span>
      </div>
      <div className={"stick-grid" + (assisted ? " assisted" : "")}>
        <Stick
          label="Left stick"
          x={sample.lateral}
          y={-sample.forward}
          connected={sample.connected}
        >
          {assisted ? (
            <Metric
              label={sitting ? "Bent" : "Upright"}
              value={operator.progress * 100}
              unit="%"
            />
          ) : (
            <Metric label="Forward" value={velocity?.forwardMps} unit="m/s" />
          )}
          {assisted ? (
            <Metric
              label="Input"
              value={
                acceptingInput
                  ? (sitting
                      ? -(sample.forward || 0)
                      : Math.max(0, sample.forward || 0)) * 100
                  : 0
              }
              unit="%"
            />
          ) : (
            <Metric label="Right" value={velocity?.rightMps} unit="m/s" />
          )}
        </Stick>
        {active && (
          <Stick
            label="Right stick"
            x={-sample.yaw}
            y={0}
            connected={sample.connected}
          >
            <Metric label="Turn" value={velocity?.yawRps} unit="rad/s" />
          </Stick>
        )}
      </div>
    </section>
  );
}
