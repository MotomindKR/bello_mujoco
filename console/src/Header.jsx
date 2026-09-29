import React from "react";
import { Settings2, Maximize, Minimize } from "lucide-react";
import { PowerStatus } from "./PowerStatus.jsx";
import { MotorTemperature } from "./MotorTemperature.jsx";

const modeName = (value) =>
  ({
    MODE_DISARMED: "Disarmed",
    MODE_HOLD: "Hold",
    MODE_STAND: "Stand",
    MODE_LOCOMOTION: "Locomotion",
    MODE_MOTION: "Motion",
    MODE_FAULT: "Fault",
  })[value] || "Waiting";

function Badge({ name, service }) {
  const live = service.fresh;
  const errorId = `service-error-${name.toLowerCase()}`;
  return (
    <div
      className={"service " + (live ? "online" : "offline")}
      tabIndex={service.error ? 0 : undefined}
      aria-describedby={service.error ? errorId : undefined}
    >
      <span className="service-dot" />
      <div>
        <strong>{name}</strong>
        <small>
          {live
            ? `Live · ${service.ageMs} ms`
            : service.busy
              ? "In use"
              : service.error
                ? "Error"
                : service.connected
                  ? "Stale"
                  : "Waiting"}
        </small>
      </div>
      {service.error && (
        <div id={errorId} role="tooltip" className="service-tooltip">
          {service.error}
        </div>
      )}
    </div>
  );
}

export function Header({ state, fault, onSettings }) {
  const operator = state.operator || {};
  const mode = modeName(operator.mode);
  return (
    <header className="topbar">
      <div
        className={`header-status${fault || mode === "Fault" ? " has-fault" : ""}`}
        role="status"
        aria-label="Robot status"
        data-fresh={state.services.runner.fresh}
        title={`${state.services.runner.fresh ? "" : "Stale. "}${fault || operator.phase || "Waiting for runner."}`}
      >
        <strong>{mode}</strong>
      </div>
      <div className="header-indicators">
        <PowerStatus state={state} />
        <MotorTemperature state={state} />
        <div className="connections">
          <Badge name="Hardware" service={state.services.hardware} />
          <Badge name="Estimator" service={state.services.estimator} />
          <Badge name="Runner" service={state.services.runner} />
          <Badge
            name="Tracking"
            service={{
              ...state.services.tracking,
              fresh:
                state.services.tracking.fresh && state.tracking?.body?.fresh,
              ageMs: state.tracking?.body
                ? Math.round(state.tracking.body.ageMs)
                : state.services.tracking.ageMs,
            }}
          />
          <button
            type="button"
            className="settings-button"
            aria-label={
              state.fullscreen ? "Exit fullscreen" : "Enter fullscreen"
            }
            title={state.fullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            onClick={() => window.bello.toggleFullscreen()}
          >
            {state.fullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
          </button>
          <button
            type="button"
            className="settings-button"
            aria-label="Settings"
            title="Settings"
            disabled={!state.endpoints}
            onClick={onSettings}
          >
            <Settings2 size={18} />
          </button>
        </div>
      </div>
    </header>
  );
}
