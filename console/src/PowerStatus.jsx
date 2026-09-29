import React from "react";
import { Battery, CircleHelp, TriangleAlert, Zap, ZapOff } from "lucide-react";
import { number } from "./format.mjs";

const value = (reading, unit) => `${number(reading)} ${unit}`;
const reading = (board, key) =>
  board?.samples?.find((sample) => sample[key] != null);
const deviceState = (board) =>
  ({
    DEVICE_STATE_ONLINE: "Online",
    DEVICE_STATE_STARTING: "Starting",
    DEVICE_STATE_CONNECTING: "Connecting",
    DEVICE_STATE_DEGRADED: "Degraded",
    DEVICE_STATE_RECONNECTING: "Reconnecting",
    DEVICE_STATE_FAULT: "Fault",
    DEVICE_STATE_SHUTDOWN: "Shutdown",
  })[board?.status?.state] || "Unknown";

export function powerAlerts(board) {
  const alerts = [];
  for (const [key, field, label, critical] of [
    ["iwc", "warning", "Warning", false],
    ["iwc", "critical", "Critical", true],
    ["legacyFault", "alarm", "Alarm", false],
    ["legacyFault", "fault", "Fault", true],
  ]) {
    const sample = reading(board, key);
    const flags = sample?.[key]?.[field];
    if (flags && (Number(flags.bits) !== 0 || flags.activeNames?.length)) {
      alerts.push({
        critical,
        text: `${label}: ${flags.activeNames?.length ? flags.activeNames.join(", ") : `bits ${flags.bits}`}${sample.stale || !board.fresh ? " (stale)" : ""}`,
      });
    }
  }
  if (board?.status?.state === "DEVICE_STATE_FAULT")
    alerts.push({
      critical: true,
      text: `Board fault: ${board.status.detail || "Check Hardware status."}${!board.fresh ? " (stale)" : ""}`,
    });
  return alerts;
}

function Metric({ label, kind, sample, board, children }) {
  const stale = sample && (!board?.fresh || sample.stale);
  const status = !sample
    ? "Waiting for a reading."
    : `${stale ? "Stale reading. " : ""}Sample age: ${sample.sampleAgeMs} ms.`;
  return (
    <span
      className={`header-power-metric power-${kind}${stale ? " power-stale" : ""}`}
      aria-label={label}
      title={`${label}. ${status}`}
    >
      <span className="power-metric-heading">
        <span>{label}</span>
        <small className={stale ? "" : "status-placeholder"}>Stale</small>
      </span>
      <span className="power-reading">{children}</span>
    </span>
  );
}

export function PowerStatus({ state }) {
  const names = [
    ...new Set([
      ...(state.inventory?.powerBoards || []).map((board) => board.name),
      ...Object.keys(state.powerBoards || {}),
    ]),
  ].sort();
  return (
    <div className="header-power" role="group" aria-label="Power telemetry">
      {(names.length ? names : [null]).map((name) => {
        const board = state.powerBoards?.[name];
        const motor = reading(board, "motorRail");
        const battery = reading(board, "battery");
        const info = reading(board, "iwc");
        const chargingKnown = Boolean(
          board?.fresh && info && !info.stale && info.iwc.info,
        );
        const charging =
          chargingKnown &&
          info.iwc.info.activeNames?.includes("BATTERY_CHARGING");
        const chargingLabel = !chargingKnown
          ? "Charging status unavailable"
          : charging
            ? "Charging"
            : "Not charging";
        const ChargingIcon = !chargingKnown
          ? CircleHelp
          : charging
            ? Zap
            : ZapOff;
        const charge =
          battery?.battery._socPercent === "socPercent" ? battery : null;
        const alerts = powerAlerts(board);
        const soc = charge?.battery.socPercent;
        const level = Number.isFinite(soc)
          ? Math.min(100, Math.max(0, soc)) / 100
          : 0;
        const batteryTone =
          !Number.isFinite(soc) || !board?.fresh || charge?.stale
            ? "unknown"
            : soc <= 10
              ? "critical"
              : soc <= 20
                ? "low"
                : "normal";
        return (
          <div
            className="header-power-board"
            key={name || "waiting"}
            title={
              name
                ? `${name}: ${deviceState(board)}`
                : "Waiting for power-board telemetry."
            }
          >
            {names.length > 1 && (
              <span className="power-board-name">{name}</span>
            )}
            <Metric
              label="Battery"
              kind="battery"
              sample={charge}
              board={board}
            >
              <Battery
                size={22}
                className={`battery-level battery-${batteryTone}`}
                aria-hidden="true"
              >
                <rect
                  className="battery-fill"
                  x="4"
                  y="8"
                  width={12 * level}
                  height="8"
                  rx="0.5"
                  stroke="none"
                  fill="currentColor"
                />
              </Battery>
              <strong>{value(charge?.battery.socPercent, "%")}</strong>
              <span
                className={
                  "battery-charging" + (charging ? " is-charging" : "")
                }
                role="img"
                aria-label={chargingLabel}
                title={chargingLabel}
              >
                <ChargingIcon size={16} aria-hidden="true" />
              </span>
            </Metric>
            <Metric label="Motor rail" kind="rail" sample={motor} board={board}>
              <strong>{value(motor?.motorRail.voltageV, "V")}</strong>
              <strong>{value(motor?.motorRail.currentA, "A")}</strong>
            </Metric>
            <span
              className={`power-warning${alerts.length ? "" : " status-placeholder"}`}
              tabIndex={alerts.length ? 0 : undefined}
              role={alerts.length ? "img" : undefined}
              aria-hidden={!alerts.length}
              aria-label={alerts.map((alert) => alert.text).join("; ")}
              title={alerts.map((alert) => alert.text).join("\n")}
            >
              <TriangleAlert size={16} aria-hidden="true" />
            </span>
          </div>
        );
      })}
    </div>
  );
}
