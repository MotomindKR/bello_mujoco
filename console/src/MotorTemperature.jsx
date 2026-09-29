import React from "react";
import { Thermometer } from "lucide-react";
import { motorTemperature } from "./motor-temperature.mjs";
import { number } from "./format.mjs";

export function MotorTemperature({ state }) {
  const { hottest, hot, incomplete } = motorTemperature(state);
  const label = hot
    ? `${number(hottest.temperature)} °C`
    : incomplete
      ? "Temp --"
      : "Temp OK";
  const detail = hot
    ? hottest.motorName
    : incomplete
      ? "Waiting for fresh telemetry"
      : "All motors at or below 60 °C";
  return (
    <div
      className={`header-temperature${hot ? " text-error" : incomplete ? "" : " text-success"}`}
      role="status"
      aria-label="Motor temperature"
      title={`${label} · ${detail}${hot && incomplete ? " · Other telemetry stale or missing" : ""}`}
    >
      <span>
        <Thermometer size={16} aria-hidden="true" />
        <strong>{label}</strong>
      </span>
      {(hot || incomplete) && (
        <small>
          {hot
            ? `${incomplete ? "Partial · " : ""}${hottest.motorName}`
            : "No fresh data"}
        </small>
      )}
    </div>
  );
}
