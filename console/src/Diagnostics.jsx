import React, { useState } from "react";
import { Search } from "lucide-react";
import { scalar } from "./format.mjs";

export const statusTabs = [
  "Overview",
  "Tracking",
  "Hardware",
  "Pose",
  "Operator",
  "Services",
];
const stamp = (value) => (value ? new Date(value).toISOString() : "--");
const matches = (query, ...values) =>
  values.join(" ").toLowerCase().includes(query.toLowerCase());

function fields(value, path = "") {
  if (value == null || typeof value !== "object")
    return [
      [path || "value", scalar(value), value == null ? "--" : String(value)],
    ];
  const entries = Object.entries(value);
  if (!entries.length)
    return [[path || "value", Array.isArray(value) ? "[]" : "{}"]];
  return entries.flatMap(([key, child]) =>
    fields(child, path ? `${path}.${key}` : key),
  );
}

function Payload({ title, value, query }) {
  const rows =
    value == null
      ? []
      : fields(value).filter(([key, value]) =>
          matches(query, title, key, value),
        );
  return (
    <section className="card diagnostic-card">
      <h2>{title}</h2>
      {rows.length ? (
        <div className="table-scroll">
          <table className="field-table">
            <thead>
              <tr>
                <th scope="col">Field</th>
                <th scope="col">Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([key, value, raw]) => (
                <tr key={key}>
                  <th scope="row">{key}</th>
                  <td title={raw}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="empty-status">
          {query
            ? "No fields match this search."
            : "No sample received from this source yet."}
        </p>
      )}
    </section>
  );
}

function Hardware({ state, query }) {
  const motors = Object.values(state.motors)
    .filter((motor) =>
      matches(query, ...Object.keys(motor), ...Object.values(motor)),
    )
    .sort((a, b) => a.motorName.localeCompare(b.motorName));
  return (
    <>
      <section className="card diagnostic-card full-width">
        <h2>
          Motor telemetry <span className="counter">{motors.length} shown</span>
        </h2>
        <div
          className="table-scroll"
          tabIndex={0}
          aria-label="Motor telemetry table"
        >
          <table className="motor-table">
            <thead>
              <tr>
                {[
                  "Motor",
                  "Group",
                  "Feedback",
                  "Position (rad)",
                  "Velocity (rad/s)",
                  "Torque (N m)",
                  "Temperature (°C)",
                  "Age (ms)",
                  "Fault bits",
                  "Mode status",
                ].map((name) => (
                  <th scope="col" key={name}>
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {motors.map((motor) => (
                <tr key={motor.motorName}>
                  <th scope="row">{motor.motorName}</th>
                  <td>{motor.group || "--"}</td>
                  <td className={motor.fresh ? "text-success" : "text-error"}>
                    {motor.fresh ? "Fresh" : "Stale"}
                  </td>
                  {[
                    "position",
                    "velocity",
                    "torque",
                    "temperature",
                    "feedbackAgeMs",
                    "faultBits",
                    "modeStatus",
                  ].map((key) => (
                    <td
                      className="numeric"
                      key={key}
                      title={String(motor[key] ?? "--")}
                    >
                      {scalar(motor[key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!motors.length && (
          <p className="empty-status">
            {query
              ? "No motors match this search."
              : "Waiting for motor feedback from hardwared."}
          </p>
        )}
      </section>
      <Payload
        title="Groups"
        value={Object.keys(state.groups).length ? state.groups : null}
        query={query}
      />
      <Payload
        title="Component health"
        value={Object.keys(state.health || {}).length ? state.health : null}
        query={query}
      />
      <Payload
        title="Power boards"
        value={
          Object.keys(state.powerBoards || {}).length ? state.powerBoards : null
        }
        query={query}
      />
      <Payload
        title="Command policy events · latest 100"
        value={
          state.commandPolicyEvents?.length ? state.commandPolicyEvents : null
        }
        query={query}
      />
      <Payload
        title="Hardware inventory"
        value={state.inventory}
        query={query}
      />
    </>
  );
}

function Services({ state, query }) {
  return (
    <>
      <section className="card diagnostic-card full-width">
        <h2>Direct connections</h2>
        <div
          className="table-scroll"
          tabIndex={0}
          aria-label="Service connections table"
        >
          <table className="service-table">
            <thead>
              <tr>
                {[
                  "Service",
                  "Endpoint",
                  "Status",
                  "Receipt age (ms)",
                  "Last receipt (UTC)",
                  "Attempts",
                  "Messages",
                  "Error",
                ].map((name) => (
                  <th scope="col" key={name}>
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(state.services)
                .filter(([key, value]) =>
                  matches(query, key, state.endpoints?.[key], value.error),
                )
                .map(([key, service]) => (
                  <tr key={key}>
                    <th scope="row">{key}</th>
                    <td>{state.endpoints?.[key] || "--"}</td>
                    <td
                      className={service.fresh ? "text-success" : "text-error"}
                    >
                      {service.fresh
                        ? "Live"
                        : service.connected
                          ? "Stale"
                          : "Disconnected"}
                    </td>
                    <td className="numeric">{scalar(service.ageMs)}</td>
                    <td>{stamp(service.last)}</td>
                    <td className="numeric">{service.attempts || 0}</td>
                    <td className="numeric">{service.messages || 0}</td>
                    <td className="service-error">{service.error || "None"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="diagnostic-note">
          Counts cover this endpoint session. Hardware messages include
          inventory responses and telemetry. Receipt age measures time since the
          latest response; sample ages appear in Hardware and Pose.
        </p>
      </section>
      <Payload
        title="Display and controller configuration"
        value={state.displayConfig}
        query={query}
      />
      <Payload
        title="Connection status fields"
        value={state.services}
        query={query}
      />
    </>
  );
}

export function Diagnostics({ tab, state }) {
  const [query, setQuery] = useState("");
  return (
    <>
      <div className="diagnostic-toolbar">
        <label className="status-search">
          <Search size={16} />
          <input
            aria-label="Search status"
            placeholder="Search fields and values"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      </div>
      <div className="diagnostic-grid">
        {tab === "Hardware" && <Hardware state={state} query={query} />}
        {tab === "Pose" && (
          <>
            <Payload title="Estimator pose" value={state.pose} query={query} />
            <Payload
              title="IMU telemetry"
              value={Object.keys(state.imus || {}).length ? state.imus : null}
              query={query}
            />
          </>
        )}
        {tab === "Operator" && (
          <>
            <Payload
              title="Runner status"
              value={state.operator}
              query={query}
            />
            <Payload
              title="Controller device"
              value={state.controller}
              query={query}
            />
            <Payload
              title="Local controller sample"
              value={state.controllerSample}
              query={query}
            />
            <Payload
              title="Operator transport"
              value={{
                endpoint: state.endpoints?.runner,
                lastSentSequence: state.lastSentSequence,
                ...state.services.runner,
              }}
              query={query}
            />
          </>
        )}
        {tab === "Services" && <Services state={state} query={query} />}
      </div>
    </>
  );
}
