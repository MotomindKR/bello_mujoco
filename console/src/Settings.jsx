import React, { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { GamepadStyle, GamepadButton } from "./GamepadButton.jsx";

export function Settings({
  endpoints,
  controllerStyle,
  theme: preferredTheme,
  onClose,
}) {
  const dialog = useRef(null);
  const [addresses, setAddresses] = useState(endpoints);
  const [style, setStyle] = useState(controllerStyle);
  const hosts = Object.values(endpoints).map((address) =>
    address.slice(0, address.lastIndexOf(":")),
  );
  const [host, setHost] = useState(
    hosts.every((value) => value === hosts[0]) ? hosts[0] : "",
  );
  const [theme, setTheme] = useState(preferredTheme || "system");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    dialog.current.showModal();
  }, []);
  const save = async (event) => {
    event.preventDefault();
    setSaving(true);
    setError("");
    try {
      const result = await window.bello.saveSettings({
        ...addresses,
        controllerStyle: style,
        theme,
      });
      if (result.error) setError(result.error);
      else onClose();
    } catch {
      setError("Could not apply settings. Try saving again.");
    } finally {
      setSaving(false);
    }
  };
  return (
    <dialog
      ref={dialog}
      className="settings-dialog"
      aria-labelledby="settings-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!saving) onClose();
      }}
    >
      <form onSubmit={save}>
        <div className="dialog-heading">
          <h2 id="settings-title">Settings</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close settings"
            disabled={saving}
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        <fieldset className="controller-style-picker" disabled={saving}>
          <legend>Appearance</legend>
          <div className="controller-style-options">
            {[
              ["light", "Light"],
              ["dark", "Dark"],
              ["system", "System"],
            ].map(([value, label]) => (
              <label key={value} className={theme === value ? "selected" : ""}>
                <input
                  type="radio"
                  name="theme"
                  aria-label={label}
                  value={value}
                  checked={theme === value}
                  onChange={() => setTheme(value)}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset className="controller-style-picker" disabled={saving}>
          <legend>Controller prompts</legend>
          <div className="controller-style-options">
            {[
              ["playstation", "PlayStation"],
              ["xbox", "Xbox"],
              ["switch", "Switch"],
            ].map(([value, label]) => (
              <label key={value} className={style === value ? "selected" : ""}>
                <input
                  type="radio"
                  name="controller-style"
                  aria-label={label}
                  value={value}
                  checked={style === value}
                  onChange={() => setStyle(value)}
                />
                <span>{label}</span>
                <GamepadStyle.Provider value={value}>
                  <span className="controller-style-preview">
                    <GamepadButton name="A" />
                    <GamepadButton name="L1" />
                  </span>
                </GamepadStyle.Provider>
              </label>
            ))}
          </div>
        </fieldset>
        <h3 className="settings-section-title">Robot connection</h3>
        <label htmlFor="robot-host">Robot IP or hostname</label>
        <input
          id="robot-host"
          autoFocus
          autoComplete="off"
          spellCheck={false}
          placeholder="10.42.0.1"
          value={host}
          disabled={saving}
          aria-describedby="host-help connection-error"
          onChange={(event) => {
            const value = event.target.value;
            setHost(value);
            const host = value.trim();
            const formatted =
              host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
            setAddresses(
              Object.fromEntries(
                Object.entries(addresses).map(([key, address]) => [
                  key,
                  `${formatted}${address.slice(address.lastIndexOf(":"))}`,
                ]),
              ),
            );
            setError("");
          }}
        />
        <small id="host-help">
          Applies to every service and keeps the current ports.
        </small>
        <details className="service-addresses">
          <summary>Individual service addresses</summary>
          {Object.entries(addresses).map(([key, address]) => (
            <div key={key}>
              <label htmlFor={`address-${key}`}>
                {key === "hardware"
                  ? "Hardware"
                  : key === "estimator"
                    ? "Estimator"
                    : key === "tracking"
                      ? "XRoboToolkit"
                      : "Runner"}
              </label>
              <input
                id={`address-${key}`}
                value={address}
                disabled={saving}
                autoComplete="off"
                spellCheck={false}
                aria-describedby="connection-error"
                onChange={(event) => {
                  setAddresses({ ...addresses, [key]: event.target.value });
                  setHost("");
                  setError("");
                }}
              />
            </div>
          ))}
        </details>
        <p className="connection-note">
          Changing connections ends the current operator session. Release all
          controller inputs before reconnecting.
        </p>
        <p id="connection-error" className="field-error" role="alert">
          {error}
        </p>
        <div className="dialog-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={saving}
            onClick={onClose}
          >
            Cancel
          </button>
          <button type="submit" className="primary-button" disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
