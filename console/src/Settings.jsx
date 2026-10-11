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
  const [version, setVersion] = useState("");
  const [updateError, setUpdateError] = useState("");
  const [update, setUpdate] = useState(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [installingUpdate, setInstallingUpdate] = useState(false);
  const [updateProgress, setUpdateProgress] = useState(null);
  useEffect(() => {
    dialog.current.showModal();
    window.bello.consoleVersion().then(setVersion);
    return window.bello.subscribeUpdateProgress(setUpdateProgress);
  }, []);
  const checkUpdate = async () => {
    setCheckingUpdate(true);
    setUpdateError("");
    setUpdate(null);
    try {
      const result = await window.bello.checkConsoleUpdate();
      if (result.error) setUpdateError(result.error);
      else setUpdate(result);
    } catch {
      setUpdateError("Could not check for updates.");
    } finally {
      setCheckingUpdate(false);
    }
  };
  const installUpdate = async () => {
    setInstallingUpdate(true);
    setUpdateError("");
    try {
      const result = await window.bello.installConsoleUpdate();
      if (result.error) setUpdateError(result.error);
    } catch {
      setUpdateError("Could not install the update.");
    } finally {
      setInstallingUpdate(false);
      setUpdateProgress(null);
    }
  };
  const openReleases = async () => {
    setUpdateError("");
    try {
      const result = await window.bello.openConsoleReleases();
      if (result.error) setUpdateError(result.error);
    } catch {
      setUpdateError("Could not open the release page in your browser.");
    }
  };
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
        if (!saving && !installingUpdate) onClose();
      }}
    >
      <form onSubmit={save}>
        <div className="dialog-heading">
          <h2 id="settings-title">Settings</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close settings"
            disabled={saving || installingUpdate}
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        <div className="settings-content">
          <fieldset className="controller-style-picker" disabled={saving}>
            <legend>Appearance</legend>
            <div className="controller-style-options">
              {[
                ["light", "Light"],
                ["dark", "Dark"],
                ["system", "System"],
              ].map(([value, label]) => (
                <label
                  key={value}
                  className={theme === value ? "selected" : ""}
                >
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
                <label
                  key={value}
                  className={style === value ? "selected" : ""}
                >
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
                      <GamepadButton name="L2" />
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
                host.includes(":") && !host.startsWith("[")
                  ? `[${host}]`
                  : host;
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
          <h3 className="settings-section-title">Console updates</h3>
          <p className="settings-version">
            {version
              ? `Installed version: ${version}`
              : "Checking installed version…"}
          </p>
          <p className="settings-update-help">
            Updates require a disarmed robot and system authorization.
          </p>
          <div className="settings-update-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={checkingUpdate || installingUpdate}
              onClick={checkUpdate}
            >
              {checkingUpdate ? "Checking…" : "Check for updates"}
            </button>
            {update?.version && update.canInstall && (
              <button
                type="button"
                className="primary-button"
                disabled={installingUpdate}
                onClick={installUpdate}
              >
                {installingUpdate
                  ? "Installing…"
                  : `Install ${update.version} (${update.packageKind.toUpperCase()})`}
              </button>
            )}
            <button
              type="button"
              className="secondary-button"
              disabled={installingUpdate}
              onClick={openReleases}
            >
              Open releases
            </button>
          </div>
          {update?.current && <p>Console is up to date.</p>}
          {update?.version && !update.canInstall && (
            <p>Install the DEB or RPM to enable in-app updates.</p>
          )}
          {updateProgress?.phase === "downloading" && (
            <p role="status">
              Downloading update
              {updateProgress.total > 0
                ? `: ${Math.floor((100 * updateProgress.bytes) / updateProgress.total)}%`
                : "…"}
            </p>
          )}
          {updateProgress?.phase === "installing" && (
            <p role="status">
              Waiting for system authorization and installation…
            </p>
          )}
          <p className="field-error" role="alert">
            {updateError}
          </p>
        </div>
        <div className="dialog-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={saving || installingUpdate}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="primary-button"
            disabled={saving || installingUpdate}
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
