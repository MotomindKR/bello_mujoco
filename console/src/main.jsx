import React, { useEffect, useLayoutEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "pretendard/dist/web/variable/pretendardvariable.css";
import "./style.css";
import { Diagnostics, statusTabs } from "./Diagnostics.jsx";
import { powerAlerts } from "./PowerStatus.jsx";
import { GamepadStyle, GamepadTheme } from "./GamepadButton.jsx";
import { OperatorCard } from "./OperatorCard.jsx";
import { TrackingView } from "./TrackingView.jsx";
import { RobotView } from "./RobotView.jsx";
import { Header } from "./Header.jsx";
import { Settings } from "./Settings.jsx";
import { FaultStatus } from "./FaultStatus.jsx";

const initial = {
  services: { hardware: {}, estimator: {}, runner: {}, tracking: {} },
  controller: {},
  motors: {},
  groups: {},
  imu: null,
  pose: null,
  operator: null,
};
function App() {
  const [state, setState] = useState(initial);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tab, setTab] = useState("Overview");
  const [systemDark, setSystemDark] = useState(
    () => matchMedia("(prefers-color-scheme: dark)").matches,
  );
  useEffect(() => {
    const preference = matchMedia("(prefers-color-scheme: dark)");
    const changed = (event) => setSystemDark(event.matches);
    preference.addEventListener("change", changed);
    return () => preference.removeEventListener("change", changed);
  }, []);
  const dark =
    state.theme === "dark" || (state.theme !== "light" && systemDark);
  useLayoutEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);
  useEffect(() => window.bello.subscribe(setState), []);
  const operator = state.operator || {};
  const fault =
    operator.fault ||
    Object.values(state.motors)
      .filter((motor) => motor.faultBits)
      .map((motor) => `${motor.motorName}: fault bits ${motor.faultBits}`)
      .join("; ") ||
    Object.values(state.groups).find(
      (group) => group.detail && group.state === "GROUP_STATE_FAULT",
    )?.detail ||
    Object.values(state.powerBoards || {})
      .flatMap((board) =>
        powerAlerts(board)
          .filter((alert) => alert.critical)
          .map((alert) => `${board.boardName}: ${alert.text}`),
      )
      .join("; ");
  return (
    <GamepadTheme.Provider value={dark}>
      <GamepadStyle.Provider value={state.controllerStyle || "xbox"}>
        <main className={tab === "Overview" ? "overview" : undefined}>
          <Header
            state={state}
            fault={fault}
            onSettings={() => setSettingsOpen(true)}
          />
          {settingsOpen && (
            <Settings
              endpoints={state.endpoints}
              controllerStyle={state.controllerStyle}
              theme={state.theme}
              onClose={() => setSettingsOpen(false)}
            />
          )}
          <nav
            className="status-tabs"
            role="tablist"
            aria-label="Console views"
          >
            {statusTabs.map((name, index) => (
              <button
                key={name}
                type="button"
                role="tab"
                id={`tab-${name}`}
                aria-controls={`panel-${name}`}
                aria-selected={tab === name}
                tabIndex={tab === name ? 0 : -1}
                onClick={() => setTab(name)}
                onKeyDown={(event) => {
                  const direction =
                    event.key === "ArrowRight"
                      ? 1
                      : event.key === "ArrowLeft"
                        ? -1
                        : 0;
                  if (!direction && event.key !== "Home" && event.key !== "End")
                    return;
                  event.preventDefault();
                  const next =
                    event.key === "Home"
                      ? statusTabs[0]
                      : event.key === "End"
                        ? statusTabs.at(-1)
                        : statusTabs[
                            (index + direction + statusTabs.length) %
                              statusTabs.length
                          ];
                  setTab(next);
                  document.getElementById(`tab-${next}`).focus();
                }}
              >
                {name}
              </button>
            ))}
          </nav>
          {statusTabs.slice(1).map((name) => (
            <section
              key={name}
              role="tabpanel"
              id={`panel-${name}`}
              aria-labelledby={`tab-${name}`}
              hidden={tab !== name}
              tabIndex={0}
            >
              {tab === name &&
                (name === "Tracking" ? (
                  <TrackingView state={state} dark={dark} />
                ) : (
                  <Diagnostics tab={tab} state={state} />
                ))}
            </section>
          ))}
          <section
            className="layout"
            role="tabpanel"
            id="panel-Overview"
            aria-labelledby="tab-Overview"
            hidden={tab !== "Overview"}
            tabIndex={0}
          >
            <div className="left">
              <div className="card hero">
                <RobotView
                  state={state}
                  dark={dark}
                  visible={tab === "Overview"}
                />
              </div>
            </div>
            <div className="right">
              <OperatorCard state={state}>
                <FaultStatus state={state} fault={fault} />
              </OperatorCard>
            </div>
          </section>
        </main>
      </GamepadStyle.Provider>
    </GamepadTheme.Provider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
