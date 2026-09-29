import React, { createContext, useContext } from "react";

export const GamepadStyle = createContext("xbox");
export const GamepadTheme = createContext(false);
const images = import.meta.glob("./assets/input-prompts/*/*/*.png", {
  eager: true,
  query: "?url",
  import: "default",
});
const fields = {
  Back: "back",
  Start: "start",
  A: "a",
  Y: "y",
  L1: "l1",
  R1: "r1",
  Left: "dpadLeft",
  Right: "dpadRight",
  Up: "dpadUp",
  Down: "dpadDown",
};
const labels = {
  playstation: {
    A: "Cross",
    Y: "Triangle",
    Back: "Create / Share",
    Start: "Options",
    L1: "L1",
    R1: "R1",
  },
  xbox: { A: "A", Y: "Y", Back: "View", Start: "Menu", L1: "LB", R1: "RB" },
  switch: { A: "A", Y: "Y", Back: "Minus", Start: "Plus", L1: "L", R1: "R" },
};
export function GamepadButton({ name, sample = {}, progress = 0 }) {
  const style = useContext(GamepadStyle);
  const dark = useContext(GamepadTheme);
  const label = labels[style][name] || `D-pad ${name.toLowerCase()}`;
  const asset = (variant) =>
    images[`./assets/input-prompts/${style}/${variant}/${name}.png`];
  return (
    <kbd
      className={`gamepad-button${sample.connected && sample[fields[name]] ? " pressed" : ""}`}
      title={label}
      aria-label={label}
      data-controller-style={style}
    >
      <span
        className="gamepad-hold-fill"
        style={{ transform: `scaleX(${Math.min(1, Math.max(0, progress))})` }}
        aria-hidden="true"
      />
      <picture aria-hidden="true">
        <img src={asset(dark ? "light" : "default")} alt="" draggable="false" />
      </picture>
    </kbd>
  );
}
