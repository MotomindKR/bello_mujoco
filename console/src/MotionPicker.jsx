import React, { useEffect, useRef } from "react";
import { DirectionHint } from "./DirectionHint.jsx";
import { GamepadButton } from "./GamepadButton.jsx";

export function MotionPicker({
  operator,
  sample,
  choosing,
  onSelect,
  onToggle,
}) {
  const selected = useRef(null);
  const active = operator.selected === "Motion";
  const open = active && choosing && operator.motionMenuOpen;
  useEffect(() => {
    const item = selected.current;
    if (open && item) {
      const list = item.parentElement;
      if (item.offsetTop < list.scrollTop) list.scrollTop = item.offsetTop;
      if (
        item.offsetTop + item.offsetHeight >
        list.scrollTop + list.clientHeight
      )
        list.scrollTop = item.offsetTop + item.offsetHeight - list.clientHeight;
    }
  }, [open, operator.selectedMotion]);
  return (
    <div className={`motion-picker${active ? " active" : ""}`}>
      <button
        type="button"
        className="motion-picker-trigger"
        disabled={!choosing || !active || !operator.motions?.length}
        aria-expanded={!!open}
        aria-controls="motion-choices"
        onClick={() => onToggle(!open)}
      >
        <span title={operator.selectedMotion}>
          {operator.selectedMotion || "No motions loaded"}
        </span>
        <span className="motion-picker-keys">
          <DirectionHint direction="Up" sample={sample} />
          <DirectionHint direction="Down" sample={sample} />
        </span>
      </button>
      {open && (
        <ul
          id="motion-choices"
          className="motion-choices"
          aria-label="Motion choices · use D-pad up/down"
        >
          {(operator.motions || []).map((name) => (
            <li
              key={name}
              ref={name === operator.selectedMotion ? selected : null}
              aria-current={
                name === operator.selectedMotion ? "true" : undefined
              }
              title={name}
            >
              <button
                type="button"
                disabled={!choosing}
                aria-label={name}
                aria-pressed={name === operator.selectedMotion}
                onClick={() => onSelect(name)}
              >
                <span className="motion-choice-name">{name}</span>
                {name === operator.selectedMotion && (
                  <span className="motion-select-hint">
                    <GamepadButton name="A" sample={sample} />
                    Select
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
