import React, { useEffect, useRef } from "react";
import { DirectionHint } from "./DirectionHint.jsx";

export function MotionPicker({ operator, sample, choosing, onSelect }) {
  const selected = useRef(null);
  useEffect(() => {
    const item = selected.current;
    if (item) {
      const list = item.parentElement;
      if (item.offsetLeft < list.scrollLeft) list.scrollLeft = item.offsetLeft;
      if (
        item.offsetLeft + item.offsetWidth >
        list.scrollLeft + list.clientWidth
      )
        list.scrollLeft = item.offsetLeft + item.offsetWidth - list.clientWidth;
    }
  }, [operator.selectedMotion]);
  return (
    <div className="motion-picker">
      <div className="motion-list-heading">
        <span>Motions</span>
        <span className="motion-picker-keys">
          <DirectionHint direction="Up" sample={sample} />
          <DirectionHint direction="Down" sample={sample} />
        </span>
      </div>
      <ul className="motion-choices" aria-label="Motions · use D-pad up/down">
        {(operator.motions || []).map((name) => (
          <li
            key={name}
            ref={name === operator.selectedMotion ? selected : null}
            aria-current={name === operator.selectedMotion ? "true" : undefined}
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
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
