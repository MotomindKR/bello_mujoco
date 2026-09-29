import React, { useCallback, useEffect, useRef, useState } from "react";

import { GamepadButton } from "./GamepadButton.jsx";

export function AbortButton({ available, sample, progress = 0, idle = false }) {
  const action = idle ? "idle" : "abort";
  const label = `Hold to ${action}`;
  const held = useRef(false);
  const [holding, setHolding] = useState(false);
  const [error, setError] = useState("");
  const cancel = useCallback(() => {
    if (!held.current) return;
    held.current = false;
    setHolding(false);
    window.bello
      .abortHold(false, action)
      .catch(() => setError("Runner unavailable."));
  }, [action]);
  const start = async () => {
    if (held.current || !available) return;
    held.current = true;
    setHolding(true);
    setError("");
    try {
      const result = await window.bello.abortHold(true, action);
      if (result.error) {
        cancel();
        setError(result.error);
      }
    } catch {
      cancel();
      setError("Runner unavailable.");
    }
  };
  useEffect(() => {
    window.addEventListener("blur", cancel);
    return () => {
      window.removeEventListener("blur", cancel);
      cancel();
    };
  }, [cancel]);
  useEffect(() => {
    if (!available) cancel();
  }, [available, cancel]);
  return (
    <button
      type="button"
      className={"abort-button" + (holding ? " holding" : "")}
      aria-label={label}
      disabled={!available}
      onPointerDown={(event) => {
        if (event.button === 0) {
          event.currentTarget.setPointerCapture(event.pointerId);
          start();
        }
      }}
      onPointerMove={(event) => {
        const r = event.currentTarget.getBoundingClientRect();
        if (
          event.clientX < r.left ||
          event.clientX > r.right ||
          event.clientY < r.top ||
          event.clientY > r.bottom
        )
          cancel();
      }}
      onPointerUp={cancel}
      onPointerCancel={cancel}
      onLostPointerCapture={cancel}
      onPointerLeave={cancel}
      onBlur={cancel}
      onKeyDown={(event) => {
        if ([" ", "Enter"].includes(event.key)) {
          event.preventDefault();
          if (!event.repeat) start();
        }
      }}
      onKeyUp={(event) => {
        if ([" ", "Enter"].includes(event.key)) {
          event.preventDefault();
          cancel();
        }
      }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <span
        className="abort-fill"
        style={
          holding
            ? undefined
            : { transform: `scaleX(${Math.min(1, Math.max(0, progress))})` }
        }
        aria-hidden="true"
      />
      <strong>{error || `${label} · 1 s`}</strong>
      <span className="abort-chord">
        <GamepadButton name="L1" sample={sample} progress={progress} /> +{" "}
        <GamepadButton name="R1" sample={sample} progress={progress} />
      </span>
    </button>
  );
}
