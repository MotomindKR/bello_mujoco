import React, { useState } from "react";

export function FaultStatus({ state, fault }) {
  const [clearingFault, setClearingFault] = useState(false);
  const [clearError, setClearError] = useState("");
  const operator = state.operator || {};
  const faultStatusAvailable =
    state.services.runner.fresh || state.services.hardware.fresh;
  return (
    <div
      tabIndex={0}
      role="region"
      aria-label="Fault status"
      className={
        "fault " + (fault ? "active" : faultStatusAvailable ? "" : "pending")
      }
    >
      <div className="fault-details" tabIndex={0}>
        <div className="eyebrow">FAULTS</div>
        <strong>
          {fault ||
            (faultStatusAvailable
              ? "No active fault reported."
              : "Waiting for fault status.")}
        </strong>
        {(clearError || operator.recoveryError) &&
          operator.mode === "MODE_FAULT" && (
            <p role="alert">{clearError || operator.recoveryError}</p>
          )}
      </div>
      {operator.mode === "MODE_FAULT" && (
        <>
          <div className="fault-recovery">
            <button
              className="primary-button clear-fault"
              aria-describedby="fault-clear-hint"
              disabled={
                clearingFault ||
                !state.services.runner.fresh ||
                !operator.canClearFault
              }
              onClick={async () => {
                setClearingFault(true);
                setClearError("");
                try {
                  const result = await window.bello.clearFault();
                  setClearError(result.error || "");
                } catch {
                  setClearError(
                    "Could not reach the runner. Check its connection before retrying.",
                  );
                } finally {
                  setClearingFault(false);
                }
              }}
            >
              {clearingFault || operator.phase === "clearing fault"
                ? "Clearing fault…"
                : "Clear fault"}
            </button>
            <span id="fault-clear-hint" className="diagnostic-note">
              Release inputs, then hold A for 1 s or click Clear fault.
            </span>
          </div>
        </>
      )}
    </div>
  );
}
