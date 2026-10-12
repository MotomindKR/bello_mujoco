export function operatorGuide(state) {
  const operator = state.operator || {};
  const runner = state.services.runner;
  if (!runner.fresh)
    return {
      title: runner.busy ? "Console in use" : "Runner offline",
    };
  if (operator.mode === "MODE_FAULT")
    return operator.canClearFault &&
      state.controller.connected &&
      operator.controllerConnected &&
      !operator.neutralRequired
      ? { title: "Clear fault", keys: ["A"], duration: "1 s · Disarmed" }
      : {
          title:
            operator.phase === "clearing fault"
              ? "Clearing fault"
              : "Control stopped",
        };
  if (operator.mode === "MODE_DISARMED" && operator.fault)
    return { title: "Startup blocked" };
  if (!state.controller.connected) return { title: "Connect a controller" };
  if (!operator.controllerConnected)
    return { title: "Waiting for controller link" };
  if (operator.neutralRequired) return { title: "Release sticks and buttons" };
  switch (operator.mode) {
    case "MODE_DISARMED":
      return operator.phase === "disarmed"
        ? { title: "Ready", keys: ["Back", "Start"], duration: "2 s · Hold" }
        : { title: "Preparing Hold" };
    case "MODE_HOLD":
      if (operator.phase === "bending") return { title: "Bending legs" };
      if (operator.phase === "awaiting lift")
        return operator.startupReady
          ? {
              title: "Confirm balance",
              keys: ["A"],
              duration: "1 s · Balanced",
            }
          : { title: "Lift and balance upright" };
      if (operator.phase === "straightening")
        return { title: "Straighten legs" };
      if (operator.phase === "upright")
        return { title: "Upright", keys: ["A"], duration: "1 s · Stand" };
      if (operator.phase === "engaging ankles")
        return { title: "Restoring ankle control" };
      return operator.phase === "holding"
        ? { title: "Holding", keys: ["A"], duration: "1 s · Stand" }
        : { title: "Ramping to Hold" };
    case "MODE_STAND":
      if (operator.selected === "Barbell")
        return {
          title: "Hands empty; ready to receive bar",
          keys: ["A"],
          duration: "1 s · Barbell",
        };
      if (operator.selected === "Tracking" && !operator.sonicReady)
        return { title: operator.sonicStatus || "Waiting for reference" };
      return {
        title: "Standing",
        keys: ["A"],
        duration: `1 s · ${operator.selected === "Motion" ? operator.selectedMotion || "Motion" : operator.selected || "Locomotion"}`,
      };
    case "MODE_LOCOMOTION":
      return {
        title: "Walking",
        keys: ["A"],
        duration: "1 s · Stand",
      };
    case "MODE_SIT": {
      const phase = operator.phase;
      if (phase === "awaiting support")
        return {
          title: "Support robot",
          keys: ["A"],
          duration: "1 s · Supported",
        };
      if (phase === "settling to hold") return { title: "Settling to Hold" };
      if (phase === "releasing ankles")
        return { title: "Releasing ankle control" };
      if (phase === "upright")
        return {
          title: "Upright",
          ...(operator.startupReady
            ? { keys: ["A"], duration: "1 s · Stand" }
            : {}),
        };
      if (phase === "crouching") return { title: "Move legs" };
      if (phase === "bent")
        return {
          title: "Bent",
          ...(operator.startupReady
            ? { keys: ["A"], duration: "1 s · Lower onto back" }
            : {}),
        };
      if (phase === "lowering onto back")
        return {
          title: "Lower onto back",
          keys: ["A"],
          duration: "1 s · Supported on floor",
        };
      if (phase === "engaging ankles")
        return { title: "Restoring ankle control" };
      return { title: "Unloading motors" };
    }
    case "MODE_TRACKING":
      if (operator.phase === "returning to neutral")
        return { title: "Returning to neutral" };
      if (
        operator.phase?.startsWith("returning") ||
        operator.phase === "transition_to_standing"
      )
        return { title: "Returning to Stand" };
      return {
        title: "Body tracking",
        keys: ["A"],
        duration: "1 s · Stand",
      };
    case "MODE_BARBELL": {
      const handoff = operator.barbell;
      const prompts = {
        bar_ready: "Bar positioned in both hands",
        grasp_secured: "Confirm secure grip; prepare load transfer",
        bar_supported: "Confirm holder supports the bar",
        hands_open: "Confirm both hands are open",
        bar_clear: "Confirm bar is clear of both hands",
      };
      if (handoff?.gripAdjustable && !handoff.confirmationReady)
        return { title: "Set grip with left stick; center to confirm" };
      if (handoff?.awaiting && handoff.confirmationReady)
        return {
          title: prompts[handoff.awaiting] || "Waiting for handoff",
          keys: ["A"],
          duration: "1 s · Confirm",
        };
      return {
        title: handoff?.awaiting
          ? "Waiting for fresh hand feedback"
          : "Barbell sequence",
      };
    }
    case "MODE_MOTION":
      return { title: operator.activeMotion || "Motion" };
    default:
      return { title: "Waiting for status" };
  }
}
