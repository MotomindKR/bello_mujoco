export function operatorGuide(state) {
  const operator = state.operator || {};
  const runner = state.services.runner;
  if (!runner.fresh)
    return {
      title: runner.busy ? "Console in use" : "Runner offline",
    };
  if (operator.mode === "MODE_FAULT")
    return {
      title:
        operator.phase === "clearing fault"
          ? "Clearing fault"
          : "Control stopped",
    };
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
        return { title: "Left stick up · Straighten" };
      if (operator.phase === "upright")
        return { title: "Upright", keys: ["A"], duration: "1 s · Stand" };
      if (operator.phase === "engaging ankles")
        return { title: "Restoring ankle control" };
      return operator.phase === "holding"
        ? { title: "Holding", keys: ["A"], duration: "1 s · Stand" }
        : { title: "Ramping to Hold" };
    case "MODE_STAND":
      if (operator.selected === "Tracking" && !operator.sonicReady)
        return { title: operator.sonicStatus || "Waiting for reference" };
      return operator.selected === "Motion" && operator.motionMenuOpen
        ? { title: "Choose motion" }
        : {
            title: "Standing",
            keys: ["A"],
            duration: `1 s · ${operator.selected === "Motion" ? operator.selectedMotion || "Motion" : operator.selected || "Stand"}`,
          };
    case "MODE_LOCOMOTION":
      return {
        title: "Walking",
        keys: ["A"],
        duration: "1 s · Stand",
      };
    case "MODE_TRACKING":
      return {
        title: operator.phase?.startsWith("returning")
          ? "Returning to Stand"
          : "Body tracking",
        keys: ["A"],
        duration: "1 s · Stand",
      };
    case "MODE_MOTION":
      return { title: operator.activeMotion || "Motion" };
    default:
      return { title: "Waiting for status" };
  }
}
