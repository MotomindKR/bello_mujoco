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
      return operator.phase === "holding"
        ? { title: "Holding", keys: ["A"], duration: "1 s · Stand" }
        : { title: "Ramping to Hold" };
    case "MODE_STAND":
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
    case "MODE_MOTION":
      return { title: operator.activeMotion || "Motion" };
    default:
      return { title: "Waiting for status" };
  }
}
