function sampleController(device, deadzone) {
  const disconnected = {
    connected: false,
    name: "",
    forward: 0,
    lateral: 0,
    yaw: 0,
    back: false,
    start: false,
    a: false,
    y: false,
    l1: false,
    r1: false,
    l2: false,
    r2: false,
    dpadUp: false,
    dpadDown: false,
    dpadLeft: false,
    dpadRight: false,
  };
  if (!device || device.closed) return disconnected;
  // SDL getters poll events synchronously and may close this device. Keep the
  // instance for this sample and discard all input if either poll closes it.
  const axes = { ...device.axes };
  if (device.closed) return disconnected;
  const buttons = { ...device.buttons };
  if (device.closed) return disconnected;
  const axis = (value) => {
    if (!Number.isFinite(value) || Math.abs(value) < deadzone) return 0;
    return (
      (Math.sign(value) * (Math.min(1, Math.abs(value)) - deadzone)) /
      (1 - deadzone)
    );
  };
  const triggerPressed = (value) => Number.isFinite(value) && value > 0.5;
  return {
    connected: true,
    name: device.device.name || "Controller",
    forward: -axis(axes.leftStickY),
    lateral: axis(axes.leftStickX),
    yaw: -axis(axes.rightStickX),
    back: Boolean(buttons.back),
    start: Boolean(buttons.start),
    a: Boolean(buttons.south),
    y: Boolean(buttons.north),
    l1: Boolean(buttons.leftShoulder),
    r1: Boolean(buttons.rightShoulder),
    l2: triggerPressed(axes.leftTrigger),
    r2: triggerPressed(axes.rightTrigger),
    dpadUp: Boolean(buttons.dpadUp),
    dpadDown: Boolean(buttons.dpadDown),
    dpadLeft: Boolean(buttons.dpadLeft),
    dpadRight: Boolean(buttons.dpadRight),
  };
}

function createTabNavigator() {
  let previousLeft = false;
  let previousRight = false;
  return (sample) => {
    const left = sample.connected && sample.l1;
    const right = sample.connected && sample.r1;
    const leftClick = left && !previousLeft;
    const rightClick = right && !previousRight;
    previousLeft = left;
    previousRight = right;
    const direction = leftClick === rightClick ? 0 : leftClick ? -1 : 1;
    return direction;
  };
}

function createOperatorMapper() {
  let previousLeft = false;
  let previousRight = false;
  let previousUp = false;
  let previousDown = false;
  let previousTracking = false;
  return (sample, status, onScreenAbort = false) => {
    const axisNeutral = [sample.forward, sample.lateral, sample.yaw].every(
      (axis) => Math.abs(axis) <= 0.05,
    );
    const otherButtons = [
      sample.back,
      sample.start,
      sample.a,
      sample.y,
      sample.l1,
      sample.r1,
      sample.dpadUp,
      sample.dpadDown,
      sample.dpadLeft,
      sample.dpadRight,
    ].some(Boolean);
    const abortOtherControlsNeutral = axisNeutral && !otherButtons;
    const controlsNeutral =
      abortOtherControlsNeutral && !sample.l2 && !sample.r2;
    const abort = onScreenAbort || (sample.connected && sample.l2 && sample.r2);
    const confirm =
      sample.connected &&
      sample.a &&
      !sample.back &&
      !sample.start &&
      !sample.y &&
      !sample.l1 &&
      !sample.r1 &&
      !sample.l2 &&
      !sample.r2;
    const hold =
      sample.connected &&
      sample.back &&
      sample.start &&
      !sample.a &&
      !sample.y &&
      !sample.l1 &&
      !sample.r1 &&
      !sample.l2 &&
      !sample.r2;
    const left = sample.connected && sample.dpadLeft;
    const right = sample.connected && sample.dpadRight;
    const up = sample.connected && sample.dpadUp;
    const down = sample.connected && sample.dpadDown;
    const modeStep =
      right && !left && !previousRight
        ? 1
        : left && !right && !previousLeft
          ? -1
          : 0;
    const motionStep =
      down && !up && !previousDown ? 1 : up && !down && !previousUp ? -1 : 0;
    const tracking = sample.connected && sample.y;
    const toggleTracking = tracking && !previousTracking;
    previousLeft = left;
    previousRight = right;
    previousUp = up;
    previousDown = down;
    previousTracking = tracking;
    const mode = status?.mode;
    return {
      connected: sample.connected,
      name: sample.name,
      forward: sample.forward,
      lateral: sample.lateral,
      yaw: sample.yaw,
      controlsNeutral,
      abortOtherControlsNeutral,
      requestHold: !abort && hold,
      confirmStartup: !abort && confirm && mode === "MODE_HOLD",
      confirmFaultRecovery: !abort && confirm && mode === "MODE_FAULT",
      confirmSit: !abort && confirm && mode === "MODE_SIT",
      requestMode:
        !abort && confirm
          ? mode === "MODE_STAND"
            ? status.selected
            : ["MODE_LOCOMOTION", "MODE_TRACKING"].includes(mode)
              ? "Stand"
              : ""
          : "",
      abort,
      modeStep,
      motionStep,
      navigating: left || right || up || down,
      toggleTracking: mode !== "MODE_BARBELL" && toggleTracking,
      barbellConfirmHeld: !abort && confirm && mode === "MODE_BARBELL",
      confirmBarbell:
        !abort &&
        confirm &&
        mode === "MODE_BARBELL" &&
        status.barbell?.confirmationReady
          ? status.barbell.awaiting || ""
          : "",
      barbellPhaseId: status?.barbell?.phaseId || "0",
      requestBarbellStop:
        !abort &&
        mode === "MODE_BARBELL" &&
        tracking &&
        axisNeutral &&
        !sample.a &&
        !sample.back &&
        !sample.start &&
        !sample.l1 &&
        !sample.r1 &&
        !sample.l2 &&
        !sample.r2,
    };
  };
}

module.exports = { sampleController, createTabNavigator, createOperatorMapper };
