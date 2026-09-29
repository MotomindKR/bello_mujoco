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
  return {
    connected: true,
    name: device.device.name || "Controller",
    forward: -axis(axes.leftStickY),
    lateral: axis(axes.leftStickX),
    yaw: -axis(axes.rightStickX),
    back: Boolean(buttons.back),
    start: Boolean(buttons.start),
    a: Boolean(buttons.a),
    y: Boolean(buttons.y),
    l1: Boolean(buttons.leftShoulder),
    r1: Boolean(buttons.rightShoulder),
    dpadUp: Boolean(buttons.dpadUp),
    dpadDown: Boolean(buttons.dpadDown),
    dpadLeft: Boolean(buttons.dpadLeft),
    dpadRight: Boolean(buttons.dpadRight),
  };
}

module.exports = { sampleController };
