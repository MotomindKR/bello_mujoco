function emptyTelemetry() {
  return {
    inventory: null,
    motors: {},
    groups: {},
    imus: {},
    health: {},
    powerBoards: {},
    commandPolicyEvents: [],
    imu: null,
    pose: null,
    tracking: null,
    operator: null,
  };
}

// Derive display ages without advancing the source packets on every render.
function displaySnapshot(snapshot, config, now = Date.now()) {
  const state = structuredClone(snapshot);
  state.capturedAt = now;
  for (const service of Object.values(state.services)) {
    service.ageMs = service.last ? now - service.last : null;
    service.fresh =
      service.connected &&
      service.ageMs !== null &&
      service.ageMs < config.freshnessMs;
  }
  if (state.pose) {
    state.pose.sampleAgeMs =
      Number(state.pose.sampleAgeMs) + now - state.services.estimator.last;
    if (state.pose.sampleAgeMs >= config.freshnessMs)
      state.services.estimator.fresh = false;
  }
  if (state.tracking) {
    const age = now - state.services.tracking.last;
    if (state.tracking.packetAgeMs != null) state.tracking.packetAgeMs += age;
    state.services.tracking.fresh &&=
      state.tracking.upstreamConnected &&
      state.tracking.packetAgeMs < config.freshnessMs;
    for (const key of ["body", "leftHand", "rightHand"]) {
      const part = state.tracking[key];
      if (part) {
        part.ageMs += age;
        part.fresh =
          state.services.tracking.fresh && part.ageMs < config.freshnessMs;
      }
    }
  }
  for (const motor of Object.values(state.motors)) {
    motor.feedbackAgeMs = now - motor.receivedAt + Number(motor.feedbackAgeMs);
    motor.fresh = motor.feedbackAgeMs < config.freshnessMs;
  }
  for (const imu of Object.values(state.imus)) {
    imu.sampleAgeMs = now - imu.receivedAt + Number(imu.sampleAgeMs);
    imu.fresh = imu.sampleAgeMs < config.freshnessMs;
  }
  for (const board of Object.values(state.powerBoards)) {
    const timeout = Number(
      state.inventory?.powerBoards?.find(
        (item) => item.name === board.boardName,
      )?.staleTimeoutMs,
    );
    board.ageMs = now - board.receivedAt;
    board.fresh =
      state.services.hardware.connected &&
      Number.isFinite(timeout) &&
      timeout > 0 &&
      board.ageMs < timeout;
    for (const sample of board.samples || []) {
      sample.sampleAgeMs = Number(sample.sampleAgeMs) + board.ageMs;
      sample.stale =
        sample.stale || !board.fresh || sample.sampleAgeMs >= timeout;
    }
  }
  for (const values of [state.groups, state.health]) {
    for (const value of Object.values(values)) {
      value.ageMs = now - value.receivedAt;
      value.fresh = value.ageMs < config.freshnessMs;
    }
  }
  return state;
}

module.exports = { emptyTelemetry, displaySnapshot };
