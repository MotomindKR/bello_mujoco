export function motorTemperature(state) {
  const names = [
    ...new Set([
      ...(state.inventory?.motors || []).map((motor) => motor.name),
      ...Object.keys(state.motors || {}),
    ]),
  ];
  let hottest;
  let incomplete = !state.services.hardware.fresh || names.length === 0;
  for (const name of names) {
    const motor = state.motors?.[name];
    if (!motor?.fresh || !Number.isFinite(motor.temperature)) {
      incomplete = true;
      continue;
    }
    if (!hottest || motor.temperature > hottest.temperature) hottest = motor;
  }
  return {
    hottest,
    hot: Boolean(hottest && hottest.temperature > 60),
    incomplete,
  };
}
