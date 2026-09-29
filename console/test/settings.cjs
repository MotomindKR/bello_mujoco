const assert = require("node:assert/strict");
const test = require("node:test");
const { validateSettings } = require("../settings.cjs");
const config = require("../config.json");
test("settings accept only the three supported controller prompt styles", () => {
  for (const controllerStyle of ["playstation", "xbox", "switch"])
    assert.equal(
      validateSettings({ ...config, controllerStyle }).controllerStyle,
      controllerStyle,
    );
  for (const controllerStyle of [undefined, "auto", "", null, "../xbox"])
    assert.throws(
      () => validateSettings({ ...config, controllerStyle }),
      /Choose PlayStation/,
    );
  assert.throws(
    () => validateSettings({ ...config, hardware: "https://bad" }),
    /HOST:PORT/,
  );
});
test("appearance accepts Light, Dark, or System", () => {
  for (const theme of ["light", "dark", "system"])
    assert.equal(validateSettings({ ...config, theme }).theme, theme);
  for (const theme of [undefined, "auto", "", null])
    assert.throws(() => validateSettings({ ...config, theme }), /Choose Light/);
});
