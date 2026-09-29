const { isIP } = require("node:net");

const services = ["hardware", "estimator", "runner", "tracking"];
function validateConnections(value) {
  if (!value || typeof value !== "object")
    throw new Error("Enter an address for each service.");
  return Object.fromEntries(
    services.map((service) => {
      const address = value[service];
      const match =
        typeof address === "string" &&
        /^(\[[^\]]+\]|[^:]+):(\d+)$/.exec(address.trim());
      if (!match) throw new Error(`Enter ${service} as HOST:PORT.`);
      const [, host, port] = match;
      const validHost = host.startsWith("[")
        ? isIP(host.slice(1, -1)) === 6
        : isIP(host) === 4 ||
          (!/^[\d.]+$/.test(host) &&
            host.length <= 253 &&
            host
              .split(".")
              .every((label) =>
                /^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/.test(label),
              ));
      if (!validHost || Number(port) < 1 || Number(port) > 65535)
        throw new Error(`Enter a valid ${service} host and port (1-65535).`);
      return [service, `${host}:${Number(port)}`];
    }),
  );
}
function validateSettings(value) {
  const endpoints = validateConnections(value);
  if (!["playstation", "xbox", "switch"].includes(value.controllerStyle))
    throw new Error("Choose PlayStation, Xbox, or Switch button prompts.");
  if (!["light", "dark", "system"].includes(value.theme))
    throw new Error("Choose Light, Dark, or System appearance.");
  return {
    ...endpoints,
    controllerStyle: value.controllerStyle,
    theme: value.theme,
  };
}
module.exports = { validateConnections, validateSettings };
