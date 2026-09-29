export function jointPositions(coordinates, transmissions) {
  const positions = { ...coordinates };
  for (const transmission of transmissions) {
    const [first, second] = transmission.actuators;
    if (!(first.name in coordinates) || !(second.name in coordinates)) continue;
    const a = coordinates[first.name] / first.reduction;
    const b = coordinates[second.name] / second.reduction;
    const [sum, difference] = transmission.joints;
    positions[sum.name] = (a + b) / (2 * sum.reduction) + sum.offset;
    positions[difference.name] =
      (a - b) / (2 * difference.reduction) + difference.offset;
  }
  return positions;
}

export function parseTransmissions(document) {
  return [...document.querySelectorAll("transmission")].map((transmission) => {
    if (
      transmission.querySelector("type")?.textContent !==
      "transmission_interface/DifferentialTransmission"
    ) {
      throw new Error("Unsupported visual transmission");
    }
    const entries = (tag, role) =>
      [1, 2].map((index) => {
        const element = transmission.querySelector(
          `${tag}[role="${role}${index}"]`,
        );
        if (!element) throw new Error("Incomplete differential transmission");
        const reduction = Number(
          element.querySelector("mechanical_reduction")?.textContent || 1,
        );
        const offset = Number(
          element.querySelector("offset")?.textContent || 0,
        );
        if (
          !Number.isFinite(reduction) ||
          reduction === 0 ||
          !Number.isFinite(offset)
        )
          throw new Error("Invalid differential transmission");
        return { name: element.getAttribute("name"), reduction, offset };
      });
    return {
      actuators: entries("actuator", "actuator"),
      joints: entries("joint", "joint"),
    };
  });
}
