export function number(value, digits = 3) {
  if (value == null || value === "" || !Number.isFinite(Number(value)))
    return "--";
  return String(Number(Number(value).toFixed(digits)));
}

export function scalar(value) {
  if (typeof value === "number") return number(value);
  return value == null ? "--" : value === "" ? '""' : String(value);
}
