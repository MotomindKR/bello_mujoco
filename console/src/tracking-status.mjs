export function trackingStatus(service, tracking) {
  if (service.error) return { tone: "offline", label: "Error" };
  if (!service.connected) return { tone: "offline", label: "Waiting" };
  const parts = [tracking?.body, tracking?.leftHand, tracking?.rightHand];
  if (service.fresh && parts.some((part) => part?.fresh))
    return { tone: "online", label: "Live" };
  return {
    tone: "pending",
    label: parts.some(Boolean) ? "Stale" : "Waiting",
  };
}
