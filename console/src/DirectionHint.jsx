import React from "react";
import {
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
} from "lucide-react";

const icons = {
  Left: ChevronLeft,
  Right: ChevronRight,
  Up: ChevronUp,
  Down: ChevronDown,
};

export function DirectionHint({ direction, sample }) {
  const Icon = icons[direction];
  const pressed = sample.connected && sample[`dpad${direction}`];
  return (
    <span
      className={`direction-hint${pressed ? " pressed" : ""}`}
      title={`D-pad ${direction.toLowerCase()}`}
      aria-label={`D-pad ${direction.toLowerCase()}`}
    >
      <Icon size={16} aria-hidden="true" />
    </span>
  );
}
