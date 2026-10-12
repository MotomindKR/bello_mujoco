//! Confirmation-gated barbell lifecycle. A stop never authorizes opening hands.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub initial_stand_seconds: f64,
    pub reach_seconds: f64,
    pub load_transfer_seconds: f64,
    pub policy_blend_seconds: f64,
    pub motion_seconds: f64,
    pub retract_seconds: f64,
    pub final_stand_seconds: f64,
}
impl Timing {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.initial_stand_seconds,
                self.reach_seconds,
                self.load_transfer_seconds,
                self.policy_blend_seconds,
                self.motion_seconds,
                self.retract_seconds,
                self.final_stand_seconds
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0. && *v < 86400.),
            "invalid barbell timing"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Stand,
    Reach,
    Receive,
    Close,
    Load,
    EnterMotion,
    Motion,
    Return,
    Support,
    Unload,
    Open,
    Clear,
    Retract,
    Finish,
    Complete,
    Interrupted,
}
impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Stand => "normal_standing",
            Self::Reach => "hands_forward",
            Self::Receive => "await_bar",
            Self::Close => "close_hands",
            Self::Load => "transfer_load",
            Self::EnterMotion => "to_motion",
            Self::Motion => "motion",
            Self::Return => "to_standing",
            Self::Support => "await_support",
            Self::Unload => "transfer_to_holder",
            Self::Open => "open_hands",
            Self::Clear => "await_bar_clear",
            Self::Retract => "hands_normal",
            Self::Finish => "normal_standing_final",
            Self::Complete => "complete",
            Self::Interrupted => "interrupted",
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Command {
    pub phase: Phase,
    pub phase_id: u64,
    pub elapsed: f64,
    pub progress: f64,
    pub arm: f64,
    pub load: f64,
    pub weight: f64,
    pub motion_time: f64,
    pub closure: f64,
    pub reason: String,
}
pub struct Sequence {
    pub command: Command,
    timing: Timing,
    entry: [f64; 3],
    possibly_grasped: bool,
    interrupted: bool,
}
impl Sequence {
    pub fn new(timing: Timing) -> Self {
        Self {
            timing,
            entry: [0.; 3],
            possibly_grasped: false,
            interrupted: false,
            command: Command {
                phase: Phase::Stand,
                phase_id: 0,
                elapsed: 0.,
                progress: 0.,
                arm: 0.,
                load: 0.,
                weight: 0.,
                motion_time: 0.,
                closure: 0.,
                reason: String::new(),
            },
        }
    }
    fn enter(&mut self, phase: Phase) {
        self.command.phase = phase;
        self.command.phase_id += 1;
        self.command.elapsed = 0.;
        self.command.progress = 0.;
        self.entry = [self.command.arm, self.command.load, self.command.weight];
        if phase == Phase::Close {
            self.possibly_grasped = true;
        }
        if phase == Phase::Open {
            self.command.closure = 0.;
        }
    }
    pub fn adjust_grip(&mut self, delta: f64) {
        if self.command.phase == Phase::Close {
            self.command.closure = (self.command.closure + delta).clamp(0., 1.);
        }
    }
    pub fn awaiting(&self) -> &'static str {
        match self.command.phase {
            Phase::Receive => "bar_ready",
            Phase::Close => "grasp_secured",
            Phase::Support => "bar_supported",
            Phase::Open => "hands_open",
            Phase::Clear => "bar_clear",
            Phase::Interrupted if self.possibly_grasped => "bar_supported",
            _ => "",
        }
    }
    pub fn confirm(&mut self, event: &str, generation: u64) -> Result<()> {
        ensure!(
            !event.is_empty() && event == self.awaiting() && generation == self.command.phase_id,
            "stale or invalid barbell confirmation"
        );
        ensure!(
            event != "grasp_secured" || self.command.closure > 0.,
            "set barbell grip before confirming grasp"
        );
        let next = match event {
            "bar_ready" => Phase::Close,
            "grasp_secured" => Phase::Load,
            "bar_supported" => Phase::Unload,
            "hands_open" => {
                self.possibly_grasped = false;
                Phase::Clear
            }
            "bar_clear" => Phase::Retract,
            _ => unreachable!(),
        };
        self.enter(next);
        Ok(())
    }
    pub fn stop(&mut self, reason: &str) {
        if self.interrupted || self.command.phase == Phase::Complete {
            return;
        }
        self.interrupted = true;
        self.command.reason = reason.into();
        let next = if self.command.weight > 0. {
            Phase::Return
        } else if self.possibly_grasped {
            Phase::Interrupted
        } else if matches!(
            self.command.phase,
            Phase::Receive | Phase::Open | Phase::Clear
        ) {
            Phase::Clear
        } else {
            Phase::Retract
        };
        self.enter(next);
    }
    pub fn advance(&mut self, dt: f64) -> Command {
        let phase = self.command.phase;
        let duration = match phase {
            Phase::Stand => self.timing.initial_stand_seconds,
            Phase::Reach => self.timing.reach_seconds,
            Phase::Load | Phase::Unload => self.timing.load_transfer_seconds,
            Phase::EnterMotion | Phase::Return => self.timing.policy_blend_seconds,
            Phase::Motion => self.timing.motion_seconds,
            Phase::Retract => self.timing.retract_seconds,
            Phase::Finish => self.timing.final_stand_seconds,
            _ => return self.command.clone(),
        };
        self.command.elapsed += dt;
        let t = (self.command.elapsed / duration).min(1.);
        let smooth = t * t * t * (10. + t * (-15. + 6. * t));
        self.command.progress = t;
        match phase {
            Phase::Reach => self.command.arm = smooth,
            Phase::Load => self.command.load = smooth,
            Phase::EnterMotion => self.command.weight = t,
            Phase::Motion => self.command.motion_time = self.command.elapsed.min(duration),
            Phase::Return => self.command.weight = self.entry[2] * (1. - t),
            Phase::Unload => self.command.load = self.entry[1] * (1. - smooth),
            Phase::Retract => self.command.arm = self.entry[0] * (1. - smooth),
            _ => {}
        }
        let output = self.command.clone();
        if self.command.elapsed + 1e-9 >= duration {
            self.enter(match phase {
                Phase::Stand => Phase::Reach,
                Phase::Reach => Phase::Receive,
                Phase::Load => Phase::EnterMotion,
                Phase::EnterMotion => Phase::Motion,
                Phase::Motion => Phase::Return,
                Phase::Return => Phase::Support,
                Phase::Unload => Phase::Open,
                Phase::Retract => Phase::Finish,
                Phase::Finish => Phase::Complete,
                _ => unreachable!(),
            });
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sequence() -> Sequence {
        Sequence::new(Timing {
            initial_stand_seconds: 1.,
            reach_seconds: 1.,
            load_transfer_seconds: 1.,
            policy_blend_seconds: 1.,
            motion_seconds: 1.,
            retract_seconds: 1.,
            final_stand_seconds: 1.,
        })
    }
    fn confirm(s: &mut Sequence, event: &str) {
        s.confirm(event, s.command.phase_id).unwrap();
    }
    #[test]
    fn grip_is_incremental_bounded_and_latched_after_confirmation() {
        let mut s = sequence();
        s.adjust_grip(1.);
        assert_eq!(s.command.closure, 0.);
        s.advance(1.);
        s.advance(1.);
        confirm(&mut s, "bar_ready");
        assert_eq!(s.command.closure, 0.);
        assert!(s.confirm("grasp_secured", s.command.phase_id).is_err());
        s.adjust_grip(0.25);
        s.adjust_grip(0.25);
        assert_eq!(s.command.closure, 0.5);
        s.adjust_grip(2.);
        assert_eq!(s.command.closure, 1.);
        s.adjust_grip(-2.);
        assert_eq!(s.command.closure, 0.);
        s.adjust_grip(0.65);
        s.adjust_grip(0.);
        assert_eq!(s.command.closure, 0.65);
        confirm(&mut s, "grasp_secured");
        for _ in 0..4 {
            s.adjust_grip(-1.);
            s.advance(1.);
            assert_eq!(s.command.closure, 0.65);
        }
        confirm(&mut s, "bar_supported");
        s.advance(0.5);
        assert_eq!(s.command.closure, 0.65);
        s.advance(0.5);
        assert_eq!(s.command.closure, 0.);
        s.adjust_grip(1.);
        assert_eq!(s.command.closure, 0.);
    }
    #[test]
    fn interrupted_grab_keeps_selected_grip_and_ignores_stick() {
        let mut s = sequence();
        s.advance(1.);
        s.advance(1.);
        confirm(&mut s, "bar_ready");
        s.adjust_grip(0.4);
        s.stop("controller lost");
        s.adjust_grip(-1.);
        s.advance(100.);
        assert_eq!(s.command.closure, 0.4);
        assert_eq!(s.awaiting(), "bar_supported");
    }
    #[test]
    fn cycle_requires_each_fresh_confirmation() {
        let mut s = sequence();
        s.advance(1.);
        s.advance(1.);
        for _ in 0..100 {
            s.advance(1.);
        }
        assert_eq!(s.awaiting(), "bar_ready");
        let old = s.command.phase_id;
        confirm(&mut s, "bar_ready");
        assert!(s.confirm("grasp_secured", old).is_err());
        s.adjust_grip(0.65);
        confirm(&mut s, "grasp_secured");
        s.advance(1.);
        s.advance(1.);
        s.advance(1.);
        s.advance(1.);
        assert_eq!(s.awaiting(), "bar_supported");
        assert_eq!(s.command.closure, 0.65);
        confirm(&mut s, "bar_supported");
        s.advance(1.);
        assert_eq!(s.command.closure, 0.);
        confirm(&mut s, "hands_open");
        confirm(&mut s, "bar_clear");
        s.advance(1.);
        s.advance(1.);
        assert_eq!(s.command.phase, Phase::Complete);
    }
    #[test]
    fn stop_with_load_never_opens_or_retracts_without_support() {
        let mut s = sequence();
        s.advance(1.);
        s.advance(1.);
        confirm(&mut s, "bar_ready");
        s.adjust_grip(0.65);
        confirm(&mut s, "grasp_secured");
        s.advance(1.);
        s.advance(0.5);
        s.stop("lost controller");
        assert_eq!(s.command.phase, Phase::Return);
        s.advance(1.);
        for _ in 0..100 {
            s.advance(1.);
        }
        assert_eq!(s.command.closure, 0.65);
        assert_eq!(s.command.load, 1.);
        assert_eq!(s.command.arm, 1.);
        assert_eq!(s.awaiting(), "bar_supported");
        confirm(&mut s, "bar_supported");
        s.advance(1.);
        assert_eq!(s.command.closure, 0.);
    }
    #[test]
    fn cancel_mid_load_preserves_preload_until_support() {
        let mut s = sequence();
        s.advance(1.);
        s.advance(1.);
        confirm(&mut s, "bar_ready");
        s.adjust_grip(0.65);
        confirm(&mut s, "grasp_secured");
        s.advance(0.5);
        s.stop("cancel");
        s.advance(100.);
        assert_eq!(s.command.load, 0.5);
        assert_eq!(s.command.closure, 0.65);
        confirm(&mut s, "bar_supported");
        s.advance(0.5);
        assert_eq!(s.command.load, 0.25);
    }
}
