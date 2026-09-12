//! Checkpoint-defined command transitions; no additional hardware slew filter.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct GaitConfig {
    pub frequency_hz: f32,
    pub speed_dependent: bool,
    pub step_frequency_base: f32,
    pub step_frequency_speed_gain: f32,
    pub stop_extra_steps: i32,
    pub command_slew_enabled: bool,
    pub command_linear_slew_rate: f32,
    pub command_yaw_slew_rate: f32,
    pub directional_start: bool,
    pub lateral_start_deadband: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CommandConfig {
    pub enabled: bool,
    pub minimum_moving_linear_speed: f32,
    pub minimum_moving_yaw_speed: f32,
    pub linear_deadband: f32,
    pub yaw_deadband: f32,
    pub standing_transitions_enabled: bool,
    pub lin_vel_x: [f32; 2],
    pub lin_vel_y: [f32; 2],
    pub ang_vel_yaw: [f32; 2],
}

#[derive(Clone, Debug, Serialize)]
pub struct GaitState {
    pub command: [f32; 3],
    pub phase: f32,
    pub pending: i32,
    next_start_left: bool,
    started: bool,
}

impl Default for GaitState {
    fn default() -> Self {
        Self {
            command: [0.; 3],
            phase: 0.,
            pending: 0,
            next_start_left: true,
            started: false,
        }
    }
}

impl CommandConfig {
    fn standing(&self, command: [f32; 3]) -> bool {
        command[0].abs() < self.linear_deadband
            && command[1].abs() < self.linear_deadband
            && command[2].abs() < self.yaw_deadband
    }
}

impl GaitConfig {
    pub fn validate(&self, command: &CommandConfig, dt: f32) -> Result<()> {
        ensure!(
            [
                self.frequency_hz,
                self.step_frequency_base,
                self.command_linear_slew_rate,
                self.command_yaw_slew_rate,
                dt
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
            "invalid gait rate"
        );
        ensure!(
            [
                self.step_frequency_speed_gain,
                self.lateral_start_deadband,
                command.linear_deadband,
                command.yaw_deadband
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.)
                && (0..i32::MAX).contains(&self.stop_extra_steps),
            "invalid gait transition settings"
        );
        for range in [command.lin_vel_x, command.lin_vel_y, command.ang_vel_yaw] {
            ensure!(
                range.iter().all(|x| x.is_finite()) && range[0] <= 0. && range[1] >= 0.,
                "invalid command range"
            );
        }
        Ok(())
    }
}

impl GaitState {
    pub fn active(&self, config: &GaitConfig, command: &CommandConfig) -> bool {
        self.pending != 0
            || if config.command_slew_enabled {
                self.command.iter().any(|x| *x != 0.)
            } else {
                !command.standing(self.command)
            }
    }

    pub fn features(&self, config: &GaitConfig, command: &CommandConfig) -> [f32; 3] {
        if self.active(config, command) {
            let angle = std::f32::consts::TAU * self.phase;
            [angle.sin(), angle.cos(), 1.]
        } else {
            [0.; 3]
        }
    }

    pub fn advance(
        &mut self,
        request: [f32; 3],
        config: &GaitConfig,
        command: &CommandConfig,
        left_axis: [f32; 2],
        dt: f32,
    ) -> Result<()> {
        ensure!(request.iter().all(|x| x.is_finite()), "nonfinite command");
        let active = self.active(config, command);
        let elapsed = if self.started { dt } else { 0. };
        let frequency = if config.speed_dependent {
            0.5 * (config.step_frequency_base
                + config.step_frequency_speed_gain * self.command[0].hypot(self.command[1]))
        } else {
            config.frequency_hz
        };
        ensure!(frequency * dt < 0.5, "control period skips gait boundaries");
        let mut phase = if !self.started {
            self.phase
        } else if !active {
            0.
        } else if self.pending < 0 {
            self.phase
        } else {
            (self.phase + frequency * dt).rem_euclid(1.)
        };
        let crossed = (self.phase < 0.5 && phase >= 0.5) || phase < self.phase;
        let standing = command.standing(request);
        let lateral = request[0] * left_axis[0] + request[1] * left_axis[1];
        let left = if lateral.abs() > config.lateral_start_deadband {
            lateral > 0.
        } else {
            self.next_start_left
        };
        let start_phase = if config.directional_start && left {
            0.5
        } else {
            0.
        };

        if config.command_slew_enabled {
            let target = if standing { [0.; 3] } else { request };
            let delta = [target[0] - self.command[0], target[1] - self.command[1]];
            let scale = (config.command_linear_slew_rate * elapsed
                / delta[0].hypot(delta[1]).max(1e-8))
            .min(1.);
            let mut next = target;
            if scale < 1. {
                next[0] = self.command[0] + delta[0] * scale;
                next[1] = self.command[1] + delta[1] * scale;
            }
            let yaw_step = config.command_yaw_slew_rate * elapsed;
            if (target[2] - self.command[2]).abs() > yaw_step {
                next[2] =
                    self.command[2] + (target[2] - self.command[2]).clamp(-yaw_step, yaw_step);
            }
            let mut remaining = if self.pending != 0 {
                self.pending
            } else {
                1 + config.stop_extra_steps
            };
            if self.pending > 0 && crossed {
                remaining -= 1;
            }
            if remaining == 0 || !command.standing_transitions_enabled {
                remaining = -1;
            }
            let finished = remaining < 0 && next == [0.; 3];
            self.pending = if standing && active && !finished {
                remaining
            } else {
                0
            };
            if !active && !standing {
                phase = start_phase;
            }
            if self.pending < 0 {
                phase = (phase * 2.).floor() * 0.5;
            }
            if standing && self.pending == 0 && next == [0.; 3] {
                phase = 0.;
            }
            self.command = next;
        } else if !command.standing_transitions_enabled {
            self.command = if standing { [0.; 3] } else { request };
            if standing || !active {
                phase = if standing { 0. } else { start_phase };
            }
            self.pending = 0;
        } else {
            let mut remaining = if self.pending > 0 {
                self.pending
            } else {
                1 + config.stop_extra_steps
            };
            if self.pending > 0 && crossed {
                remaining -= 1;
            }
            let finished = standing && active && remaining == 0;
            let keep_moving = standing && active && !finished;
            if !keep_moving {
                self.command = if standing { [0.; 3] } else { request };
            }
            if finished || !active {
                phase = if standing { 0. } else { start_phase };
            }
            self.pending = if keep_moving { remaining } else { 0 };
        }
        if !active && self.active(config, command) {
            self.next_start_left = !self.next_start_left;
        }
        self.phase = phase;
        self.started = true;
        Ok(())
    }
}
