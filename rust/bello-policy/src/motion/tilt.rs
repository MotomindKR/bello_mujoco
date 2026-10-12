//! Direction-sensitive, yaw-independent tilt deviation for live motion.

use anyhow::{Result, ensure};
use glam::{DQuat, DVec3};

use super::rotation::checked_quaternion;

#[derive(Clone, Copy)]
pub(crate) struct Envelope {
    pub gravity: DVec3,
    pub max_error: f64,
}

impl Envelope {
    pub fn new(orientation: DQuat, max_error: f64) -> Self {
        Self {
            gravity: orientation.conjugate() * DVec3::NEG_Z,
            max_error,
        }
    }

    pub fn blend(self, destination: Self, fraction: f64) -> Self {
        let rotation = DQuat::from_rotation_arc(self.gravity, destination.gravity);
        Self {
            gravity: DQuat::IDENTITY.slerp(rotation, fraction) * self.gravity,
            max_error: self.max_error + fraction * (destination.max_error - self.max_error),
        }
    }

    pub fn error(self, orientation: DQuat) -> Result<f64> {
        let gravity = checked_quaternion(orientation)?.conjugate() * DVec3::NEG_Z;
        Ok(self.gravity.angle_between(gravity))
    }

    pub fn check(self, orientation: DQuat) -> Result<f64> {
        let error = self.error(orientation)?;
        ensure!(
            error <= self.max_error,
            "motion tilt deviation {error:.4} rad exceeds reference limit {:.4} rad",
            self.max_error
        );
        Ok(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_reference_tilt_is_allowed_but_wrong_direction_is_not() {
        let reference = DQuat::from_rotation_y(1.4);
        let guard = Envelope::new(reference, 0.3);
        assert!(guard.check(reference).unwrap() < 1e-7);
        assert!(guard.check(DQuat::from_rotation_y(-1.4)).is_err());
        assert!(guard.check(DQuat::from_rotation_x(1.4)).is_err());
        assert!(guard.check(DQuat::from_rotation_y(1.6)).is_ok());
        assert!(guard.check(DQuat::from_rotation_y(1.8)).is_err());
    }

    #[test]
    fn world_yaw_quaternion_sign_and_fixed_imu_mount_do_not_change_error() {
        let reference = DQuat::from_rotation_y(1.3);
        let actual = DQuat::from_rotation_y(1.4);
        let mount = DQuat::from_rotation_x(0.2) * DQuat::from_rotation_z(-1.5);
        let guard = Envelope::new(reference * mount, 0.3);
        let actual = DQuat::from_rotation_z(2.0) * actual * mount;
        assert!((guard.check(actual).unwrap() - 0.1).abs() < 1e-10);
        assert!((guard.check(-actual).unwrap() - 0.1).abs() < 1e-10);
        assert!(guard.check(DQuat::from_xyzw(f64::NAN, 0., 0., 1.)).is_err());
    }

    #[test]
    fn transition_reference_and_tolerance_are_continuous_in_both_directions() {
        let standing = Envelope::new(DQuat::IDENTITY, 0.7);
        let leaning = Envelope::new(DQuat::from_rotation_y(1.4), 0.3);
        for step in 0..=10 {
            let fraction = step as f64 / 10.;
            let outbound = standing.blend(leaning, fraction);
            let inbound = leaning.blend(standing, 1. - fraction);
            assert!(outbound.gravity.distance(inbound.gravity) < 1e-10);
            assert!((outbound.max_error - inbound.max_error).abs() < 1e-10);
            assert!(
                outbound
                    .check(DQuat::from_rotation_y(1.4 * fraction))
                    .is_ok()
            );
        }
        assert!(
            leaning
                .blend(standing, 0.)
                .check(DQuat::from_rotation_y(1.4))
                .is_ok()
        );
        assert!(
            leaning
                .blend(standing, 1.)
                .check(DQuat::from_rotation_y(1.4))
                .is_err()
        );
        assert!(leaning.blend(standing, 1.).check(DQuat::IDENTITY).is_ok());
    }
}
