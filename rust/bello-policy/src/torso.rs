//! Checkpoint-owned torso angle bounds and slew, independent of walking intent.
use anyhow::{Result, ensure};
use glam::Vec3;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub roll: [f32; 2],
    pub pitch: [f32; 2],
    pub yaw: [f32; 2],
    pub slew_rate: [f32; 3],
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let half_pi = std::f32::consts::FRAC_PI_2;
        ensure!(
            [self.roll, self.pitch, self.yaw]
                .iter()
                .all(|r| r.iter().all(|x| x.is_finite())
                    && -half_pi < r[0]
                    && r[0] <= 0.
                    && 0. <= r[1]
                    && r[1] < half_pi)
                && self.slew_rate.iter().all(|x| x.is_finite() && *x > 0.),
            "invalid torso command ranges/rates"
        );
        Ok(())
    }

    pub fn validate_target(&self, target: [f32; 3]) -> Result<()> {
        ensure!(
            target
                .iter()
                .zip([self.roll, self.pitch, self.yaw])
                .all(|(x, r)| x.is_finite() && (r[0]..=r[1]).contains(x)),
            "torso command must be finite and within trained ranges"
        );
        Ok(())
    }

    pub fn advance(&self, current: [f32; 3], target: [f32; 3], dt: f32) -> Result<[f32; 3]> {
        self.validate_target(target)?;
        let current = Vec3::from_array(current);
        let step = Vec3::from_array(self.slew_rate) * dt;
        Ok((current + (Vec3::from_array(target) - current).clamp(-step, step)).to_array())
    }

    pub fn clamp(&self, target: [f32; 3]) -> Result<[f32; 3]> {
        let target = Vec3::from_array(target);
        ensure!(target.is_finite(), "non-finite tracked torso command");
        Ok(target
            .clamp(
                Vec3::new(self.roll[0], self.pitch[0], self.yaw[0]),
                Vec3::new(self.roll[1], self.pitch[1], self.yaw[1]),
            )
            .to_array())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_slew_and_bounds() {
        let config = Config {
            roll: [-0.2, 0.2],
            pitch: [-0.3, 0.3],
            yaw: [-0.5, 0.5],
            slew_rate: [0.3, 0.2, 0.5],
        };
        config.validate().unwrap();
        let target = [0.1, -0.2, 0.4];
        assert_eq!(config.advance([0.; 3], target, 0.).unwrap(), [0.; 3]);
        let next = Vec3::from_array(config.advance([0.; 3], target, 0.02).unwrap());
        assert!(next.abs_diff_eq(Vec3::new(0.006, -0.004, 0.01), 1e-7));
        assert_eq!(config.advance(target, target, 1.).unwrap(), target);
        assert!(config.validate_target([0., 0., 0.6]).is_err());
        assert!(config.validate_target([f32::NAN, 0., 0.]).is_err());
        assert_eq!(config.clamp([1., -2., 3.]).unwrap(), [0.2, -0.3, 0.5]);
        assert!(config.clamp([f32::INFINITY, 0., 0.]).is_err());
    }
}
