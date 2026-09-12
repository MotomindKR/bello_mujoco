//! Conversion at MuJoCo/NPZ (wxyz) and actor observation boundaries.

use anyhow::{Result, ensure};
use glam::{DMat3, DQuat};

pub fn from_wxyz([w, x, y, z]: [f64; 4]) -> DQuat {
    DQuat::from_xyzw(x, y, z, w)
}

pub fn to_wxyz(q: DQuat) -> [f64; 4] {
    [q.w, q.x, q.y, q.z]
}

pub fn checked_quaternion(q: DQuat) -> Result<DQuat> {
    ensure!(
        q.is_finite() && (q.length() - 1.).abs() < 0.1,
        "invalid motion quaternion"
    );
    Ok(q.normalize())
}

pub fn rotation_6d(q: DQuat) -> [f64; 6] {
    let m = DMat3::from_quat(q);
    // Training flattens the first two matrix columns in row-major order.
    [
        m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y, m.x_axis.z, m.y_axis.z,
    ]
}
