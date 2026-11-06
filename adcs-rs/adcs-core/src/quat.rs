//! Quaternions: scalar-last, `A(q ⊗ p) = A(q) A(p)`, `A(q)` maps inertial to body
//! coordinates (docs/conventions.md). Port of `adcs-c/src/quat.c`.

use crate::linalg::{cross, dot};
use crate::real::{Real, cos, sin, sqrt};

pub type Quat = [Real; 4];

#[must_use]
pub fn mul(q: &[Real], p: &[Real]) -> Quat {
    let c = cross(q, p);
    [
        q[3] * p[0] + p[3] * q[0] - c[0],
        q[3] * p[1] + p[3] * q[1] - c[1],
        q[3] * p[2] + p[3] * q[2] - c[2],
        q[3] * p[3] - dot(q, p),
    ]
}

#[must_use]
pub fn conj(q: &[Real]) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

#[must_use]
pub fn norm(q: &[Real]) -> Real {
    sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
}

/// Divides by the norm; leaves `q` unchanged if the norm is zero.
pub fn normalize(q: &mut [Real]) {
    let n = norm(q);
    if n > 0.0 {
        for x in &mut q[..4] {
            *x /= n;
        }
    }
}

/// Attitude matrix `A(q)`, row-major.
#[must_use]
pub fn to_dcm(q: &[Real]) -> [Real; 9] {
    let (q1, q2, q3, q4) = (q[0], q[1], q[2], q[3]);
    [
        q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4,
        2.0 * (q1 * q2 + q3 * q4),
        2.0 * (q1 * q3 - q2 * q4),
        2.0 * (q1 * q2 - q3 * q4),
        -q1 * q1 + q2 * q2 - q3 * q3 + q4 * q4,
        2.0 * (q2 * q3 + q1 * q4),
        2.0 * (q1 * q3 + q2 * q4),
        2.0 * (q2 * q3 - q1 * q4),
        -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4,
    ]
}

/// `A(q) v`
#[must_use]
pub fn rotate(q: &[Real], v: &[Real]) -> [Real; 3] {
    let a = to_dcm(q);
    let mut out = [0.0; 3];
    crate::linalg::mul_vec(3, 3, &a, v, &mut out);
    out
}

/// Quaternion of a body rotating at the constant rate `w` for `dt` (exact).
#[must_use]
pub fn from_rate(w: &[Real], dt: Real) -> Quat {
    let rate = crate::linalg::norm(w);
    let half = 0.5 * rate * dt;
    // sin(x)/x expanded near 0 to avoid dividing by zero
    let sinc = if rate > 1e-12 {
        sin(half) / rate
    } else {
        0.5 * dt
    };
    [w[0] * sinc, w[1] * sinc, w[2] * sinc, cos(half)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg;
    use crate::real::TEST_TOL;

    #[test]
    fn frame_rotation_and_composition() {
        let s = sqrt(0.5);
        let qz = [0.0, 0.0, s, s];
        let v = rotate(&qz, &[1.0, 0.0, 0.0]);
        assert!(v[0].abs() < TEST_TOL && (v[1] + 1.0).abs() < TEST_TOL && v[2].abs() < TEST_TOL);

        let mut p = [0.1, -0.3, 0.2, 0.9];
        normalize(&mut p);
        let (aq, ap, aqp) = (to_dcm(&qz), to_dcm(&p), to_dcm(&mul(&qz, &p)));
        let mut prod = [0.0; 9];
        linalg::mul(3, 3, 3, &aq, &ap, &mut prod);
        for (x, y) in aqp.iter().zip(prod.iter()) {
            assert!((x - y).abs() < TEST_TOL);
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn rate_quaternion() {
        let s = sqrt(0.5);
        let r = from_rate(&[0.0, 0.0, core::f64::consts::FRAC_PI_2 as Real], 1.0);
        for (x, y) in r.iter().zip([0.0, 0.0, s, s].iter()) {
            assert!((x - y).abs() < TEST_TOL.max(1e-6));
        }
        assert_eq!(from_rate(&[0.0; 3], 0.1), [0.0, 0.0, 0.0, 1.0]);
    }
}
