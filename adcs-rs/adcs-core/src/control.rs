//! Mode manager and magnetorquer control laws, shared by both estimator variants. Port of
//! `adcs-c/src/control.c` (thesis §4.2.3).

use crate::abi::{
    Config, MODE_DETUMBLE, MODE_ESTIMATE, MODE_POINT, MODE_SAFE, STATUS_BAD_INPUT,
    STATUS_COVARIANCE_RESET, STATUS_DIPOLE_SATURATED,
};
use crate::linalg::{cross, dot, norm};
use crate::quat::{conj, mul};
use crate::real::{Real, abs};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control {
    pub mode: u32,
    pub mode_time: Real,
    pub b_prev: [Real; 3],
    pub has_b_prev: bool,
}

impl Control {
    pub const NEW: Self = Self {
        mode: MODE_ESTIMATE,
        mode_time: 0.0,
        b_prev: [0.0; 3],
        has_b_prev: false,
    };

    /// New mode and dipole command from the current estimate (`b_body` in tesla). Adds
    /// `STATUS_DIPOLE_SATURATED` to `status` when the command was scaled down.
    pub fn update(
        &mut self,
        cfg: &Config,
        q: &[Real],
        w: &[Real],
        b_body: &[Real; 3],
        status: &mut u32,
    ) -> (u32, [Real; 3]) {
        let rate = norm(w);
        let mode = if *status & (STATUS_BAD_INPUT | STATUS_COVARIANCE_RESET) != 0 {
            MODE_SAFE
        } else if cfg.control_enabled == 0 {
            MODE_ESTIMATE
        } else if self.mode != MODE_DETUMBLE {
            if rate > cfg.detumble_enter_rate {
                MODE_DETUMBLE
            } else if self.mode == MODE_POINT
                || (self.mode == MODE_ESTIMATE && self.mode_time >= cfg.settle_time)
            {
                MODE_POINT
            } else {
                MODE_ESTIMATE
            }
        } else if rate < cfg.detumble_exit_rate {
            MODE_ESTIMATE
        } else {
            MODE_DETUMBLE
        };
        self.mode_time = if mode == self.mode {
            self.mode_time + cfg.dt
        } else {
            0.0
        };
        self.mode = mode;

        let mut m = [0.0; 3];
        if mode == MODE_DETUMBLE && self.has_b_prev {
            // B-dot, eq. 114: m = −k dB/dt
            for i in 0..3 {
                m[i] = -cfg.bdot_gain * ((b_body[i] - self.b_prev[i]) / cfg.dt);
            }
        } else if mode == MODE_POINT {
            let tau = quaternion_feedback(q, &cfg.q_target, w, cfg.kp, cfg.kd);
            m = torque_to_dipole(&tau, b_body);
        }
        if saturate(&mut m, &cfg.max_dipole) {
            *status |= STATUS_DIPOLE_SATURATED;
        }
        self.b_prev = *b_body;
        self.has_b_prev = true;
        (mode, m)
    }
}

/// Eq. 119: `τ = −kp sign(δq4) δq1:3 − kd ω`, with `δq = q ⊗ q_d⁻¹` (eq. 116).
#[must_use]
pub fn quaternion_feedback(
    q: &[Real],
    q_target: &[Real],
    w: &[Real],
    kp: Real,
    kd: Real,
) -> [Real; 3] {
    let dq = mul(q, &conj(q_target));
    let sign = if dq[3] >= 0.0 { 1.0 } else { -1.0 };
    [
        -kp * sign * dq[0] - kd * w[0],
        -kp * sign * dq[1] - kd * w[1],
        -kp * sign * dq[2] - kd * w[2],
    ]
}

/// Eq. 121: `m = (B × τ) / |B|²`
#[must_use]
pub fn torque_to_dipole(tau: &[Real], b: &[Real]) -> [Real; 3] {
    let b2 = dot(b, b);
    if b2 == 0.0 {
        return [0.0; 3];
    }
    let c = cross(b, tau);
    [c[0] / b2, c[1] / b2, c[2] / b2]
}

/// Scales the whole vector so no coil exceeds its limit, keeping the direction. A NaN
/// command propagates (and is reported as saturated), like the reference.
pub fn saturate(m: &mut [Real; 3], max_dipole: &[Real; 3]) -> bool {
    let mut ratio = abs(m[0]) / max_dipole[0];
    for i in 1..3 {
        let r = abs(m[i]) / max_dipole[i];
        if (r.is_nan() || r > ratio) && !ratio.is_nan() {
            ratio = r;
        }
    }
    if ratio <= 1.0 {
        return false;
    }
    for x in m.iter_mut() {
        *x /= ratio;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::TEST_TOL;

    #[test]
    fn control_laws() {
        let tau = quaternion_feedback(
            &[0.0, 0.0, 0.0, 1.0],
            &[0.0, 0.0, 0.0, 1.0],
            &[0.1, 0.0, 0.0],
            1.0,
            2.0,
        );
        assert!((tau[0] + 0.2).abs() < TEST_TOL);

        let b = [0.0, 0.0, 2e-5];
        let t_perp = [1e-6, -2e-6, 0.0];
        let produced = cross(&torque_to_dipole(&t_perp, &b), &b);
        for i in 0..3 {
            assert!((produced[i] - t_perp[i]).abs() < 1e-18);
        }

        let max = [0.1, 0.1, 0.05];
        let mut big = [0.2, 0.05, 0.05];
        assert!(saturate(&mut big, &max));
        assert!((big[0] - 0.1).abs() < TEST_TOL && (big[1] - 0.025).abs() < TEST_TOL);
        let mut small = [0.01, 0.0, 0.0];
        assert!(!saturate(&mut small, &max));
        let mut nan = [Real::NAN, 0.0, 0.0];
        assert!(saturate(&mut nan, &max) && nan[0].is_nan());
    }
}
