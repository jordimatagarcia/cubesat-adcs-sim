//! Multiplicative extended Kalman filter, Markley & Crassidis (2014) §6.2. Port of
//! `adcs-c/src/mekf.c`.
//!
//! State (9): attitude error δθ (folded into `q_ref` after every update), gyro bias (rad/s),
//! magnetometer bias (µT). The gyro is the rate input; the sun and magnetometer vectors are
//! measured against the attitude: `y_sun = A(q) s_i`, `y_B = A(q) B_i + β_m`.

use crate::abi::MEKF_NX as NX;
use crate::linalg::{
    self, all_finite, diag, eye, mul, mul_bt, mul_vec, norm, right_divide, skew, solve, symmetrize,
};
use crate::quat::{self, Quat};
use crate::real::{NAN, Real, cos, sin};

pub const NY: usize = 6;
pub const ATT: usize = 0;
pub const GYRO: usize = 3;
pub const MAG: usize = 6;
const UNAVAILABLE_VARIANCE: Real = 1e12;

/// Temporaries, kept in the filter so that a step uses little stack.
#[derive(Clone, Debug, PartialEq)]
struct Work {
    phi: [Real; NX * NX],
    tmp: [Real; NX * NX],
    p_pred: [Real; NX * NX],
    ikh: [Real; NX * NX],
    p_upd: [Real; NX * NX],
    krk: [Real; NX * NX],
    h: [Real; NY * NX],
    hp: [Real; NY * NX],
    pht: [Real; NX * NY],
    gain: [Real; NX * NY],
    kr: [Real; NX * NY],
    solve: [Real; NY * NY + NY * NX],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mekf {
    sigma_gyro_noise: Real,
    sigma_gyro_bias_walk: Real,
    sigma_mag_bias_walk: Real,
    r_diag: [Real; NY],
    pub q_ref: Quat,
    pub x: [Real; NX],
    pub p: [Real; NX * NX],
    p0_diag: [Real; NX],
    // last update, for the fault detector
    has_update: bool,
    available: [bool; NY],
    innovation: [Real; NY],
    innovation_cov: [Real; NY * NY],
    work: Work,
}

/// Exact `Φ11 = exp(−[ω×]dt)` and `Φ12 = −∫₀^dt exp(−[ω×]s) ds` for a constant rate.
fn rotation_blocks(w: &[Real; 3], dt: Real) -> ([Real; 9], [Real; 9]) {
    let wx = skew(w);
    let rate = norm(w);
    let (mut phi11, mut phi12) = ([0.0; 9], [0.0; 9]);
    if rate < 1e-9 {
        for i in 0..9 {
            phi11[i] = -wx[i] * dt;
            phi12[i] = 0.5 * wx[i] * dt * dt;
        }
    } else {
        let th = rate * dt;
        let (s, c) = (sin(th), cos(th));
        let rate2 = rate * rate;
        let mut scaled = [0.0; 9];
        for i in 0..9 {
            scaled[i] = ((1.0 - c) / rate2) * wx[i];
        }
        mul(3, 3, 3, &scaled, &wx, &mut phi11); // ((1 − cos θ)/ω²) [ω×] [ω×]
        let mut wx2 = [0.0; 9];
        mul(3, 3, 3, &wx, &wx, &mut wx2);
        for i in 0..9 {
            phi11[i] -= s / rate * wx[i];
            phi12[i] = wx[i] * (1.0 - c) / rate2 - wx2[i] * (th - s) / (rate2 * rate);
        }
    }
    for i in 0..3 {
        phi11[i * 4] += 1.0;
        phi12[i * 4] -= dt;
    }
    (phi11, phi12)
}

impl Mekf {
    pub const ZERO: Self = Self {
        sigma_gyro_noise: 0.0,
        sigma_gyro_bias_walk: 0.0,
        sigma_mag_bias_walk: 0.0,
        r_diag: [0.0; NY],
        q_ref: [0.0, 0.0, 0.0, 1.0],
        x: [0.0; NX],
        p: [0.0; NX * NX],
        p0_diag: [0.0; NX],
        has_update: false,
        available: [false; NY],
        innovation: [0.0; NY],
        innovation_cov: [0.0; NY * NY],
        work: Work {
            phi: [0.0; NX * NX],
            tmp: [0.0; NX * NX],
            p_pred: [0.0; NX * NX],
            ikh: [0.0; NX * NX],
            p_upd: [0.0; NX * NX],
            krk: [0.0; NX * NX],
            h: [0.0; NY * NX],
            hp: [0.0; NY * NX],
            pht: [0.0; NX * NY],
            gain: [0.0; NX * NY],
            kr: [0.0; NX * NY],
            solve: [0.0; NY * NY + NY * NX],
        },
    };

    pub fn configure(
        &mut self,
        sigma_gyro_noise: Real,
        sigma_gyro_bias_walk: Real,
        sigma_mag_noise: Real,
        sigma_mag_bias_walk: Real,
        sigma_sun_noise: Real,
    ) {
        self.sigma_gyro_noise = sigma_gyro_noise;
        self.sigma_gyro_bias_walk = sigma_gyro_bias_walk;
        self.sigma_mag_bias_walk = sigma_mag_bias_walk;
        for i in 0..3 {
            self.r_diag[i] = sigma_sun_noise * sigma_sun_noise;
            self.r_diag[3 + i] = sigma_mag_noise * sigma_mag_noise;
        }
        self.reset(&[0.0, 0.0, 0.0, 1.0], &[1.0; NX]);
    }

    pub fn reset(&mut self, q0: &[Real], p0_diag: &[Real; NX]) {
        self.q_ref.copy_from_slice(&q0[..4]);
        quat::normalize(&mut self.q_ref);
        self.x = [0.0; NX];
        self.p0_diag = *p0_diag;
        diag(NX, p0_diag, &mut self.p);
        self.has_update = false;
    }

    pub fn clear_update(&mut self) {
        self.has_update = false;
    }

    /// Propagates over `dt` with the rate gyro (rad/s), then updates with the vectors (unit
    /// sun, field in µT). Returns false if the result was not finite and the covariance was
    /// reset.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn step(
        &mut self,
        gyro: &[Real; 3],
        dt: Real,
        sun_b: &[Real; 3],
        sun_i: &[Real; 3],
        sun_available: bool,
        mag_ut: &[Real; 3],
        mag_i_ut: &[Real; 3],
        mag_available: bool,
    ) -> bool {
        let k = &mut self.work;

        // ---- propagation ----------------------------------------------------------------
        let w = [
            gyro[0] - self.x[GYRO],
            gyro[1] - self.x[GYRO + 1],
            gyro[2] - self.x[GYRO + 2],
        ];
        self.q_ref = quat::mul(&quat::from_rate(&w, dt), &self.q_ref);

        let (phi11, phi12) = rotation_blocks(&w, dt);
        eye(NX, &mut k.phi);
        for i in 0..3 {
            for j in 0..3 {
                k.phi[(ATT + i) * NX + ATT + j] = phi11[i * 3 + j];
                k.phi[(ATT + i) * NX + GYRO + j] = phi12[i * 3 + j];
            }
        }
        let sv2 = self.sigma_gyro_noise * self.sigma_gyro_noise * dt;
        let su2 = self.sigma_gyro_bias_walk * self.sigma_gyro_bias_walk;
        let dt2 = dt * dt;
        mul(NX, NX, NX, &k.phi, &self.p, &mut k.tmp);
        mul_bt(NX, NX, NX, &k.tmp, &k.phi, &mut k.p_pred);
        for i in 0..3 {
            k.p_pred[(ATT + i) * NX + ATT + i] += sv2 * dt + su2 * (dt2 * dt) / 3.0;
            k.p_pred[(ATT + i) * NX + GYRO + i] += -(su2 * dt2 / 2.0);
            k.p_pred[(GYRO + i) * NX + ATT + i] += -(su2 * dt2 / 2.0);
            k.p_pred[(GYRO + i) * NX + GYRO + i] += su2 * dt;
            k.p_pred[(MAG + i) * NX + MAG + i] +=
                self.sigma_mag_bias_walk * self.sigma_mag_bias_walk * dt;
        }

        // ---- update ---------------------------------------------------------------------
        let a = quat::to_dcm(&self.q_ref);
        let (mut sun_hat, mut mag_body) = ([0.0; 3], [0.0; 3]);
        mul_vec(3, 3, &a, sun_i, &mut sun_hat);
        mul_vec(3, 3, &a, mag_i_ut, &mut mag_body);
        let (mut y_hat, mut y, mut avail) = ([0.0; NY], [0.0; NY], [false; NY]);
        for i in 0..3 {
            y_hat[i] = sun_hat[i];
            y_hat[3 + i] = mag_body[i] + self.x[MAG + i];
            avail[i] = sun_available;
            avail[3 + i] = mag_available;
            y[i] = if sun_available { sun_b[i] } else { y_hat[i] };
            y[3 + i] = if mag_available {
                mag_ut[i]
            } else {
                y_hat[3 + i]
            };
        }
        k.h = [0.0; NY * NX];
        let sk = skew(&sun_hat);
        for i in 0..3 {
            for j in 0..3 {
                k.h[i * NX + ATT + j] = sk[i * 3 + j];
            }
        }
        let sk = skew(&mag_body);
        for i in 0..3 {
            for j in 0..3 {
                k.h[(3 + i) * NX + ATT + j] = sk[i * 3 + j];
            }
            k.h[(3 + i) * NX + MAG + i] = 1.0;
        }
        let mut r = [0.0; NY];
        for i in 0..NY {
            r[i] = if avail[i] {
                self.r_diag[i]
            } else {
                UNAVAILABLE_VARIANCE
            };
        }
        let mut s = [0.0; NY * NY];
        mul(NY, NX, NX, &k.h, &k.p_pred, &mut k.hp);
        mul_bt(NY, NX, NY, &k.hp, &k.h, &mut s);
        for i in 0..NY {
            s[i * NY + i] += r[i];
        }
        mul_bt(NX, NX, NY, &k.p_pred, &k.h, &mut k.pht);
        let solved = right_divide(NX, NY, &k.pht, &s, &mut k.gain, &mut k.solve);
        let mut innov = [0.0; NY];
        for i in 0..NY {
            innov[i] = y[i] - y_hat[i];
        }
        let mut dx = [0.0; NX];
        mul_vec(NX, NY, &k.gain, &innov, &mut dx);
        // Joseph form: (I − KH) P (I − KH)ᵀ + K R Kᵀ
        mul(NX, NY, NX, &k.gain, &k.h, &mut k.ikh);
        for x in &mut k.ikh {
            *x = -*x;
        }
        for i in 0..NX {
            k.ikh[i * NX + i] += 1.0;
        }
        mul(NX, NX, NX, &k.ikh, &k.p_pred, &mut k.tmp);
        mul_bt(NX, NX, NX, &k.tmp, &k.ikh, &mut k.p_upd);
        for i in 0..NX {
            for j in 0..NY {
                k.kr[i * NY + j] = k.gain[i * NY + j] * r[j];
            }
        }
        mul_bt(NX, NY, NX, &k.kr, &k.gain, &mut k.krk);
        for i in 0..NX * NX {
            k.p_upd[i] += k.krk[i];
        }
        symmetrize(NX, &mut k.p_upd);

        self.has_update = true;
        for i in 0..NY {
            self.available[i] = avail[i];
            self.innovation[i] = if avail[i] { innov[i] } else { NAN };
        }
        self.innovation_cov = s;

        if solved.is_err() || !all_finite(&dx) || !all_finite(&k.p_upd) {
            diag(NX, &self.p0_diag, &mut self.p);
            return false;
        }
        let mut e = [0.5 * dx[ATT], 0.5 * dx[ATT + 1], 0.5 * dx[ATT + 2], 1.0];
        quat::normalize(&mut e);
        let mut qn = quat::mul(&e, &self.q_ref);
        quat::normalize(&mut qn);
        self.q_ref = qn;
        for i in 0..3 {
            self.x[GYRO + i] += dx[GYRO + i];
            self.x[MAG + i] += dx[MAG + i];
        }
        self.p = k.p_upd;
        true
    }

    /// Adds variance to the diagonal (standard deviations given).
    pub fn inflate(&mut self, attitude: Real, gyro_bias: Real, mag_bias: Real) {
        for i in 0..3 {
            self.p[(ATT + i) * NX + ATT + i] += attitude * attitude;
            self.p[(GYRO + i) * NX + GYRO + i] += gyro_bias * gyro_bias;
            self.p[(MAG + i) * NX + MAG + i] += mag_bias * mag_bias;
        }
    }

    /// Normalised innovation squared of the sun (block 0) or magnetometer (block 1) in the
    /// last update; NaN if there was none or the measurement was unavailable.
    #[must_use]
    pub fn nis(&self, block: usize) -> Real {
        let r0 = if block == 0 { 0 } else { 3 };
        if !self.has_update || !self.available[r0..r0 + 3].iter().all(|&a| a) {
            return NAN;
        }
        let (mut s, mut e) = ([0.0; 9], [0.0; 3]);
        for i in 0..3 {
            e[i] = self.innovation[r0 + i];
            for j in 0..3 {
                s[i * 3 + j] = self.innovation_cov[(r0 + i) * NY + r0 + j];
            }
        }
        let mut x = e;
        if solve(3, 1, &mut s, &mut x).is_err() {
            return NAN;
        }
        linalg::dot(&e, &x)
    }
}
