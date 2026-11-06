//! The ADCS: input checks, the thesis (FLAE + two-stage UKF) and MEKF variants, and the
//! mode manager. Port of `adcs-c/src/adcs.c`; the golden vectors check that both match the
//! Python reference.

use crate::abi::{
    Config, Input, MEKF_NX, MODE_OFF, NB, NX, Output, STATUS_BAD_INPUT, STATUS_BIAS_FAULT,
    STATUS_COVARIANCE_RESET, STATUS_FLAE_FAILED, STATUS_NO_ATTITUDE_FIX, STATUS_NOT_INITIALISED,
    VARIANT_MEKF, VARIANT_THESIS,
};
use crate::control::Control;
use crate::flae::flae2;
use crate::linalg::{all_finite, cross, det, inverse, mul_vec, norm};
use crate::mekf::{self, Mekf};
use crate::quat::{self, Quat};
use crate::real::{NAN, Real, cos, rint, sin};
use crate::tsukf::{self, Tsukf};

const MAGIC: u32 = 0x4144_4353; // "ADCS"
/// Tesla → microtesla, the filters' unit for the field.
const UT: Real = 1e6;

// Plausibility limits (src/adcssim/adcs_ref/limits.py): a reading outside them is treated
// like a NaN, so absurd values cannot overflow the filter state.
const GYRO_MAX: Real = 50.0; // rad/s
const FIELD_MAX: Real = 1e-3; // T
const UNIT_MIN: Real = 0.5; // norm of a direction that should be unit
const UNIT_MAX: Real = 2.0;

/// The configuration was rejected by [`Adcs::init`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidConfig;

/// [`Adcs::step`] was called before a successful [`Adcs::init`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotInitialised;

fn gyro_plausible(v: &[Real; 3]) -> bool {
    all_finite(v) && norm(v) <= GYRO_MAX
}

fn field_plausible(v: &[Real; 3]) -> bool {
    all_finite(v) && {
        let n = norm(v);
        n > 0.0 && n <= FIELD_MAX
    }
}

fn unit_plausible(v: &[Real; 3]) -> bool {
    all_finite(v) && {
        let n = norm(v);
        (UNIT_MIN..=UNIT_MAX).contains(&n)
    }
}

fn scale3(v: &[Real; 3], s: Real) -> [Real; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn div3(v: &[Real], d: Real) -> [Real; 3] {
    [v[0] / d, v[1] / d, v[2] / d]
}

fn pair(a: &[Real], b: &[Real]) -> [Real; 6] {
    [a[0], a[1], a[2], b[0], b[1], b[2]]
}

fn all_positive(v: &[Real]) -> bool {
    v.iter().all(|&x| x > 0.0 && x.is_finite())
}

fn all_nonnegative(v: &[Real]) -> bool {
    v.iter().all(|&x| x >= 0.0 && x.is_finite())
}

fn config_valid(c: &Config) -> bool {
    let mut work = [0.0; 9];
    let weights = [c.flae_weight_sun, c.flae_weight_mag];
    let control = [
        c.bdot_gain,
        c.kp,
        c.kd,
        c.detumble_enter_rate,
        c.detumble_exit_rate,
        c.settle_time,
    ];
    if !all_positive(&[c.dt])
        || !all_finite(&c.inertia)
        || det(3, &c.inertia, &mut work) == 0.0
        || !all_positive(&c.max_dipole)
        || !all_nonnegative(&weights)
        || !(weights[0] + weights[1] > 0.0)
        || !all_positive(&[c.flae_tolerance])
        || !all_nonnegative(&control)
        || !all_finite(&c.q_target)
    {
        return false;
    }
    match c.variant {
        VARIANT_THESIS => {
            all_positive(&[c.process_dt, c.ukf_alpha])
                && all_positive(&c.p_x0)
                && all_positive(&c.p_b0)
                && all_nonnegative(&c.w_x)
                && all_nonnegative(&c.w_b)
                && all_positive(&c.v)
        }
        VARIANT_MEKF => {
            all_positive(&[
                c.sigma_mag_noise,
                c.sigma_sun_noise,
                c.p0_attitude,
                c.p0_gyro_bias,
                c.p0_mag_bias,
            ]) && all_nonnegative(&[
                c.sigma_gyro_noise,
                c.sigma_gyro_bias_walk,
                c.sigma_mag_bias_walk,
                c.fault_gyro_bias_sigma,
                c.fault_mag_bias_sigma,
                c.fault_attitude_sigma,
                c.fault_nis_threshold,
            ])
        }
        _ => false,
    }
}

// ---- thesis variant: process model ---------------------------------------------------

struct ProcessCtx<'a> {
    dt: Real,
    process_dt: Real,
    torque: [Real; 3],
    inertia: &'a [Real; 9],
    inertia_inv: &'a [Real; 9],
}

/// Eq. 126, except that the body vectors rotate as `−ω × v` using the state's own vectors
/// (docs/adcs.md explains why the thesis form diverges in eclipse).
fn xdot(x: &[Real], c: &ProcessCtx<'_>) -> [Real; NX] {
    let mut q: Quat = [x[0], x[1], x[2], x[3]];
    let qn = quat::norm(&q);
    for v in &mut q {
        *v /= qn;
    }
    let w = &x[tsukf::W..tsukf::W + 3];
    let mut out = [0.0; NX];
    let t = cross(&q, w);
    for i in 0..3 {
        out[i] = 0.5 * (q[3] * w[i] + t[i]);
    }
    out[3] = -0.5 * (q[0] * w[0] + q[1] * w[1] + q[2] * w[2]);
    let t = cross(w, &x[tsukf::SUN..tsukf::SUN + 3]);
    for i in 0..3 {
        out[tsukf::SUN + i] = -t[i];
    }
    let t = cross(w, &x[tsukf::MAG..tsukf::MAG + 3]);
    for i in 0..3 {
        out[tsukf::MAG + i] = -t[i];
    }
    // ω̇ = I⁻¹ (−ω × Iω + τ)
    let mut iw = [0.0; 3];
    mul_vec(3, 3, c.inertia, w, &mut iw);
    let mut g = cross(&[-w[0], -w[1], -w[2]], &iw);
    for i in 0..3 {
        g[i] += c.torque[i];
    }
    mul_vec(3, 3, c.inertia_inv, &g, &mut out[tsukf::W..tsukf::W + 3]);
    out
}

/// RK4 over `dt` in steps of at most `process_dt`, applied to one sigma point in place.
fn process_model(x: &mut [Real], c: &ProcessCtx<'_>) {
    let mut steps = rint(c.dt / c.process_dt); // round half to even, like Python
    if steps < 1.0 {
        steps = 1.0;
    }
    let h = c.dt / steps;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = steps as u32;
    let mut xt = [0.0; NX];
    for _ in 0..n {
        let mut k1 = xdot(x, c);
        for i in 0..NX {
            k1[i] *= h;
            xt[i] = x[i] + 0.5 * k1[i];
        }
        let mut k2 = xdot(&xt, c);
        for i in 0..NX {
            k2[i] *= h;
            xt[i] = x[i] + 0.5 * k2[i];
        }
        let mut k3 = xdot(&xt, c);
        for i in 0..NX {
            k3[i] *= h;
            xt[i] = x[i] + k3[i];
        }
        let mut k4 = xdot(&xt, c);
        for i in 0..NX {
            k4[i] *= h;
            x[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
    }
}

/// A fixed inertial direction seen from a body rotating at `w` for `dt` (Rodrigues), unit.
fn propagate_body_vector(v: &[Real], w: &[Real], dt: Real) -> [Real; 3] {
    let rate = norm(w);
    let angle = rate * dt;
    let mut out = [v[0], v[1], v[2]];
    if angle >= 1e-12 {
        let k = div3(w, rate);
        let (c, s) = (cos(angle), sin(angle));
        let kv = cross(&k, v);
        let kd = crate::linalg::dot(&k, v);
        for i in 0..3 {
            out[i] = v[i] * c - kv[i] * s + k[i] * kd * (1.0 - c);
        }
    }
    let n = norm(&out);
    if n > 0.0 {
        out = div3(&out, n);
    }
    out
}

// ---- state -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct Thesis {
    filter: Tsukf,
    inertia: [Real; 9],
    inertia_inv: [Real; 9],
    m_prev: [Real; 3],
    last_flae_q: Option<Quat>,
}

#[derive(Clone, Debug, PartialEq)]
struct MekfState {
    filter: Mekf,
    gyro_prev: [Real; 3],
    suspect_steps: u32,
}

/// The whole ADCS state. Large (about 40 KB in double precision) and with no heap use:
/// place it in static memory, or let the caller of the C interface allocate it.
#[derive(Clone, Debug, PartialEq)]
pub struct Adcs {
    magic: u32,
    cfg: Config,
    initialised: bool,
    control: Control,
    thesis: Thesis,
    mekf: MekfState,
}

impl Default for Adcs {
    fn default() -> Self {
        Self::new()
    }
}

impl Adcs {
    /// An ADCS that is not yet initialised: [`step`](Self::step) fails until
    /// [`init`](Self::init) succeeds.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            magic: 0,
            cfg: Config::ZERO,
            initialised: false,
            control: Control::NEW,
            thesis: Thesis {
                filter: Tsukf::ZERO,
                inertia: [0.0; 9],
                inertia_inv: [0.0; 9],
                m_prev: [0.0; 3],
                last_flae_q: None,
            },
            mekf: MekfState {
                filter: Mekf::ZERO,
                gyro_prev: [0.0; 3],
                suspect_steps: 0,
            },
        }
    }

    /// (Re)starts the ADCS with a configuration.
    ///
    /// # Errors
    /// [`InvalidConfig`] if the configuration is rejected; the ADCS is then left
    /// uninitialised.
    pub fn init(&mut self, cfg: &Config) -> Result<(), InvalidConfig> {
        self.magic = 0;
        if !config_valid(cfg) {
            return Err(InvalidConfig);
        }
        self.cfg = *cfg;
        self.initialised = false;
        self.control = Control::NEW;
        if cfg.variant == VARIANT_THESIS {
            let t = &mut self.thesis;
            let mut work = [0.0; 9];
            t.inertia = cfg.inertia;
            inverse(3, &cfg.inertia, &mut t.inertia_inv, &mut work).map_err(|_| InvalidConfig)?;
            t.m_prev = [0.0; 3];
            t.last_flae_q = None;
            t.filter
                .configure(cfg.ukf_alpha, &cfg.w_x, &cfg.w_b, &cfg.v);
            let mut x0 = [0.0; NX];
            x0[3] = 1.0;
            t.filter.reset(&x0, &cfg.p_x0, &[0.0; NB], &cfg.p_b0);
        } else {
            let m = &mut self.mekf;
            m.filter.configure(
                cfg.sigma_gyro_noise,
                cfg.sigma_gyro_bias_walk,
                cfg.sigma_mag_noise,
                cfg.sigma_mag_bias_walk,
                cfg.sigma_sun_noise,
            );
            m.gyro_prev = [0.0; 3];
            m.suspect_steps = 0;
        }
        self.magic = MAGIC;
        Ok(())
    }

    /// Processes one control period.
    ///
    /// # Errors
    /// [`NotInitialised`] before a successful [`init`](Self::init). Data problems are
    /// reported in the output's status flags instead.
    pub fn step(&mut self, input: &Input) -> Result<Output, NotInitialised> {
        if self.magic != MAGIC {
            return Err(NotInitialised);
        }
        Ok(if self.cfg.variant == VARIANT_THESIS {
            self.thesis_step(input)
        } else {
            self.mekf_step(input)
        })
    }

    /// The output reported for a step on an uninitialised ADCS.
    #[must_use]
    pub fn not_initialised_output() -> Output {
        Output::empty(MODE_OFF, STATUS_NOT_INITIALISED)
    }

    // ---- thesis variant ----------------------------------------------------------------

    fn thesis_finish(&mut self, mut status: u32) -> Output {
        let x = self.thesis.filter.x;
        let b_body = div3(&x[tsukf::MAG..tsukf::MAG + 3], UT);
        let (mode, m) = self.control.update(
            &self.cfg,
            &x[tsukf::Q..tsukf::Q + 4],
            &x[tsukf::W..tsukf::W + 3],
            &b_body,
            &mut status,
        );
        let f = &self.thesis.filter;
        self.thesis.m_prev = m;
        let mut p_diag = [0.0; NX + NB];
        for i in 0..NX {
            p_diag[i] = f.px[i * NX + i];
        }
        for i in 0..NB {
            p_diag[NX + i] = f.pb[i * NB + i];
        }
        Output {
            q_est: [x[0], x[1], x[2], x[3]],
            w_est: [x[tsukf::W], x[tsukf::W + 1], x[tsukf::W + 2]],
            bias_mag: div3(&f.b[0..3], UT),
            bias_gyro: [f.b[3], f.b[4], f.b[5]],
            p_diag,
            m_cmd: m,
            mode,
            status,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn thesis_step(&mut self, input: &Input) -> Output {
        let cfg = self.cfg;
        let mut status = 0;

        // Sensor readings and on-board reference models are checked separately: in eclipse
        // the sun sensor has no reading but the ephemeris still gives the Sun direction.
        let mag_ref_ok = field_plausible(&input.mag_ref_i);
        let sun_ref_ok = unit_plausible(&input.sun_ref_i);
        let mag_ok = input.mag_valid != 0 && field_plausible(&input.mag_b);
        let sun_ok = input.sun_valid != 0 && unit_plausible(&input.sun_b);
        let gyro_ok = input.gyro_valid != 0 && gyro_plausible(&input.gyro_b);
        if (input.mag_valid != 0 && !mag_ok)
            || (input.sun_valid != 0 && !sun_ok)
            || (input.gyro_valid != 0 && !gyro_ok)
        {
            status |= STATUS_BAD_INPUT;
        }
        let zero = [0.0; 3];
        let mag_ut = if mag_ok {
            scale3(&input.mag_b, UT)
        } else {
            zero
        };
        let mag_ref_ut = if mag_ref_ok {
            scale3(&input.mag_ref_i, UT)
        } else {
            zero
        };
        let sun_b = if sun_ok { input.sun_b } else { zero };
        let gyro = if gyro_ok { input.gyro_b } else { zero };

        // Attitude measurement from FLAE (needs two vectors). Without a sun reading, the
        // filter's sun vector, propagated over this step, stands in for it.
        let t = &mut self.thesis;
        let sun_for_flae = if sun_ok {
            Some(sun_b)
        } else if self.initialised {
            let x = &t.filter.x;
            Some(propagate_body_vector(
                &x[tsukf::SUN..tsukf::SUN + 3],
                &x[tsukf::W..tsukf::W + 3],
                cfg.dt,
            ))
        } else {
            None
        };
        let mut q_meas: Option<Quat> = None;
        match sun_for_flae {
            Some(sun) if mag_ok && mag_ref_ok && sun_ref_ok => {
                let body = pair(&sun, &div3(&mag_ut, norm(&mag_ut)));
                let reference = pair(&input.sun_ref_i, &div3(&mag_ref_ut, norm(&mag_ref_ut)));
                let prev_state = [t.filter.x[0], t.filter.x[1], t.filter.x[2], t.filter.x[3]];
                let prev = if self.initialised {
                    Some(&prev_state[..])
                } else {
                    t.last_flae_q.as_ref().map(|q| &q[..])
                };
                match flae2(
                    &body,
                    &reference,
                    [cfg.flae_weight_sun, cfg.flae_weight_mag],
                    prev,
                    cfg.flae_tolerance,
                    cfg.flae_max_iterations,
                ) {
                    Ok(res) => {
                        q_meas = Some(res.q);
                        t.last_flae_q = Some(res.q);
                    }
                    Err(_) => status |= STATUS_FLAE_FAILED,
                }
            }
            _ => status |= STATUS_NO_ATTITUDE_FIX,
        }

        if !self.initialised {
            // never start from an attitude that has no real sun reading
            let Some(q) = q_meas.filter(|_| sun_ok && gyro_ok) else {
                return Output::empty(MODE_OFF, status);
            };
            let mut x0 = [0.0; NX];
            x0[..4].copy_from_slice(&q);
            x0[tsukf::SUN..tsukf::SUN + 3].copy_from_slice(&sun_b);
            x0[tsukf::MAG..tsukf::MAG + 3].copy_from_slice(&mag_ut);
            x0[tsukf::W..tsukf::W + 3].copy_from_slice(&gyro);
            t.filter.reset(&x0, &cfg.p_x0, &[0.0; NB], &cfg.p_b0);
            self.initialised = true;
            self.control.has_b_prev = false;
            return self.thesis_finish(status);
        }

        let torque = if mag_ok {
            cross(&t.m_prev, &div3(&mag_ut, UT))
        } else {
            zero
        };
        let ctx = ProcessCtx {
            dt: cfg.dt,
            process_dt: cfg.process_dt,
            torque,
            inertia: &t.inertia,
            inertia_inv: &t.inertia_inv,
        };
        let mut y = [0.0; tsukf::NY];
        y[..4].copy_from_slice(&q_meas.unwrap_or([0.0; 4]));
        y[tsukf::SUN..tsukf::SUN + 3].copy_from_slice(&sun_b);
        y[tsukf::MAG..tsukf::MAG + 3].copy_from_slice(&mag_ut);
        y[tsukf::W..tsukf::W + 3].copy_from_slice(&gyro);
        let mut available = [false; tsukf::NY];
        for (i, a) in available.iter_mut().enumerate() {
            *a = match i {
                0..tsukf::SUN => q_meas.is_some(),
                tsukf::SUN..tsukf::MAG => sun_ok,
                tsukf::MAG..tsukf::W => mag_ok,
                _ => gyro_ok,
            };
        }
        let ok = t.filter.step(|x| process_model(x, &ctx), &y, &available);
        if !ok {
            status |= STATUS_COVARIANCE_RESET;
        }
        self.thesis_finish(status)
    }

    // ---- MEKF variant ------------------------------------------------------------------

    fn mekf_finish(
        &mut self,
        input: &Input,
        mut status: u32,
        gyro: &[Real; 3],
        mag_ut: &[Real; 3],
        mag_ok: bool,
    ) -> Output {
        let cfg = self.cfg;
        let f = &self.mekf.filter;
        let w = [
            gyro[0] - f.x[mekf::GYRO],
            gyro[1] - f.x[mekf::GYRO + 1],
            gyro[2] - f.x[mekf::GYRO + 2],
        ];
        let b_body = if mag_ok {
            [
                (mag_ut[0] - f.x[mekf::MAG]) / UT,
                (mag_ut[1] - f.x[mekf::MAG + 1]) / UT,
                (mag_ut[2] - f.x[mekf::MAG + 2]) / UT,
            ]
        } else if field_plausible(&input.mag_ref_i) {
            quat::rotate(&f.q_ref, &input.mag_ref_i)
        } else {
            [0.0; 3]
        };
        let q_ref = f.q_ref;
        let (mode, m) = self.control.update(&cfg, &q_ref, &w, &b_body, &mut status);
        let f = &self.mekf.filter;
        let mut p_diag = [NAN; NX + NB];
        for i in 0..3 {
            let pg = f.p[(mekf::GYRO + i) * MEKF_NX + mekf::GYRO + i];
            p_diag[i] = f.p[(mekf::ATT + i) * MEKF_NX + mekf::ATT + i]; // rad²
            p_diag[10 + i] = cfg.sigma_gyro_noise * cfg.sigma_gyro_noise + pg; // rate
            p_diag[13 + i] = f.p[(mekf::MAG + i) * MEKF_NX + mekf::MAG + i];
            p_diag[16 + i] = pg;
        }
        Output {
            q_est: q_ref,
            w_est: w,
            bias_mag: div3(&f.x[mekf::MAG..mekf::MAG + 3], UT),
            bias_gyro: [f.x[mekf::GYRO], f.x[mekf::GYRO + 1], f.x[mekf::GYRO + 2]],
            p_diag,
            m_cmd: m,
            mode,
            status,
        }
    }

    /// Declares a bias fault when the sun or magnetometer innovation stays too large:
    /// inflates the attitude and both bias uncertainties and lets the measurements
    /// re-estimate them. Only in sunlight, where the unbiased sun sensor can arbitrate.
    fn detect_fault(&mut self, sun_ok: bool) -> bool {
        let cfg = &self.cfg;
        let m = &mut self.mekf;
        let nis_sun = m.filter.nis(0);
        let nis_mag = m.filter.nis(1);
        let thr = cfg.fault_nis_threshold;
        let suspect = sun_ok && (nis_sun > thr || nis_mag > thr);
        m.suspect_steps = if suspect { m.suspect_steps + 1 } else { 0 };
        if m.suspect_steps >= cfg.fault_steps {
            m.filter.inflate(
                cfg.fault_attitude_sigma,
                cfg.fault_gyro_bias_sigma,
                cfg.fault_mag_bias_sigma,
            );
            m.suspect_steps = 0;
            return true;
        }
        false
    }

    fn mekf_step(&mut self, input: &Input) -> Output {
        let cfg = self.cfg;
        let mut status = 0;
        self.mekf.filter.clear_update();

        let mag_ref_ok = field_plausible(&input.mag_ref_i);
        let sun_ref_ok = unit_plausible(&input.sun_ref_i);
        let mag_plausible = field_plausible(&input.mag_b);
        let sun_plausible = unit_plausible(&input.sun_b);
        let mag_ok = input.mag_valid != 0 && mag_plausible && mag_ref_ok;
        let sun_ok = input.sun_valid != 0 && sun_plausible && sun_ref_ok;
        let gyro_ok = input.gyro_valid != 0 && gyro_plausible(&input.gyro_b);
        if (input.mag_valid != 0 && !mag_plausible)
            || (input.sun_valid != 0 && !sun_plausible)
            || (input.gyro_valid != 0 && !gyro_ok)
        {
            status |= STATUS_BAD_INPUT;
        }
        let zero = [0.0; 3];
        let mag_ut = if mag_ok {
            scale3(&input.mag_b, UT)
        } else {
            zero
        };
        let mag_ref_ut = if mag_ref_ok {
            scale3(&input.mag_ref_i, UT)
        } else {
            zero
        };
        let sun_b = if sun_ok { input.sun_b } else { zero };
        let sun_ref = if sun_ref_ok { input.sun_ref_i } else { zero };
        let gyro = if gyro_ok {
            input.gyro_b
        } else {
            self.mekf.gyro_prev
        };
        if !(sun_ok && mag_ok) {
            status |= STATUS_NO_ATTITUDE_FIX;
        }

        if !self.initialised {
            if !(sun_ok && mag_ok && gyro_ok) {
                return Output::empty(MODE_OFF, status);
            }
            let body = pair(&sun_b, &div3(&mag_ut, norm(&mag_ut)));
            let reference = pair(&sun_ref, &div3(&mag_ref_ut, norm(&mag_ref_ut)));
            let Ok(res) = flae2(
                &body,
                &reference,
                [cfg.flae_weight_sun, cfg.flae_weight_mag],
                None,
                cfg.flae_tolerance,
                cfg.flae_max_iterations,
            ) else {
                return Output::empty(MODE_OFF, status | STATUS_FLAE_FAILED);
            };
            let mut p0 = [0.0; MEKF_NX];
            for i in 0..3 {
                p0[mekf::ATT + i] = cfg.p0_attitude * cfg.p0_attitude;
                p0[mekf::GYRO + i] = cfg.p0_gyro_bias * cfg.p0_gyro_bias;
                p0[mekf::MAG + i] = cfg.p0_mag_bias * cfg.p0_mag_bias;
            }
            self.mekf.filter.reset(&res.q, &p0);
            self.initialised = true;
            self.mekf.gyro_prev = gyro;
            return self.mekf_finish(input, status, &gyro, &mag_ut, mag_ok);
        }

        // Propagate with the mean of the previous and current rate readings: the body rate
        // changes within a step, and holding the previous reading leaves an O(ω̇ dt²) error.
        let prev = self.mekf.gyro_prev;
        let rate = [
            0.5 * (prev[0] + gyro[0]),
            0.5 * (prev[1] + gyro[1]),
            0.5 * (prev[2] + gyro[2]),
        ];
        if !self.mekf.filter.step(
            &rate,
            cfg.dt,
            &sun_b,
            &sun_ref,
            sun_ok,
            &mag_ut,
            &mag_ref_ut,
            mag_ok,
        ) {
            status |= STATUS_COVARIANCE_RESET;
        }
        if self.detect_fault(sun_ok) {
            status |= STATUS_BIAS_FAULT;
        }
        self.mekf.gyro_prev = gyro;
        self.mekf_finish(input, status, &gyro, &mag_ut, mag_ok)
    }
}
