//! The ADCS: input checks, the thesis (FLAE + two-stage UKF) and MEKF variants, and the
//! mode manager. Port of `adcs-c/src/adcs.c`; the golden vectors check that both match the
//! Python reference.

use crate::abi::{
    Config, Input, MEKF_NX, MODE_OFF, NB, NX, Output, STATUS_BAD_INPUT, STATUS_BIAS_FAULT,
    STATUS_COVARIANCE_RESET, STATUS_FLAE_FAILED, STATUS_NO_ATTITUDE_FIX, STATUS_NOT_INITIALISED,
    STATUS_RATE_PROPAGATED, VARIANT_MEKF, VARIANT_THESIS,
};
use crate::control::Control;
use crate::flae::flae2;
use crate::linalg::{all_finite, cholesky, cross, inverse, mul_vec, norm};
use crate::mekf::{self, Mekf};
use crate::quat::{self, Quat};
use crate::real::{NAN, Real, abs, cos, rint, sin};
use crate::tsukf::{self, Tsukf};

/// Marks an initialised [`Adcs`]; the first field of its `repr(C)` layout.
pub const MAGIC: u32 = 0x4144_4353; // "ADCS"
const _: () = assert!(core::mem::offset_of!(Adcs, magic) == 0);
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

// Configuration rules, the same in every implementation (config_problems in
// src/adcssim/backends/config.py; docs/adcs.md, "Configuration rules").
const MAX_DT: Real = 10.0;
/// Bounds the RK4 steps per filter step, and so its run time.
const MAX_PROCESS_STEPS: Real = 1000.0;
const MAX_FLAE_ITERATIONS: u32 = 1000;
/// Below this the unscented-transform weights lose precision.
const ALPHA_MIN: Real = 1e-2;
const ALPHA_MAX: Real = 1e2;
const MAX_BDOT_GAIN: Real = 1e9;
const MAX_PD_GAIN: Real = 1e3;

fn in_range(x: Real, lo: Real, hi: Real) -> bool {
    x.is_finite() && x >= lo && x <= hi
}

fn inertia_valid(m: &[Real; 9]) -> bool {
    if !all_finite(m) {
        return false;
    }
    let (mut scale, mut asym): (Real, Real) = (0.0, 0.0);
    for x in m {
        scale = if abs(*x) > scale { abs(*x) } else { scale };
    }
    for i in 0..3 {
        for j in 0..3 {
            let d = abs(m[i * 3 + j] - m[j * 3 + i]);
            asym = if d > asym { d } else { asym };
        }
    }
    let mut lower = [0.0; 9];
    asym <= 1e-9 * scale && cholesky(3, m, &mut lower).is_ok()
}

fn config_valid(c: &Config) -> bool {
    let wsum = c.flae_weight_sun + c.flae_weight_mag;
    let qn = quat::norm(&c.q_target);
    let common = in_range(c.dt, 0.0, MAX_DT)
        && c.dt > 0.0
        && inertia_valid(&c.inertia)
        && all_positive(&c.max_dipole)
        && all_nonnegative(&[c.flae_weight_sun, c.flae_weight_mag])
        && all_positive(&[wsum, c.flae_tolerance])
        && (1..=MAX_FLAE_ITERATIONS).contains(&c.flae_max_iterations)
        && in_range(c.bdot_gain, 0.0, MAX_BDOT_GAIN)
        && in_range(c.kp, 0.0, MAX_PD_GAIN)
        && in_range(c.kd, 0.0, MAX_PD_GAIN)
        && all_nonnegative(&[c.detumble_exit_rate, c.settle_time])
        && in_range(c.detumble_enter_rate, c.detumble_exit_rate, Real::INFINITY)
        && all_finite(&c.q_target)
        && abs(qn - 1.0) <= 1e-6;
    if !common {
        return false;
    }
    match c.variant {
        VARIANT_THESIS => {
            all_positive(&[c.process_dt])
                && c.dt / c.process_dt <= MAX_PROCESS_STEPS
                && in_range(c.ukf_alpha, ALPHA_MIN, ALPHA_MAX)
                && all_positive(&c.p_x0)
                && all_positive(&c.p_b0)
                && all_positive(&c.v)
                && all_nonnegative(&c.w_x)
                && all_nonnegative(&c.w_b)
        }
        VARIANT_MEKF => {
            all_positive(&[
                c.sigma_gyro_noise,
                c.sigma_mag_noise,
                c.sigma_sun_noise,
                c.p0_attitude,
                c.p0_gyro_bias,
                c.p0_mag_bias,
                c.fault_nis_threshold,
                c.fault_gyro_bias_sigma,
                c.fault_mag_bias_sigma,
                c.fault_attitude_sigma,
            ]) && all_nonnegative(&[
                c.sigma_gyro_bias_walk,
                c.sigma_mag_bias_walk,
                c.sigma_outage_accel,
            ]) && c.fault_steps >= 1
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
    inertia: [Real; 9],
    inertia_inv: [Real; 9],
    /// s since the gyro became unavailable, 0 while it works
    outage_time: Real,
}

/// The outage growth stops once an attitude variance reaches this σ² (rad): beyond ~15° the
/// attitude is effectively unknown, and more would only break the linearisation.
const OUTAGE_MAX_ATTITUDE_SIGMA: Real = 0.25;

/// Torque-free Euler equations, ω̇ = I⁻¹ (−ω × Iω).
fn euler_rate(w: &[Real; 3], inertia: &[Real; 9], inertia_inv: &[Real; 9]) -> [Real; 3] {
    let mut iw = [0.0; 3];
    mul_vec(3, 3, inertia, w, &mut iw);
    let g = cross(&[-w[0], -w[1], -w[2]], &iw);
    let mut out = [0.0; 3];
    mul_vec(3, 3, inertia_inv, &g, &mut out);
    out
}

/// Body rate after `dt` without torques (one RK4 step).
fn propagate_rate(w: &[Real; 3], dt: Real, inertia: &[Real; 9], inv: &[Real; 9]) -> [Real; 3] {
    let k1 = euler_rate(w, inertia, inv);
    let xt = [0, 1, 2].map(|i| w[i] + 0.5 * dt * k1[i]);
    let k2 = euler_rate(&xt, inertia, inv);
    let xt = [0, 1, 2].map(|i| w[i] + 0.5 * dt * k2[i]);
    let k3 = euler_rate(&xt, inertia, inv);
    let xt = [0, 1, 2].map(|i| w[i] + dt * k3[i]);
    let k4 = euler_rate(&xt, inertia, inv);
    [0, 1, 2].map(|i| w[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
}

/// The whole ADCS state. Large (about 42 KB in double precision) and with no heap use:
/// place it in static memory, or let the caller of the C interface allocate it.
///
/// `repr(C)` with the magic number first, so the C interface can read it through a raw
/// pointer before treating the memory as an `Adcs`.
#[repr(C)]
#[derive(Clone, Debug)]
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
                inertia: [0.0; 9],
                inertia_inv: [0.0; 9],
                outage_time: 0.0,
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
            m.outage_time = 0.0;
            m.inertia = cfg.inertia;
            let mut work = [0.0; 9];
            inverse(3, &cfg.inertia, &mut m.inertia_inv, &mut work).map_err(|_| InvalidConfig)?;
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
            None,
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
        // Directions are normalised: the plausibility check accepts lengths 0.5–2.
        let sun_b = if sun_ok {
            div3(&input.sun_b, norm(&input.sun_b))
        } else {
            zero
        };
        let sun_ref = if sun_ref_ok {
            div3(&input.sun_ref_i, norm(&input.sun_ref_i))
        } else {
            zero
        };
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
                let reference = pair(&sun_ref, &div3(&mag_ref_ut, norm(&mag_ref_ut)));
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

    /// Control and output. `mag_ut`/`mag_ok`: the magnetometer reading, usable for B-dot
    /// whether or not the field model is available.
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
        // B-dot from Ḃ = −ω̂ × B: a fixed inertial field seen from the rotating body.
        let c = cross(&w, &b_body);
        let b_dot = [-c[0], -c[1], -c[2]];
        let (mode, m) = self
            .control
            .update(&cfg, &q_ref, &w, &b_body, Some(&b_dot), &mut status);
        let rate_outage = cfg.sigma_outage_accel * self.mekf.outage_time;
        let f = &self.mekf.filter;
        let mut p_diag = [NAN; NX + NB];
        for i in 0..3 {
            let pg = f.p[(mekf::GYRO + i) * MEKF_NX + mekf::GYRO + i];
            p_diag[i] = f.p[(mekf::ATT + i) * MEKF_NX + mekf::ATT + i]; // rad²
            // rate = gyro − bias (+ the propagation uncertainty during a gyro outage)
            p_diag[10 + i] =
                cfg.sigma_gyro_noise * cfg.sigma_gyro_noise + pg + rate_outage * rate_outage;
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

    /// One step without the gyro: the rate propagated with the dynamics (as a gyro reading,
    /// bias included) and the attitude variance to add for what they leave out, capped.
    fn gyro_outage(&mut self, dt: Real) -> ([Real; 3], Real) {
        let m = &mut self.mekf;
        let b = &m.filter.x[mekf::GYRO..mekf::GYRO + 3];
        let w_prev = [0, 1, 2].map(|i| m.gyro_prev[i] - b[i]);
        let w_new = propagate_rate(&w_prev, dt, &m.inertia, &m.inertia_inv);
        let gyro = [0, 1, 2].map(|i| w_new[i] + b[i]);
        let t0 = m.outage_time;
        m.outage_time = t0 + dt;
        let (t1, sa) = (m.outage_time, self.cfg.sigma_outage_accel);
        // rate error σ_α t → attitude error σ_α t²/2 → variance (σ_α²/4) t⁴
        let var = 0.25 * sa * sa * (t1 * t1 * t1 * t1 - t0 * t0 * t0 * t0);
        let p = &m.filter.p;
        let mut largest = p[0];
        for i in 1..3 {
            let v = p[i * MEKF_NX + i];
            largest = if v > largest { v } else { largest };
        }
        let room = OUTAGE_MAX_ATTITUDE_SIGMA * OUTAGE_MAX_ATTITUDE_SIGMA - largest;
        let cap = if room > 0.0 { room } else { 0.0 };
        (gyro, if var < cap { var } else { cap })
    }

    #[allow(clippy::too_many_lines)]
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
        // The magnetometer reading itself (for B-dot) needs no reference model.
        let mag_reading_ok = input.mag_valid != 0 && mag_plausible;
        let mag_ut = if mag_ok {
            scale3(&input.mag_b, UT)
        } else {
            zero
        };
        let mag_meas_ut = if mag_reading_ok {
            scale3(&input.mag_b, UT)
        } else {
            zero
        };
        let mag_ref_ut = if mag_ref_ok {
            scale3(&input.mag_ref_i, UT)
        } else {
            zero
        };
        // Directions are normalised: the plausibility check accepts lengths 0.5–2.
        let sun_b = if sun_ok {
            div3(&input.sun_b, norm(&input.sun_b))
        } else {
            zero
        };
        let sun_ref = if sun_ref_ok {
            div3(&input.sun_ref_i, norm(&input.sun_ref_i))
        } else {
            zero
        };
        let mut gyro = if gyro_ok {
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
            return self.mekf_finish(input, status, &gyro, &mag_meas_ut, mag_reading_ok);
        }

        // Without the gyro, propagate the rate with the dynamics, grow the attitude
        // uncertainty with what they leave out, and freeze the (unobservable) gyro bias.
        let dt = cfg.dt;
        let mut outage_var: Real = 0.0;
        if gyro_ok {
            self.mekf.outage_time = 0.0;
        } else {
            (gyro, outage_var) = self.gyro_outage(dt);
            status |= STATUS_RATE_PROPAGATED;
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
            dt,
            &sun_b,
            &sun_ref,
            sun_ok,
            &mag_ut,
            &mag_ref_ut,
            mag_ok,
            outage_var,
            !gyro_ok,
        ) {
            status |= STATUS_COVARIANCE_RESET;
        }
        if self.detect_fault(sun_ok) {
            status |= STATUS_BIAS_FAULT;
        }
        self.mekf.gyro_prev = gyro;
        self.mekf_finish(input, status, &gyro, &mag_meas_ut, mag_reading_ok)
    }
}
