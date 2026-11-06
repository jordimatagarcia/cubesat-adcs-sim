//! Two-stage unscented Kalman filter, Chen et al. as derived in the thesis (§2.4.3,
//! eqs. 49–68). Port of `adcs-c/src/tsukf.c`, specialised to the thesis ADCS: state `x`
//! (13), bias `b` (6), measurement `y = h(x) + F_s b` with `h(x) = x`, `F_a = 0`, and `F_s`
//! mapping the magnetometer bias onto B and the gyro bias onto ω.
//!
//! Every temporary lives in the struct, so the filter uses a fixed amount of memory and
//! little stack.

use crate::abi::{NB, NX};
use crate::linalg::{
    Singular, all_finite, cholesky, diag, inverse, mul, mul_bt, mul_vec, right_divide, symmetrize,
};
use crate::quat;
use crate::real::{EPS, NAN, Real, TINY, abs, sqrt};

pub const NY: usize = NX;
pub const NS: usize = 2 * NX + 1;

pub const Q: usize = 0;
pub const SUN: usize = 4;
pub const MAG: usize = 7;
pub const W: usize = 10;

const UNAVAILABLE_VARIANCE: Real = 1e12;
const JITTER: [Real; 4] = [0.0, 1e-12, 1e-10, 1e-8];

#[derive(Clone, Debug, PartialEq)]
struct Work {
    lower: [Real; NX * NX],
    chi: [Real; NS * NX],
    diff: [Real; NX * NX],
    m: [Real; NX * NX],
    n: [Real; NX * NX],
    r: [Real; NX * NB],
    h: [Real; NY * NB],
    beta_pred: [Real; NX * NB],
    beta_upd: [Real; NX * NB],
    t1: [Real; NX * NB],
    t2: [Real; NX * NB],
    px_pred: [Real; NX * NX],
    px_t: [Real; NX * NX],
    px_upd: [Real; NX * NX],
    py_t: [Real; NY * NY],
    p_y: [Real; NY * NY],
    k_t: [Real; NX * NY],
    k_b: [Real; NB * NY],
    p_by: [Real; NB * NY],
    a: [Real; NX * NX],
    b: [Real; NX * NX],
    c: [Real; NX * NX],
    pb_pred: [Real; NB * NB],
    pb_inv: [Real; NB * NB],
    pb_upd: [Real; NB * NB],
    six_a: [Real; NB * NB],
    nb_ny: [Real; NB * NY],
    solve: [Real; 2 * NX * NX],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tsukf {
    // weights of the unscented transform
    xi: Real,
    wm0: Real,
    wc0: Real,
    wi: Real,
    w_x: [Real; NX],
    w_b: [Real; NB],
    v: [Real; NY],
    px0: [Real; NX],
    pb0: [Real; NB],
    pub x: [Real; NX],
    pub px: [Real; NX * NX],
    pub b: [Real; NB],
    pub pb: [Real; NB * NB],
    beta: [Real; NX * NB],
    work: Work,
}

/// Lower Cholesky factor of `p + εI`, retrying with more jitter if needed.
fn robust_cholesky(p: &[Real], lower: &mut [Real], tmp: &mut [Real]) -> Result<(), Singular> {
    let mut scale = TINY;
    for i in 0..NX {
        let d = abs(p[i * NX + i]);
        if d > scale {
            scale = d;
        }
    }
    for jitter in JITTER {
        tmp[..NX * NX].copy_from_slice(&p[..NX * NX]);
        for i in 0..NX {
            tmp[i * NX + i] += EPS + jitter * scale;
        }
        if cholesky(NX, tmp, lower).is_ok() {
            return Ok(());
        }
    }
    Err(Singular)
}

/// eqs. 50/57: `Σ_i [g(χ_i) − g(χ_{i+n})] e_iᵀ / (2ξ) · L⁻¹`
fn transform(
    xi: Real,
    out: &[Real],
    lower: &[Real],
    diff: &mut [Real],
    m: &mut [Real],
    solve: &mut [Real],
) -> Result<(), Singular> {
    for i in 0..NX {
        for j in 0..NX {
            diff[j * NX + i] = out[(1 + i) * NX + j] - out[(1 + NX + i) * NX + j];
        }
    }
    right_divide(NX, NX, diff, lower, m, solve)?;
    let d = 2.0 * xi;
    for x in &mut m[..NX * NX] {
        *x /= d;
    }
    Ok(())
}

/// `y += F_s b`: magnetometer bias onto B, gyro bias onto ω
fn add_fs_b(y: &mut [Real; NY], b: &[Real; NB]) {
    for i in 0..3 {
        y[MAG + i] += b[i];
        y[W + i] += b[3 + i];
    }
}

/// A zero or overflowed norm gives NaN, which the finiteness check then catches.
fn normalize_q(x: &mut [Real; NX]) {
    let mut n = quat::norm(&x[Q..Q + 4]);
    if !(n > 0.0) || !n.is_finite() {
        n = NAN;
    }
    for v in &mut x[Q..Q + 4] {
        *v /= n;
    }
}

fn normalize_q_sun(x: &mut [Real; NX]) {
    normalize_q(x);
    let n = crate::linalg::norm(&x[SUN..SUN + 3]);
    if n > 0.0 {
        for v in &mut x[SUN..SUN + 3] {
            *v /= n;
        }
    }
}

/// `c = a·b·aᵀ` for `a` (r×k) and `b` (k×k); `t` is r×k scratch.
fn sandwich(r: usize, k: usize, a: &[Real], b: &[Real], t: &mut [Real], c: &mut [Real]) {
    mul(r, k, k, a, b, t);
    mul_bt(r, k, r, t, a, c);
}

impl Tsukf {
    pub const ZERO: Self = Self {
        xi: 0.0,
        wm0: 0.0,
        wc0: 0.0,
        wi: 0.0,
        w_x: [0.0; NX],
        w_b: [0.0; NB],
        v: [0.0; NY],
        px0: [0.0; NX],
        pb0: [0.0; NB],
        x: [0.0; NX],
        px: [0.0; NX * NX],
        b: [0.0; NB],
        pb: [0.0; NB * NB],
        beta: [0.0; NX * NB],
        work: Work {
            lower: [0.0; NX * NX],
            chi: [0.0; NS * NX],
            diff: [0.0; NX * NX],
            m: [0.0; NX * NX],
            n: [0.0; NX * NX],
            r: [0.0; NX * NB],
            h: [0.0; NY * NB],
            beta_pred: [0.0; NX * NB],
            beta_upd: [0.0; NX * NB],
            t1: [0.0; NX * NB],
            t2: [0.0; NX * NB],
            px_pred: [0.0; NX * NX],
            px_t: [0.0; NX * NX],
            px_upd: [0.0; NX * NX],
            py_t: [0.0; NY * NY],
            p_y: [0.0; NY * NY],
            k_t: [0.0; NX * NY],
            k_b: [0.0; NB * NY],
            p_by: [0.0; NB * NY],
            a: [0.0; NX * NX],
            b: [0.0; NX * NX],
            c: [0.0; NX * NX],
            pb_pred: [0.0; NB * NB],
            pb_inv: [0.0; NB * NB],
            pb_upd: [0.0; NB * NB],
            six_a: [0.0; NB * NB],
            nb_ny: [0.0; NB * NY],
            solve: [0.0; 2 * NX * NX],
        },
    };

    pub fn configure(&mut self, alpha: Real, w_x: &[Real; NX], w_b: &[Real; NB], v: &[Real; NY]) {
        // eq. 35 with κ = 3 − n, so n + λ = 3α² (thesis appendix), β = 2
        #[allow(clippy::cast_precision_loss)]
        let n = NX as Real;
        let lam = 3.0 * alpha * alpha - n;
        self.wm0 = lam / (n + lam);
        self.wc0 = self.wm0 + 1.0 - alpha * alpha + 2.0;
        self.wi = 1.0 / (2.0 * (n + lam));
        self.xi = sqrt(n + lam);
        self.w_x = *w_x;
        self.w_b = *w_b;
        self.v = *v;
    }

    pub fn reset(&mut self, x0: &[Real; NX], px0: &[Real; NX], b0: &[Real; NB], pb0: &[Real; NB]) {
        self.x = *x0;
        self.b = *b0;
        self.px0 = *px0;
        self.pb0 = *pb0;
        diag(NX, px0, &mut self.px);
        diag(NB, pb0, &mut self.pb);
        self.beta = [0.0; NX * NB];
    }

    /// Restarts from the initial covariances (the estimate is kept).
    fn reset_covariance(&mut self) -> bool {
        diag(NX, &self.px0, &mut self.px);
        diag(NB, &self.pb0, &mut self.pb);
        self.beta = [0.0; NX * NB];
        false
    }

    fn sigma_points(&mut self, x: &[Real; NX]) {
        let w = &mut self.work;
        w.chi[..NX].copy_from_slice(x);
        for i in 0..NX {
            for j in 0..NX {
                let s = self.xi * w.lower[j * NX + i];
                w.chi[(1 + i) * NX + j] = x[j] + s;
                w.chi[(1 + NX + i) * NX + j] = x[j] - s;
            }
        }
    }

    fn ut_mean(&self) -> [Real; NX] {
        let out = &self.work.chi;
        let mut mean = [0.0; NX];
        for j in 0..NX {
            let mut s = out[NX + j];
            for i in 2..NS {
                s += out[i * NX + j];
            }
            mean[j] = self.wm0 * out[j] + self.wi * s;
        }
        mean
    }

    /// `wc0 d0 d0ᵀ + wi Σ dᵢ dᵢᵀ` with `dᵢ = χᵢ − mean`, into `p`.
    fn ut_cov(xs: &[Real], mean: &[Real; NX], wc0: Real, wi: Real, p: &mut [Real]) {
        for a in 0..NX {
            for b in a..NX {
                let mut s = 0.0;
                for i in 1..NS {
                    s += (xs[i * NX + a] - mean[a]) * (xs[i * NX + b] - mean[b]);
                }
                let d0 = (xs[a] - mean[a]) * (xs[b] - mean[b]);
                let v = wc0 * d0 + wi * s;
                p[a * NX + b] = v;
                p[b * NX + a] = v;
            }
        }
    }

    /// One predict + update. Unavailable measurements get an effectively infinite variance
    /// and the predicted value. `q` is renormalised after the prediction, `q` and the sun
    /// vector at the output. Returns false if a covariance had to be reset.
    #[allow(clippy::too_many_lines)]
    pub fn step<F: Fn(&mut [Real])>(
        &mut self,
        process: F,
        y: &[Real; NY],
        available: &[bool; NY],
    ) -> bool {
        let mut ok = true;

        // ---- prediction ---------------------------------------------------------------
        if robust_cholesky(&self.px, &mut self.work.lower, &mut self.work.a).is_err() {
            diag(NX, &self.px0, &mut self.px);
            ok = false;
            let _ = cholesky(NX, &self.px, &mut self.work.lower);
        }
        let x = self.x;
        self.sigma_points(&x);
        for row in self.work.chi.as_chunks_mut::<NX>().0 {
            process(row);
        }
        let w = &mut self.work;
        if transform(
            self.xi,
            &w.chi,
            &w.lower,
            &mut w.diff,
            &mut w.m,
            &mut w.solve,
        )
        .is_err()
        {
            return self.reset_covariance(); // eq. 50
        }
        mul(NX, NX, NB, &w.m, &self.beta, &mut w.r); // eq. 51 (F_a = 0)
        let mk = self.ut_mean(); // eq. 52
        let w = &mut self.work;
        let mut x_pred = mk; // eq. 53
        normalize_q(&mut x_pred);
        // eq. 54: P = cov − M β Pb βᵀ Mᵀ + R Pb Rᵀ + W_x
        Self::ut_cov(&w.chi, &mk, self.wc0, self.wi, &mut w.px_pred);
        mul(NX, NB, NB, &w.r, &self.pb, &mut w.t1); // M β Pb = R Pb
        mul_bt(NX, NB, NX, &w.t1, &self.beta, &mut w.a);
        mul_bt(NX, NX, NX, &w.a, &w.m, &mut w.b);
        mul_bt(NX, NB, NX, &w.t1, &w.r, &mut w.c);
        for i in 0..NX * NX {
            w.px_pred[i] = w.px_pred[i] - w.b[i] + w.c[i];
        }
        for i in 0..NX {
            w.px_pred[i * NX + i] += self.w_x[i];
        }
        w.pb_pred = self.pb; // eq. 56
        for i in 0..NB {
            w.pb_pred[i * NB + i] += self.w_b[i];
        }
        if inverse(NB, &w.pb_pred, &mut w.pb_inv, &mut w.six_a).is_err() {
            return self.reset_covariance();
        }
        mul(NX, NB, NB, &w.t1, &w.pb_inv, &mut w.beta_pred); // eq. 55
        let b_pred = self.b;

        // ---- measurement sigma points -------------------------------------------------
        symmetrize(NX, &mut w.px_pred);
        if robust_cholesky(&w.px_pred, &mut w.lower, &mut w.a).is_err() {
            diag(NX, &self.px0, &mut w.px_pred);
            ok = false;
            let _ = cholesky(NX, &w.px_pred, &mut w.lower);
        }
        self.sigma_points(&x_pred); // h(x) = x
        let w = &mut self.work;
        if transform(
            self.xi,
            &w.chi,
            &w.lower,
            &mut w.diff,
            &mut w.n,
            &mut w.solve,
        )
        .is_err()
        {
            return self.reset_covariance(); // eq. 57
        }
        mul(NY, NX, NB, &w.n, &w.beta_pred, &mut w.h); // eq. 58
        for i in 0..3 {
            w.h[(MAG + i) * NB + i] += 1.0;
            w.h[(W + i) * NB + 3 + i] += 1.0;
        }
        let nk = self.ut_mean(); // eq. 59
        let w = &mut self.work;

        let mut y_b = nk;
        add_fs_b(&mut y_b, &b_pred);
        let mut y_eff = [0.0; NY];
        for i in 0..NY {
            y_eff[i] = if available[i] { y[i] } else { y_b[i] };
        }

        // ---- bias-free filter (eq. 60) ------------------------------------------------
        let (mut x_t, mut y_t, mut tmpv) = ([0.0; NX], [0.0; NY], [0.0; NX]);
        mul_vec(NX, NB, &w.beta_pred, &b_pred, &mut tmpv);
        for i in 0..NX {
            x_t[i] = x_pred[i] - tmpv[i];
        }
        sandwich(NX, NB, &w.beta_pred, &w.pb_pred, &mut w.t1, &mut w.a);
        for i in 0..NX * NX {
            w.px_t[i] = w.px_pred[i] - w.a[i];
        }
        mul(NY, NX, NB, &w.n, &w.beta_pred, &mut w.t2); // N β
        mul_vec(NY, NB, &w.t2, &b_pred, &mut tmpv);
        for i in 0..NY {
            y_t[i] = nk[i] - tmpv[i];
        }
        Self::ut_cov(&w.chi, &nk, self.wc0, self.wi, &mut w.py_t);
        mul(NY, NB, NB, &w.t2, &w.pb_pred, &mut w.t1);
        mul_bt(NY, NB, NX, &w.t1, &w.beta_pred, &mut w.a);
        mul_bt(NY, NX, NY, &w.a, &w.n, &mut w.b);
        for i in 0..NY * NY {
            w.py_t[i] -= w.b[i];
        }
        for i in 0..NY {
            w.py_t[i * NY + i] += if available[i] {
                self.v[i]
            } else {
                UNAVAILABLE_VARIANCE
            };
        }
        mul_bt(NX, NX, NY, &w.px_t, &w.n, &mut w.a); // Px Nᵀ
        if right_divide(NX, NY, &w.a, &w.py_t, &mut w.k_t, &mut w.solve).is_err() {
            return self.reset_covariance();
        }
        let w = &mut self.work;
        mul(NX, NY, NB, &w.k_t, &w.h, &mut w.t1); // eq. 61
        for i in 0..NX * NB {
            w.beta_upd[i] = w.beta_pred[i] - w.t1[i];
        }
        let (mut e, mut x_upd) = ([0.0; NY], [0.0; NX]);
        for i in 0..NY {
            e[i] = y_eff[i] - y_t[i];
        }
        mul_vec(NX, NY, &w.k_t, &e, &mut tmpv); // eq. 62
        for i in 0..NX {
            x_upd[i] = x_t[i] + tmpv[i];
        }
        sandwich(NX, NY, &w.k_t, &w.py_t, &mut w.b, &mut w.a); // eq. 63
        for i in 0..NX * NX {
            w.px_upd[i] = w.px_t[i] - w.a[i];
        }
        symmetrize(NX, &mut w.px_upd);

        // ---- bias filter (eqs. 64–66) -------------------------------------------------
        mul_bt(NB, NB, NY, &w.pb_pred, &w.h, &mut w.p_by);
        sandwich(NY, NB, &w.h, &w.pb_pred, &mut w.t1, &mut w.a);
        for i in 0..NY * NY {
            w.p_y[i] = w.py_t[i] + w.a[i];
        }
        if right_divide(NB, NY, &w.p_by, &w.p_y, &mut w.k_b, &mut w.solve).is_err() {
            return self.reset_covariance();
        }
        let w = &mut self.work;
        let mut b_upd = [0.0; NB];
        for i in 0..NY {
            e[i] = y_eff[i] - y_b[i];
        }
        mul_vec(NB, NY, &w.k_b, &e, &mut b_upd);
        for i in 0..NB {
            b_upd[i] += b_pred[i];
        }
        mul(NB, NY, NY, &w.k_b, &w.p_y, &mut w.nb_ny);
        mul_bt(NB, NY, NB, &w.nb_ny, &w.k_b, &mut w.six_a);
        for i in 0..NB * NB {
            w.pb_upd[i] = w.pb_pred[i] - w.six_a[i];
        }
        symmetrize(NB, &mut w.pb_upd);

        // ---- coupled result (eqs. 67–68) ----------------------------------------------
        mul_vec(NX, NB, &w.beta_upd, &b_upd, &mut tmpv);
        for i in 0..NX {
            x_upd[i] += tmpv[i];
        }
        sandwich(NX, NB, &w.beta_upd, &w.pb_upd, &mut w.t1, &mut w.a);
        for i in 0..NX * NX {
            w.px_upd[i] += w.a[i];
        }
        symmetrize(NX, &mut w.px_upd);
        normalize_q_sun(&mut x_upd);

        if !all_finite(&x_upd) || !all_finite(&w.px_upd) {
            return self.reset_covariance();
        }
        self.x = x_upd;
        self.px = w.px_upd;
        self.b = b_upd;
        self.pb = w.pb_upd;
        self.beta = w.beta_upd;
        ok
    }
}
