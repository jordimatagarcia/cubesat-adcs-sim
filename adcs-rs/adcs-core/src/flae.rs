//! Fast Linear Attitude Estimator, J. Wu et al. (2018), for two vector pairs. Port of
//! `adcs-c/src/flae.c` (and `src/adcssim/adcs_ref/flae.py`, which documents the two
//! deviations from the thesis text: full-pivoting elimination and normalised weights).

use crate::linalg::{cross, det, norm};
use crate::quat::Quat;
use crate::real::{Real, abs, sqrt};

/// Vector pairs closer to parallel than this (sine of the angle) carry no attitude
/// information.
const MIN_SIN_ANGLE: Real = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlaeResult {
    /// Scalar-last, inertial → body.
    pub q: Quat,
    /// Largest eigenvalue of `W`.
    pub lambda_max: Real,
    pub iterations: u32,
}

/// Degenerate geometry, or Newton's method did not converge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlaeFailed;

/// Non-zero solution of `n x = 0` for a rank-3 4×4 matrix, by Gaussian elimination with
/// full pivoting. `n` is overwritten.
///
/// # Errors
/// [`FlaeFailed`] if the rank is lower than 3.
pub fn null_vector(n: &mut [Real; 16]) -> Result<[Real; 4], FlaeFailed> {
    let mut cols = [0usize, 1, 2, 3];
    let mut scale: Real = 0.0;
    for &x in n.iter() {
        let v = abs(x);
        if v > scale {
            scale = v;
        }
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    for k in 0..3 {
        // largest remaining element, first in row-major order
        let (mut pi, mut pj, mut best) = (k, k, -1.0);
        for i in k..4 {
            for j in k..4 {
                let v = abs(n[i * 4 + j]);
                if v > best {
                    best = v;
                    pi = i;
                    pj = j;
                }
            }
        }
        if !(best >= 1e-9 * scale) {
            return Err(FlaeFailed);
        }
        for j in 0..4 {
            n.swap(k * 4 + j, pi * 4 + j);
        }
        for i in 0..4 {
            n.swap(i * 4 + k, i * 4 + pj);
        }
        cols.swap(k, pj);
        for i in k + 1..4 {
            let f = n[i * 4 + k] / n[k * 4 + k];
            for j in 0..4 {
                n[i * 4 + j] -= f * n[k * 4 + j];
            }
        }
    }
    // Upper-triangular in the first three rows; the last (pivoted) unknown is free.
    let mut x: [Real; 4] = [0.0, 0.0, 0.0, 1.0];
    for kk in (0..3).rev() {
        let mut s = 0.0;
        for p in kk + 1..4 {
            s += n[kk * 4 + p] * x[p];
        }
        x[kk] = -s / n[kk * 4 + kk];
    }
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[cols[i]] = x[i];
    }
    Ok(out)
}

/// Attitude from two observations `b_i = A r_i`; `v_body` and `v_ref` hold the two vectors
/// one after the other. The sign of `q` is chosen closest to `q_prev` when it is given and
/// finite, otherwise with `q4 >= 0`.
///
/// # Errors
/// [`FlaeFailed`] if the geometry is degenerate or Newton's method did not converge.
pub fn flae2(
    v_body: &[Real; 6],
    v_ref: &[Real; 6],
    weights: [Real; 2],
    q_prev: Option<&[Real]>,
    tolerance: Real,
    max_iterations: u32,
) -> Result<FlaeResult, FlaeFailed> {
    let wsum = weights[0] + weights[1];
    if !(wsum > 0.0) {
        return Err(FlaeFailed);
    }
    let a = [weights[0] / wsum, weights[1] / wsum];
    if !(norm(&cross(&v_body[0..3], &v_body[3..6])) >= MIN_SIN_ANGLE) {
        return Err(FlaeFailed);
    }

    // Attitude profile matrix B = Σ a b rᵀ; H_x, H_y, H_z are its columns.
    let mut b = [0.0; 9];
    for j in 0..3 {
        for k in 0..3 {
            b[j * 3 + k] = a[0] * v_body[j] * v_ref[k] + a[1] * v_body[3 + j] * v_ref[3 + k];
        }
    }
    let hx = [b[0], b[3], b[6]];
    let hy = [b[1], b[4], b[7]];
    let hz = [b[2], b[5], b[8]];
    // eq. 105
    #[rustfmt::skip]
    let w: [Real; 16] = [
        hx[0] + hy[1] + hz[2], -hy[2] + hz[1],        -hz[0] + hx[2],        -hx[1] + hy[0],
        -hy[2] + hz[1],        hx[0] - hy[1] - hz[2], hx[1] + hy[0],         hx[2] + hz[0],
        -hz[0] + hx[2],        hx[1] + hy[0],         hy[1] - hx[0] - hz[2], hy[2] + hz[1],
        -hx[1] + hy[0],        hx[2] + hz[0],         hy[2] + hz[1],         hz[2] - hy[1] - hx[0],
    ];
    // eq. 107: det(W − λI) = λ⁴ + τ1 λ² + τ2 λ + τ3
    let mut bb = 0.0;
    for x in b {
        bb += x * x;
    }
    let tau1 = -2.0 * bb;
    let tau2 = 8.0
        * (hx[2] * hy[1] * hz[0] - hx[1] * hy[2] * hz[0] - hx[2] * hy[0] * hz[1]
            + hx[0] * hy[2] * hz[1]
            + hx[1] * hy[0] * hz[2]
            - hx[0] * hy[1] * hz[2]);
    let mut work = [0.0; 16];
    let tau3 = det(4, &w, &mut work);

    // Newton from Σ a |b| |r|, an upper bound of λmax: from above it converges monotonically to
    // the largest root (λ = 1 assumed unit vectors, and could converge to a smaller root).
    let mut lam: Real = a[0] * norm(&v_body[0..3]) * norm(&v_ref[0..3])
        + a[1] * norm(&v_body[3..6]) * norm(&v_ref[3..6]);
    let mut it = 0;
    let mut converged = false;
    while it < max_iterations {
        it += 1;
        let l2 = lam * lam;
        let f = l2 * l2 + tau1 * l2 + tau2 * lam + tau3;
        let df = 4.0 * l2 * lam + 2.0 * tau1 * lam + tau2;
        if df == 0.0 {
            return Err(FlaeFailed);
        }
        let step = f / df;
        lam -= step;
        if abs(step) < tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(FlaeFailed);
    }

    let mut n = w;
    for i in 0..4 {
        n[i * 4 + i] -= lam;
    }
    let v = null_vector(&mut n)?;
    let mut q = [v[1], v[2], v[3], v[0]]; // scalar-first eigenvector → scalar-last
    let qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for x in &mut q {
        *x /= qn;
    }
    let flip = match q_prev {
        Some(p) => {
            p[..4].iter().all(|x| x.is_finite())
                && (q[0] * p[0] + q[1] * p[1] + q[2] * p[2] + q[3] * p[3]) < 0.0
        }
        None => q[3] < 0.0,
    };
    if flip {
        for x in &mut q {
            *x = -*x;
        }
    }
    Ok(FlaeResult {
        q,
        lambda_max: lam,
        iterations: it,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quat::{normalize, rotate};
    use crate::real::{TEST_NEWTON_TOL, TEST_TOL};

    fn observe(q: &[Real], r: &[Real; 6]) -> [Real; 6] {
        let (a, b) = (rotate(q, &r[0..3]), rotate(q, &r[3..6]));
        [a[0], a[1], a[2], b[0], b[1], b[2]]
    }

    #[test]
    fn recovers_attitude_and_follows_previous_sign() {
        let mut q = [0.3, -0.5, 0.1, 0.8];
        normalize(&mut q);
        let mut r = [0.6, 0.8, 0.0, 0.2, -0.3, 0.9];
        let n = norm(&r[3..6]);
        for x in &mut r[3..6] {
            *x /= n;
        }
        let body = observe(&q, &r);
        let res = flae2(&body, &r, [1.0, 1.0], None, TEST_NEWTON_TOL, 20).unwrap();
        for i in 0..4 {
            assert!((res.q[i] - q[i]).abs() < TEST_TOL.max(1e-6));
        }
        assert!((res.lambda_max - 1.0).abs() < TEST_TOL.max(1e-6));
        let prev = [-q[0], -q[1], -q[2], -q[3]];
        let res = flae2(&body, &r, [1.0, 1.0], Some(&prev), TEST_NEWTON_TOL, 20).unwrap();
        assert!((res.q[3] + q[3]).abs() < TEST_TOL.max(1e-6));
    }

    #[test]
    fn scalar_part_near_zero() {
        let q = [0.0, 0.6, 0.8, 0.0];
        let r = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let res = flae2(&observe(&q, &r), &r, [1.0, 1.0], None, TEST_NEWTON_TOL, 20).unwrap();
        let d = res.q[0] * q[0] + res.q[1] * q[1] + res.q[2] * q[2] + res.q[3] * q[3];
        assert!((d.abs() - 1.0).abs() < TEST_TOL.max(1e-6));
    }

    #[test]
    fn rejects_degenerate_input() {
        let parallel = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(flae2(&parallel, &parallel, [1.0, 1.0], None, TEST_NEWTON_TOL, 20).is_err());
        let ok = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(flae2(&ok, &ok, [0.0, 0.0], None, TEST_NEWTON_TOL, 20).is_err());
        assert!(flae2(&ok, &ok, [1.0, 1.0], None, TEST_NEWTON_TOL, 0).is_err());
    }
}
