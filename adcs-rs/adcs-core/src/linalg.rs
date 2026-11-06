//! Fixed-size dense linear algebra, a port of `adcs-c/src/linalg.c`.
//!
//! Matrices are row-major slices with their dimensions passed explicitly, so the same
//! routines serve every size in the filters. Nothing allocates; callers provide outputs and
//! workspace. The operation order matches the C version, so both round the same way.

use crate::real::{Real, abs, sqrt};

/// The matrix was singular or not positive definite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Singular;

#[inline]
fn at(cols: usize, i: usize, j: usize) -> usize {
    i * cols + j
}

pub fn eye(n: usize, dst: &mut [Real]) {
    dst[..n * n].fill(0.0);
    for i in 0..n {
        dst[at(n, i, i)] = 1.0;
    }
}

pub fn diag(n: usize, d: &[Real], dst: &mut [Real]) {
    dst[..n * n].fill(0.0);
    for i in 0..n {
        dst[at(n, i, i)] = d[i];
    }
}

#[must_use]
pub fn all_finite(a: &[Real]) -> bool {
    a.iter().all(|x| x.is_finite())
}

/// `c = a·b`, `a` is m×k, `b` is k×n.
pub fn mul(m: usize, k: usize, n: usize, a: &[Real], b: &[Real], c: &mut [Real]) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[at(k, i, p)] * b[at(n, p, j)];
            }
            c[at(n, i, j)] = s;
        }
    }
}

/// `c = a·bᵀ`, `a` is m×k, `b` is n×k.
pub fn mul_bt(m: usize, k: usize, n: usize, a: &[Real], b: &[Real], c: &mut [Real]) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[at(k, i, p)] * b[at(k, j, p)];
            }
            c[at(n, i, j)] = s;
        }
    }
}

/// `y = a·x`, `a` is m×n.
pub fn mul_vec(m: usize, n: usize, a: &[Real], x: &[Real], y: &mut [Real]) {
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[at(n, i, j)] * x[j];
        }
        y[i] = s;
    }
}

/// `aᵀ` of an m×n matrix.
pub fn transpose(m: usize, n: usize, a: &[Real], at_: &mut [Real]) {
    for i in 0..m {
        for j in 0..n {
            at_[at(m, j, i)] = a[at(n, i, j)];
        }
    }
}

/// `a ← (a + aᵀ) / 2`
pub fn symmetrize(n: usize, a: &mut [Real]) {
    for i in 0..n {
        for j in i + 1..n {
            let s = 0.5 * (a[at(n, i, j)] + a[at(n, j, i)]);
            a[at(n, i, j)] = s;
            a[at(n, j, i)] = s;
        }
    }
}

/// Lower Cholesky factor of an n×n matrix (only its lower triangle is read).
///
/// # Errors
/// [`Singular`] if the matrix is not positive definite.
pub fn cholesky(n: usize, a: &[Real], lower: &mut [Real]) -> Result<(), Singular> {
    lower[..n * n].fill(0.0);
    for j in 0..n {
        let mut d = a[at(n, j, j)];
        for p in 0..j {
            d -= lower[at(n, j, p)] * lower[at(n, j, p)];
        }
        if !(d > 0.0) || !d.is_finite() {
            return Err(Singular);
        }
        let ljj = sqrt(d);
        lower[at(n, j, j)] = ljj;
        for i in j + 1..n {
            let mut s = a[at(n, i, j)];
            for p in 0..j {
                s -= lower[at(n, i, p)] * lower[at(n, j, p)];
            }
            lower[at(n, i, j)] = s / ljj;
        }
    }
    Ok(())
}

/// Solves `a·x = b` for an n×n `a` and n×m `b` by Gaussian elimination with partial
/// pivoting. Both are overwritten; `b` becomes `x`.
///
/// # Errors
/// [`Singular`] if `a` is singular.
pub fn solve(n: usize, m: usize, a: &mut [Real], b: &mut [Real]) -> Result<(), Singular> {
    for k in 0..n {
        let mut piv = k;
        let mut best = abs(a[at(n, k, k)]);
        for i in k + 1..n {
            let v = abs(a[at(n, i, k)]);
            if v > best {
                best = v;
                piv = i;
            }
        }
        if !(best > 0.0) {
            return Err(Singular);
        }
        if piv != k {
            for j in 0..n {
                a.swap(at(n, k, j), at(n, piv, j));
            }
            for j in 0..m {
                b.swap(at(m, k, j), at(m, piv, j));
            }
        }
        for i in k + 1..n {
            let f = a[at(n, i, k)] / a[at(n, k, k)];
            if f == 0.0 {
                continue;
            }
            for j in k..n {
                a[at(n, i, j)] -= f * a[at(n, k, j)];
            }
            for j in 0..m {
                b[at(m, i, j)] -= f * b[at(m, k, j)];
            }
        }
    }
    for kk in (0..n).rev() {
        for j in 0..m {
            let mut s = b[at(m, kk, j)];
            for p in kk + 1..n {
                s -= a[at(n, kk, p)] * b[at(m, p, j)];
            }
            b[at(m, kk, j)] = s / a[at(n, kk, kk)];
        }
    }
    Ok(())
}

/// `x = b·a⁻¹` for an n×n `a` and m×n `b` (right division), solving on the transposes.
/// `work` needs `n*n + n*m` entries.
///
/// # Errors
/// [`Singular`] if `a` is singular.
pub fn right_divide(
    m: usize,
    n: usize,
    b: &[Real],
    a: &[Real],
    x: &mut [Real],
    work: &mut [Real],
) -> Result<(), Singular> {
    let (at_, rest) = work.split_at_mut(n * n);
    let xt = &mut rest[..n * m];
    transpose(n, n, a, at_);
    transpose(m, n, b, xt);
    solve(n, m, at_, xt)?;
    transpose(n, m, xt, x);
    Ok(())
}

/// Inverse of an n×n matrix; `work` needs `n*n` entries.
///
/// # Errors
/// [`Singular`] if `a` is singular.
pub fn inverse(n: usize, a: &[Real], inv: &mut [Real], work: &mut [Real]) -> Result<(), Singular> {
    work[..n * n].copy_from_slice(&a[..n * n]);
    eye(n, inv);
    solve(n, n, work, inv)
}

/// Determinant of an n×n matrix by LU with partial pivoting; `work` needs `n*n` entries.
pub fn det(n: usize, a: &[Real], work: &mut [Real]) -> Real {
    work[..n * n].copy_from_slice(&a[..n * n]);
    let mut d = 1.0;
    for k in 0..n {
        let mut piv = k;
        let mut best = abs(work[at(n, k, k)]);
        for i in k + 1..n {
            let v = abs(work[at(n, i, k)]);
            if v > best {
                best = v;
                piv = i;
            }
        }
        if !(best > 0.0) {
            return 0.0;
        }
        if piv != k {
            for j in 0..n {
                work.swap(at(n, k, j), at(n, piv, j));
            }
            d = -d;
        }
        d *= work[at(n, k, k)];
        for i in k + 1..n {
            let f = work[at(n, i, k)] / work[at(n, k, k)];
            for j in k..n {
                work[at(n, i, j)] -= f * work[at(n, k, j)];
            }
        }
    }
    d
}

#[must_use]
pub fn cross(a: &[Real], b: &[Real]) -> [Real; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[must_use]
pub fn dot(a: &[Real], b: &[Real]) -> Real {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[must_use]
pub fn norm(a: &[Real]) -> Real {
    sqrt(dot(a, a))
}

/// Cross-product matrix `[v×]`, row-major.
#[must_use]
pub fn skew(v: &[Real]) -> [Real; 9] {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::TEST_TOL;

    fn spd5() -> [Real; 25] {
        let mut m = [0.0; 25];
        for (i, x) in m.iter_mut().enumerate() {
            *x = Real::from(u8::try_from(i * 37 % 11).unwrap()) / 7.0 - 5.0 / 7.0;
        }
        let mut p = [0.0; 25];
        mul_bt(5, 5, 5, &m, &m, &mut p);
        for i in 0..5 {
            p[at(5, i, i)] += 1.0;
        }
        p
    }

    #[test]
    fn multiplication() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0; 4];
        mul(2, 3, 2, &a, &b, &mut c);
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
        let mut bt = [0.0; 6];
        transpose(3, 2, &b, &mut bt);
        let mut c2 = [0.0; 4];
        mul_bt(2, 3, 2, &a, &bt, &mut c2);
        assert_eq!(c, c2);
    }

    #[test]
    fn cholesky_reconstructs_and_rejects() {
        let mut p = spd5();
        let mut l = [0.0; 25];
        cholesky(5, &p, &mut l).unwrap();
        let mut llt = [0.0; 25];
        mul_bt(5, 5, 5, &l, &l, &mut llt);
        for (x, y) in llt.iter().zip(p.iter()) {
            assert!((x - y).abs() < TEST_TOL);
        }
        p[at(5, 2, 2)] = -1.0;
        assert_eq!(cholesky(5, &p, &mut l), Err(Singular));
    }

    #[test]
    fn inverse_right_divide_det() {
        let p = spd5();
        let (mut inv, mut work, mut prod) = ([0.0; 25], [0.0; 50], [0.0; 25]);
        inverse(5, &p, &mut inv, &mut work).unwrap();
        mul(5, 5, 5, &p, &inv, &mut prod);
        for i in 0..5 {
            for j in 0..5 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((prod[at(5, i, j)] - e).abs() < TEST_TOL);
            }
        }
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 0.0, 2.0, 0.0, 1.0];
        let mut x = [0.0; 10];
        right_divide(2, 5, &b, &p, &mut x, &mut work).unwrap();
        mul(2, 5, 5, &x, &p, &mut prod);
        for i in 0..10 {
            assert!((prod[i] - b[i]).abs() < TEST_TOL);
        }
        let m3 = [2.0, -1.0, 0.0, 1.0, 3.0, 4.0, 0.0, 5.0, -2.0];
        assert!((det(3, &m3, &mut work) + 54.0).abs() < TEST_TOL);
        let singular = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
        assert_eq!(det(3, &singular, &mut work), 0.0);
        assert_eq!(inverse(3, &singular, &mut inv, &mut work), Err(Singular));
    }
}
