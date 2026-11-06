#include "linalg.h"

#include <math.h>

void la_copy(size_t n, adcs_real *dst, const adcs_real *src)
{
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void la_zero(size_t n, adcs_real *dst)
{
    for (size_t i = 0; i < n; i++) {
        dst[i] = (adcs_real)0;
    }
}

void la_eye(size_t n, adcs_real *dst)
{
    la_zero(n * n, dst);
    for (size_t i = 0; i < n; i++) {
        LA_AT(dst, n, i, i) = (adcs_real)1;
    }
}

int la_all_finite(size_t n, const adcs_real *a)
{
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(a[i])) {
            return 0;
        }
    }
    return 1;
}

void la_mul(size_t m, size_t k, size_t n, const adcs_real *a, const adcs_real *b, adcs_real *c)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            adcs_real s = (adcs_real)0;
            for (size_t p = 0; p < k; p++) {
                s += LA_AT(a, k, i, p) * LA_AT(b, n, p, j);
            }
            LA_AT(c, n, i, j) = s;
        }
    }
}

void la_mul_bt(size_t m, size_t k, size_t n, const adcs_real *a, const adcs_real *b,
               adcs_real *c)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            adcs_real s = (adcs_real)0;
            for (size_t p = 0; p < k; p++) {
                s += LA_AT(a, k, i, p) * LA_AT(b, k, j, p);
            }
            LA_AT(c, n, i, j) = s;
        }
    }
}

void la_mul_vec(size_t m, size_t n, const adcs_real *a, const adcs_real *x, adcs_real *y)
{
    for (size_t i = 0; i < m; i++) {
        adcs_real s = (adcs_real)0;
        for (size_t j = 0; j < n; j++) {
            s += LA_AT(a, n, i, j) * x[j];
        }
        y[i] = s;
    }
}

void la_transpose(size_t m, size_t n, const adcs_real *a, adcs_real *at)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            LA_AT(at, m, j, i) = LA_AT(a, n, i, j);
        }
    }
}

void la_symmetrize(size_t n, adcs_real *a)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            const adcs_real s = (adcs_real)0.5 * (LA_AT(a, n, i, j) + LA_AT(a, n, j, i));
            LA_AT(a, n, i, j) = s;
            LA_AT(a, n, j, i) = s;
        }
    }
}

int la_cholesky(size_t n, const adcs_real *a, adcs_real *lower)
{
    la_zero(n * n, lower);
    for (size_t j = 0; j < n; j++) {
        adcs_real d = LA_AT(a, n, j, j);
        for (size_t p = 0; p < j; p++) {
            d -= LA_AT(lower, n, j, p) * LA_AT(lower, n, j, p);
        }
        if (!(d > (adcs_real)0) || !isfinite(d)) {
            return -1;
        }
        const adcs_real ljj = LA_SQRT(d);
        LA_AT(lower, n, j, j) = ljj;
        for (size_t i = j + 1; i < n; i++) {
            adcs_real s = LA_AT(a, n, i, j);
            for (size_t p = 0; p < j; p++) {
                s -= LA_AT(lower, n, i, p) * LA_AT(lower, n, j, p);
            }
            LA_AT(lower, n, i, j) = s / ljj;
        }
    }
    return 0;
}

int la_solve(size_t n, size_t m, adcs_real *a, adcs_real *b)
{
    for (size_t k = 0; k < n; k++) {
        size_t piv = k;
        adcs_real best = LA_FABS(LA_AT(a, n, k, k));
        for (size_t i = k + 1; i < n; i++) {
            const adcs_real v = LA_FABS(LA_AT(a, n, i, k));
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (!(best > (adcs_real)0)) {
            return -1;
        }
        if (piv != k) {
            for (size_t j = 0; j < n; j++) {
                const adcs_real t = LA_AT(a, n, k, j);
                LA_AT(a, n, k, j) = LA_AT(a, n, piv, j);
                LA_AT(a, n, piv, j) = t;
            }
            for (size_t j = 0; j < m; j++) {
                const adcs_real t = LA_AT(b, m, k, j);
                LA_AT(b, m, k, j) = LA_AT(b, m, piv, j);
                LA_AT(b, m, piv, j) = t;
            }
        }
        for (size_t i = k + 1; i < n; i++) {
            const adcs_real f = LA_AT(a, n, i, k) / LA_AT(a, n, k, k);
            if (f == (adcs_real)0) {
                continue;
            }
            for (size_t j = k; j < n; j++) {
                LA_AT(a, n, i, j) -= f * LA_AT(a, n, k, j);
            }
            for (size_t j = 0; j < m; j++) {
                LA_AT(b, m, i, j) -= f * LA_AT(b, m, k, j);
            }
        }
    }
    for (size_t kk = n; kk-- > 0;) {
        for (size_t j = 0; j < m; j++) {
            adcs_real s = LA_AT(b, m, kk, j);
            for (size_t p = kk + 1; p < n; p++) {
                s -= LA_AT(a, n, kk, p) * LA_AT(b, m, p, j);
            }
            LA_AT(b, m, kk, j) = s / LA_AT(a, n, kk, kk);
        }
    }
    return 0;
}

int la_right_divide(size_t m, size_t n, const adcs_real *b, const adcs_real *a, adcs_real *x,
                    adcs_real *work)
{
    /* x·a = b  ⇔  aᵀ·xᵀ = bᵀ */
    adcs_real *at = work;
    adcs_real *xt = work + n * n;
    la_transpose(n, n, a, at);
    la_transpose(m, n, b, xt);
    if (la_solve(n, m, at, xt) != 0) {
        return -1;
    }
    la_transpose(n, m, xt, x);
    return 0;
}

int la_inverse(size_t n, const adcs_real *a, adcs_real *inv, adcs_real *work)
{
    la_copy(n * n, work, a);
    la_eye(n, inv);
    return la_solve(n, n, work, inv);
}

adcs_real la_det(size_t n, const adcs_real *a, adcs_real *work)
{
    la_copy(n * n, work, a);
    adcs_real det = (adcs_real)1;
    for (size_t k = 0; k < n; k++) {
        size_t piv = k;
        adcs_real best = LA_FABS(LA_AT(work, n, k, k));
        for (size_t i = k + 1; i < n; i++) {
            const adcs_real v = LA_FABS(LA_AT(work, n, i, k));
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (!(best > (adcs_real)0)) {
            return (adcs_real)0;
        }
        if (piv != k) {
            for (size_t j = 0; j < n; j++) {
                const adcs_real t = LA_AT(work, n, k, j);
                LA_AT(work, n, k, j) = LA_AT(work, n, piv, j);
                LA_AT(work, n, piv, j) = t;
            }
            det = -det;
        }
        det *= LA_AT(work, n, k, k);
        for (size_t i = k + 1; i < n; i++) {
            const adcs_real f = LA_AT(work, n, i, k) / LA_AT(work, n, k, k);
            for (size_t j = k; j < n; j++) {
                LA_AT(work, n, i, j) -= f * LA_AT(work, n, k, j);
            }
        }
    }
    return det;
}

void v3_cross(const adcs_real a[3], const adcs_real b[3], adcs_real out[3])
{
    const adcs_real x = a[1] * b[2] - a[2] * b[1];
    const adcs_real y = a[2] * b[0] - a[0] * b[2];
    const adcs_real z = a[0] * b[1] - a[1] * b[0];
    out[0] = x;
    out[1] = y;
    out[2] = z;
}

adcs_real v3_dot(const adcs_real a[3], const adcs_real b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

adcs_real v3_norm(const adcs_real a[3])
{
    return LA_SQRT(v3_dot(a, a));
}

void v3_skew(const adcs_real v[3], adcs_real m[9])
{
    m[0] = (adcs_real)0;
    m[1] = -v[2];
    m[2] = v[1];
    m[3] = v[2];
    m[4] = (adcs_real)0;
    m[5] = -v[0];
    m[6] = -v[1];
    m[7] = v[0];
    m[8] = (adcs_real)0;
}
