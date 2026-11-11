#include "tsukf.h"

#include <math.h>
#include <stddef.h>

#include "linalg.h"
#include "quat.h"

#define NX TS_NX
#define NB TS_NB
#define NY TS_NY
#define NS TS_NS
#define UNAVAILABLE_VARIANCE ((adcs_real)1e12)

#ifdef ADCS_USE_FLOAT
#define TINY ((adcs_real)1e-30f)
#else
#define TINY ((adcs_real)1e-300)
#endif

/* Added to the diagonal before every factorisation. The double-precision epsilon in both
 * precisions: float's (1.2e-7) is larger than the process noise on q and ω and would act as
 * extra process noise, wrecking the single-precision filter. */
#define CHOLESKY_EPS ((adcs_real)2.220446049250313e-16)

static const adcs_real JITTER[4] = {(adcs_real)0, (adcs_real)1e-12, (adcs_real)1e-10,
                                    (adcs_real)1e-8};

void tsukf_configure(tsukf *f, adcs_real alpha, const adcs_real w_x[NX], const adcs_real w_b[NB],
                     const adcs_real v[NY])
{
    /* eq. 35 with κ = 3 − n, so n + λ = 3α² (thesis appendix), β = 2 */
    const adcs_real n = (adcs_real)NX;
    const adcs_real lam = (adcs_real)3 * alpha * alpha - n;
    f->wm0 = lam / (n + lam);
    f->wc0 = f->wm0 + (adcs_real)1 - alpha * alpha + (adcs_real)2;
    f->wi = (adcs_real)1 / ((adcs_real)2 * (n + lam));
    f->xi = LA_SQRT(n + lam);
    la_copy(NX, f->w_x, w_x);
    la_copy(NB, f->w_b, w_b);
    la_copy(NY, f->v, v);
}

static void diag_matrix(size_t n, const adcs_real *d, adcs_real *m)
{
    la_zero(n * n, m);
    for (size_t i = 0; i < n; i++) {
        LA_AT(m, n, i, i) = d[i];
    }
}

void tsukf_reset(tsukf *f, const adcs_real x0[NX], const adcs_real px0[NX],
                 const adcs_real b0[NB], const adcs_real pb0[NB])
{
    la_copy(NX, f->x, x0);
    la_copy(NB, f->b, b0);
    la_copy(NX, f->px0, px0);
    la_copy(NB, f->pb0, pb0);
    diag_matrix(NX, px0, f->px);
    diag_matrix(NB, pb0, f->pb);
    la_zero(NX * NB, f->beta);
}

/* Lower Cholesky factor of p + εI, retrying with more jitter if needed. */
static int robust_cholesky(const adcs_real *p, adcs_real *lower, adcs_real *tmp)
{
    adcs_real scale = TINY;
    for (size_t i = 0; i < NX; i++) {
        const adcs_real d = LA_FABS(LA_AT(p, NX, i, i));
        if (d > scale) {
            scale = d;
        }
    }
    for (size_t k = 0; k < 4; k++) {
        la_copy(NX * NX, tmp, p);
        for (size_t i = 0; i < NX; i++) {
            LA_AT(tmp, NX, i, i) += CHOLESKY_EPS + JITTER[k] * scale;
        }
        if (la_cholesky(NX, tmp, lower) == 0) {
            return 0;
        }
    }
    return -1;
}

static void sigma_points(const tsukf *f, const adcs_real x[NX], const adcs_real *lower,
                         adcs_real *chi)
{
    for (size_t j = 0; j < NX; j++) {
        LA_AT(chi, NX, 0, j) = x[j];
    }
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NX; j++) {
            const adcs_real s = f->xi * LA_AT(lower, NX, j, i);
            LA_AT(chi, NX, 1 + i, j) = x[j] + s;
            LA_AT(chi, NX, 1 + NX + i, j) = x[j] - s;
        }
    }
}

/* eqs. 50/57: Σ_i [g(χ_i) − g(χ_{i+n})] e_iᵀ / (2ξ) · L⁻¹ */
static int transform(tsukf *f, const adcs_real *out, const adcs_real *lower, adcs_real *m)
{
    tsukf_work *w = &f->work;
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NX; j++) {
            LA_AT(w->diff, NX, j, i) = LA_AT(out, NX, 1 + i, j) - LA_AT(out, NX, 1 + NX + i, j);
        }
    }
    if (la_right_divide(NX, NX, w->diff, lower, m, w->solve) != 0) {
        return -1;
    }
    const adcs_real d = (adcs_real)2 * f->xi;
    for (size_t i = 0; i < NX * NX; i++) {
        m[i] /= d;
    }
    return 0;
}

static void ut_mean(const tsukf *f, const adcs_real *out, adcs_real mean[NX])
{
    for (size_t j = 0; j < NX; j++) {
        adcs_real s = LA_AT(out, NX, 1, j);
        for (size_t i = 2; i < NS; i++) {
            s += LA_AT(out, NX, i, j);
        }
        mean[j] = f->wm0 * LA_AT(out, NX, 0, j) + f->wi * s;
    }
}

/* wc0 d0 d0ᵀ + wi Σ dᵢ dᵢᵀ with dᵢ = outᵢ − mean */
static void ut_cov(const tsukf *f, const adcs_real *out, const adcs_real mean[NX], adcs_real *p)
{
    for (size_t a = 0; a < NX; a++) {
        for (size_t b = a; b < NX; b++) {
            adcs_real s = (adcs_real)0;
            for (size_t i = 1; i < NS; i++) {
                s += (LA_AT(out, NX, i, a) - mean[a]) * (LA_AT(out, NX, i, b) - mean[b]);
            }
            const adcs_real d0 = (LA_AT(out, NX, 0, a) - mean[a]) * (LA_AT(out, NX, 0, b) - mean[b]);
            const adcs_real v = f->wc0 * d0 + f->wi * s;
            LA_AT(p, NX, a, b) = v;
            LA_AT(p, NX, b, a) = v;
        }
    }
}

/* y += F_s b: magnetometer bias onto B, gyro bias onto ω */
static void add_fs_b(adcs_real y[NY], const adcs_real b[NB])
{
    for (size_t i = 0; i < 3; i++) {
        y[TS_MAG + i] += b[i];
        y[TS_W + i] += b[3 + i];
    }
}

/* A zero or overflowed norm gives NaN, which the finiteness check then catches. */
static void normalize_q(adcs_real x[NX])
{
    adcs_real n = quat_norm(&x[TS_Q]);
    if (!(n > (adcs_real)0) || !isfinite(n)) {
        n = (adcs_real)NAN;
    }
    for (size_t i = 0; i < 4; i++) {
        x[TS_Q + i] /= n;
    }
}

static void normalize_q_sun(adcs_real x[NX])
{
    normalize_q(x);
    const adcs_real n = v3_norm(&x[TS_SUN]);
    if (n > (adcs_real)0) {
        for (size_t i = 0; i < 3; i++) {
            x[TS_SUN + i] /= n;
        }
    }
}

/* Restart from the initial covariances (the estimate is kept). */
static bool reset_covariance(tsukf *f)
{
    diag_matrix(NX, f->px0, f->px);
    diag_matrix(NB, f->pb0, f->pb);
    la_zero(NX * NB, f->beta);
    return false;
}

/* c = a·b·aᵀ for a (r×k), b (k×k) symmetric; t is r×k scratch. */
static void sandwich(size_t r, size_t k, const adcs_real *a, const adcs_real *b, adcs_real *t,
                     adcs_real *c)
{
    la_mul(r, k, k, a, b, t);
    la_mul_bt(r, k, r, t, a, c);
}

bool tsukf_step(tsukf *f, tsukf_process_fn process, const void *ctx, const adcs_real y[NY],
                const bool available[NY])
{
    tsukf_work *w = &f->work;
    bool ok = true;

    /* ---- prediction ------------------------------------------------------------------- */
    if (robust_cholesky(f->px, w->lower, w->a) != 0) {
        /* Restart Pˣ, Pᵇ and β together: a reset Pˣ with the old β and Pᵇ can leave the
         * bias-free covariance Pˣ − βPᵇβᵀ indefinite. */
        ok = reset_covariance(f);
        (void)la_cholesky(NX, f->px, w->lower);
    }
    sigma_points(f, f->x, w->lower, w->chi);
    for (size_t i = 0; i < NS; i++) {
        process(&w->chi[i * NX], ctx);
    }
    if (transform(f, w->chi, w->lower, w->m) != 0) { /* eq. 50 */
        return reset_covariance(f);
    }
    la_mul(NX, NX, NB, w->m, f->beta, w->r); /* eq. 51 (F_a = 0) */
    adcs_real mk[NX], x_pred[NX];
    ut_mean(f, w->chi, mk); /* eq. 52 */
    la_copy(NX, x_pred, mk); /* eq. 53 */
    normalize_q(x_pred);
    /* eq. 54: P = cov − M β Pb βᵀ Mᵀ + R Pb Rᵀ + W_x */
    ut_cov(f, w->chi, mk, w->px_pred);
    la_mul(NX, NB, NB, w->r, f->pb, w->t1); /* M β Pb = R Pb */
    la_mul_bt(NX, NB, NX, w->t1, f->beta, w->a);
    la_mul_bt(NX, NX, NX, w->a, w->m, w->b);
    la_mul_bt(NX, NB, NX, w->t1, w->r, w->c);
    for (size_t i = 0; i < NX * NX; i++) {
        w->px_pred[i] = w->px_pred[i] - w->b[i] + w->c[i];
    }
    for (size_t i = 0; i < NX; i++) {
        LA_AT(w->px_pred, NX, i, i) += f->w_x[i];
    }
    la_copy(NB * NB, w->pb_pred, f->pb); /* eq. 56 */
    for (size_t i = 0; i < NB; i++) {
        LA_AT(w->pb_pred, NB, i, i) += f->w_b[i];
    }
    if (la_inverse(NB, w->pb_pred, w->pb_inv, w->six_a) != 0) {
        return reset_covariance(f);
    }
    la_mul(NX, NB, NB, w->t1, w->pb_inv, w->beta_pred); /* eq. 55 */
    const adcs_real *b_pred = f->b;

    /* ---- measurement sigma points ----------------------------------------------------- */
    la_symmetrize(NX, w->px_pred);
    if (robust_cholesky(w->px_pred, w->lower, w->a) != 0) {
        /* Reset the predicted Pˣ and drop the predicted coupling, so that the bias-free
         * covariance Pˣ − βPᵇβᵀ is P₀ˣ (positive definite); Pᵇ is kept. */
        diag_matrix(NX, f->px0, w->px_pred);
        la_zero(NX * NB, w->beta_pred);
        ok = false;
        (void)la_cholesky(NX, w->px_pred, w->lower);
    }
    sigma_points(f, x_pred, w->lower, w->chi); /* h(x) = x */
    if (transform(f, w->chi, w->lower, w->n) != 0) { /* eq. 57 */
        return reset_covariance(f);
    }
    la_mul(NY, NX, NB, w->n, w->beta_pred, w->h); /* eq. 58 */
    for (size_t i = 0; i < 3; i++) {
        LA_AT(w->h, NB, TS_MAG + i, i) += (adcs_real)1;
        LA_AT(w->h, NB, TS_W + i, 3 + i) += (adcs_real)1;
    }
    adcs_real nk[NY];
    ut_mean(f, w->chi, nk); /* eq. 59 */

    adcs_real y_b[NY], y_eff[NY];
    la_copy(NY, y_b, nk);
    add_fs_b(y_b, b_pred);
    for (size_t i = 0; i < NY; i++) {
        y_eff[i] = available[i] ? y[i] : y_b[i];
    }

    /* ---- bias-free filter (eq. 60) ---------------------------------------------------- */
    adcs_real x_t[NX], y_t[NY], tmpv[NX];
    la_mul_vec(NX, NB, w->beta_pred, b_pred, tmpv);
    for (size_t i = 0; i < NX; i++) {
        x_t[i] = x_pred[i] - tmpv[i];
    }
    sandwich(NX, NB, w->beta_pred, w->pb_pred, w->t1, w->a);
    for (size_t i = 0; i < NX * NX; i++) {
        w->px_t[i] = w->px_pred[i] - w->a[i];
    }
    la_mul(NY, NX, NB, w->n, w->beta_pred, w->t2); /* N β */
    la_mul_vec(NY, NB, w->t2, b_pred, tmpv);
    for (size_t i = 0; i < NY; i++) {
        y_t[i] = nk[i] - tmpv[i];
    }
    ut_cov(f, w->chi, nk, w->py_t);
    la_mul(NY, NB, NB, w->t2, w->pb_pred, w->t1);
    la_mul_bt(NY, NB, NX, w->t1, w->beta_pred, w->a);
    la_mul_bt(NY, NX, NY, w->a, w->n, w->b);
    for (size_t i = 0; i < NY * NY; i++) {
        w->py_t[i] -= w->b[i];
    }
    for (size_t i = 0; i < NY; i++) {
        LA_AT(w->py_t, NY, i, i) += available[i] ? f->v[i] : UNAVAILABLE_VARIANCE;
    }
    la_mul_bt(NX, NX, NY, w->px_t, w->n, w->a); /* Px Nᵀ */
    if (la_right_divide(NX, NY, w->a, w->py_t, w->k_t, w->solve) != 0) {
        return reset_covariance(f);
    }
    la_mul(NX, NY, NB, w->k_t, w->h, w->t1); /* eq. 61 */
    for (size_t i = 0; i < NX * NB; i++) {
        w->beta_upd[i] = w->beta_pred[i] - w->t1[i];
    }
    adcs_real e[NY], x_upd[NX];
    for (size_t i = 0; i < NY; i++) {
        e[i] = y_eff[i] - y_t[i];
    }
    la_mul_vec(NX, NY, w->k_t, e, tmpv); /* eq. 62 */
    for (size_t i = 0; i < NX; i++) {
        x_upd[i] = x_t[i] + tmpv[i];
    }
    sandwich(NX, NY, w->k_t, w->py_t, w->b, w->a); /* eq. 63 */
    for (size_t i = 0; i < NX * NX; i++) {
        w->px_upd[i] = w->px_t[i] - w->a[i];
    }
    la_symmetrize(NX, w->px_upd);

    /* ---- bias filter (eqs. 64–66) ----------------------------------------------------- */
    la_mul_bt(NB, NB, NY, w->pb_pred, w->h, w->p_by);
    sandwich(NY, NB, w->h, w->pb_pred, w->t1, w->a);
    for (size_t i = 0; i < NY * NY; i++) {
        w->p_y[i] = w->py_t[i] + w->a[i];
    }
    if (la_right_divide(NB, NY, w->p_by, w->p_y, w->k_b, w->solve) != 0) {
        return reset_covariance(f);
    }
    adcs_real b_upd[NB];
    for (size_t i = 0; i < NY; i++) {
        e[i] = y_eff[i] - y_b[i];
    }
    la_mul_vec(NB, NY, w->k_b, e, b_upd);
    for (size_t i = 0; i < NB; i++) {
        b_upd[i] += b_pred[i];
    }
    la_mul(NB, NY, NY, w->k_b, w->p_y, w->nb_ny);
    la_mul_bt(NB, NY, NB, w->nb_ny, w->k_b, w->six_a);
    for (size_t i = 0; i < NB * NB; i++) {
        w->pb_upd[i] = w->pb_pred[i] - w->six_a[i];
    }
    la_symmetrize(NB, w->pb_upd);

    /* ---- coupled result (eqs. 67–68) -------------------------------------------------- */
    la_mul_vec(NX, NB, w->beta_upd, b_upd, tmpv);
    for (size_t i = 0; i < NX; i++) {
        x_upd[i] += tmpv[i];
    }
    sandwich(NX, NB, w->beta_upd, w->pb_upd, w->t1, w->a);
    for (size_t i = 0; i < NX * NX; i++) {
        w->px_upd[i] += w->a[i];
    }
    la_symmetrize(NX, w->px_upd);
    normalize_q_sun(x_upd);

    if (!la_all_finite(NX, x_upd) || !la_all_finite(NX * NX, w->px_upd)) {
        return reset_covariance(f);
    }
    la_copy(NX, f->x, x_upd);
    la_copy(NX * NX, f->px, w->px_upd);
    la_copy(NB, f->b, b_upd);
    la_copy(NB * NB, f->pb, w->pb_upd);
    la_copy(NX * NB, f->beta, w->beta_upd);
    return ok;
}
