/* Unit tests of the building blocks: linear algebra, quaternions, FLAE, control laws, and
 * the API's handling of misuse. The full algorithms are tested against the golden vectors
 * (golden_runner.c). */
#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "adcs.h"
#include "check.h"
#include "control.h"
#include "flae.h"
#include "linalg.h"
#include "quat.h"
#include "tsukf.h"

/* Tolerances for both precisions (the tests also run in the float build). */
#ifdef ADCS_USE_FLOAT
#define TOL 2e-4
#define TOL9 2e-4
#define NEWTON_TOL ((adcs_real)1e-5)
#else
#define TOL 1e-12
#define TOL9 1e-9
#define NEWTON_TOL ((adcs_real)1e-10)
#endif

static void test_abi_version(void)
{
    CHECK(adcs_abi_version() == ADCS_ABI_VERSION);
    CHECK(adcs_state_size() > 0);
}

static void test_mul_and_transpose(void)
{
    const adcs_real a[6] = {1, 2, 3, 4, 5, 6}; /* 2×3 */
    const adcs_real b[6] = {7, 8, 9, 10, 11, 12}; /* 3×2 */
    adcs_real c[4], bt[6], c2[4];
    la_mul(2, 3, 2, a, b, c);
    CHECK_NEAR(c[0], 58, 0);
    CHECK_NEAR(c[1], 64, 0);
    CHECK_NEAR(c[2], 139, 0);
    CHECK_NEAR(c[3], 154, 0);
    la_transpose(3, 2, b, bt);
    la_mul_bt(2, 3, 2, a, bt, c2);
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(c2[i], c[i], 0);
    }
}

/* A fixed symmetric positive definite 5×5 matrix */
static void spd5(adcs_real p[25])
{
    adcs_real m[25];
    for (int i = 0; i < 25; i++) {
        m[i] = (adcs_real)((i * 37 % 11) - 5) / 7;
    }
    la_mul_bt(5, 5, 5, m, m, p);
    for (size_t i = 0; i < 5; i++) {
        LA_AT(p, 5, i, i) += 1;
    }
}

static void test_cholesky(void)
{
    adcs_real p[25], l[25], llt[25];
    spd5(p);
    CHECK(la_cholesky(5, p, l) == 0);
    la_mul_bt(5, 5, 5, l, l, llt);
    for (int i = 0; i < 25; i++) {
        CHECK_NEAR(llt[i], p[i], TOL);
    }
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = i + 1; j < 5; j++) {
            CHECK(LA_AT(l, 5, i, j) == 0);
        }
    }
    LA_AT(p, 5, 2, 2) = -1;
    CHECK(la_cholesky(5, p, l) != 0);
}

static void test_solve_inverse_det(void)
{
    adcs_real p[25], inv[25], work[50], prod[25], x[25];
    spd5(p);
    CHECK(la_inverse(5, p, inv, work) == 0);
    la_mul(5, 5, 5, p, inv, prod);
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 5; j++) {
            CHECK_NEAR(LA_AT(prod, 5, i, j), i == j ? 1 : 0, TOL);
        }
    }
    /* right division: x·p = b */
    const adcs_real b[10] = {1, 2, 3, 4, 5, -1, 0, 2, 0, 1};
    CHECK(la_right_divide(2, 5, b, p, x, work) == 0);
    la_mul(2, 5, 5, x, p, prod);
    for (int i = 0; i < 10; i++) {
        CHECK_NEAR(prod[i], b[i], TOL);
    }
    const adcs_real m3[9] = {2, -1, 0, 1, 3, 4, 0, 5, -2};
    CHECK_NEAR(la_det(3, m3, work), -54, TOL);
    const adcs_real singular[9] = {1, 2, 3, 2, 4, 6, 1, 0, 1};
    CHECK(la_det(3, singular, work) == 0);
    CHECK(la_inverse(3, singular, inv, work) != 0);
}

static void test_quaternions(void)
{
    /* 90° about z: A maps inertial x to body −y (frame rotation) */
    const adcs_real s = (adcs_real)sqrt(0.5);
    const adcs_real qz[4] = {0, 0, s, s};
    const adcs_real x[3] = {1, 0, 0};
    adcs_real v[3];
    quat_rotate(qz, x, v);
    CHECK_NEAR(v[0], 0, TOL);
    CHECK_NEAR(v[1], -1, TOL);
    CHECK_NEAR(v[2], 0, TOL);
    /* A(q ⊗ p) = A(q) A(p) */
    const adcs_real p[4] = {0.1, -0.3, 0.2, 0.927361849549570};
    adcs_real qp[4], aq[9], ap[9], aqp[9], prod[9];
    quat_mul(qz, p, qp);
    quat_to_dcm(qz, aq);
    quat_to_dcm(p, ap);
    quat_to_dcm(qp, aqp);
    la_mul(3, 3, 3, aq, ap, prod);
    for (int i = 0; i < 9; i++) {
        CHECK_NEAR(aqp[i], prod[i], TOL);
    }
    /* rate quaternion: 90°/s about z for 1 s */
    const adcs_real w[3] = {0, 0, (adcs_real)(3.14159265358979323846 / 2)};
    adcs_real r[4];
    quat_from_rate(w, 1, r);
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(r[i], qz[i], TOL);
    }
    const adcs_real zero[3] = {0, 0, 0};
    quat_from_rate(zero, 0.1, r);
    CHECK(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 1);
}

static void test_flae_recovers_attitude(void)
{
    adcs_real q[4] = {0.3, -0.5, 0.1, 0.8};
    quat_normalize(q);
    adcs_real ref[6] = {0.6, 0.8, 0, 0.2, -0.3, 0.9};
    adcs_real n = v3_norm(&ref[3]);
    for (int i = 3; i < 6; i++) {
        ref[i] /= n;
    }
    adcs_real body[6];
    quat_rotate(q, &ref[0], &body[0]);
    quat_rotate(q, &ref[3], &body[3]);
    const adcs_real weights[2] = {1, 1};
    flae_result res;
    CHECK(flae2(body, ref, weights, NULL, NEWTON_TOL, 20, &res) == 0);
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(res.q[i], q[i], TOL9);
    }
    CHECK_NEAR(res.lambda_max, 1, TOL9);
    /* sign follows q_prev */
    const adcs_real prev[4] = {-q[0], -q[1], -q[2], -q[3]};
    CHECK(flae2(body, ref, weights, prev, NEWTON_TOL, 20, &res) == 0);
    CHECK_NEAR(res.q[3], -q[3], TOL9);
}

static void test_flae_scalar_part_near_zero(void)
{
    /* 180° rotation: q4 = 0, where fixing q4 = −1 (thesis) would divide by zero */
    const adcs_real q[4] = {0, 0.6, 0.8, 0};
    const adcs_real ref[6] = {1, 0, 0, 0, 0, 1};
    adcs_real body[6];
    quat_rotate(q, &ref[0], &body[0]);
    quat_rotate(q, &ref[3], &body[3]);
    const adcs_real weights[2] = {1, 1};
    flae_result res;
    CHECK(flae2(body, ref, weights, NULL, NEWTON_TOL, 20, &res) == 0);
    const adcs_real d = fabs(res.q[0] * q[0] + res.q[1] * q[1] + res.q[2] * q[2] + res.q[3] * q[3]);
    CHECK_NEAR(d, 1, TOL9);
}

static void test_flae_rejects_degenerate_input(void)
{
    const adcs_real ref[6] = {1, 0, 0, 1, 0, 0};
    const adcs_real weights[2] = {1, 1};
    const adcs_real zero_weights[2] = {0, 0};
    flae_result res;
    CHECK(flae2(ref, ref, weights, NULL, NEWTON_TOL, 20, &res) != 0);
    const adcs_real ok[6] = {1, 0, 0, 0, 1, 0};
    CHECK(flae2(ok, ok, zero_weights, NULL, NEWTON_TOL, 20, &res) != 0);
    CHECK(flae2(ok, ok, weights, NULL, NEWTON_TOL, 0, &res) != 0); /* no iterations allowed */
}

static void test_control_laws(void)
{
    const adcs_real q[4] = {0, 0, 0, 1};
    const adcs_real target[4] = {0, 0, 0, 1};
    const adcs_real w[3] = {0.1, 0, 0};
    adcs_real tau[3];
    control_quaternion_feedback(q, target, w, 1, 2, tau);
    CHECK_NEAR(tau[0], -0.2, TOL);
    /* the dipole produces the torque component perpendicular to B */
    const adcs_real b[3] = {0, 0, 2e-5};
    const adcs_real t_perp[3] = {1e-6, -2e-6, 0};
    adcs_real m[3], produced[3];
    control_torque_to_dipole(t_perp, b, m);
    v3_cross(m, b, produced);
    for (int i = 0; i < 3; i++) {
        CHECK_NEAR(produced[i], t_perp[i], TOL * 1e-6);
    }
    const adcs_real max_dipole[3] = {0.1, 0.1, 0.05};
    adcs_real big[3] = {0.2, 0.05, 0.05};
    CHECK(control_saturate(big, max_dipole));
    CHECK_NEAR(big[0], 0.1, TOL);
    CHECK_NEAR(big[1], 0.025, TOL);
    adcs_real small[3] = {0.01, 0, 0};
    CHECK(!control_saturate(small, max_dipole));
    adcs_real nan_cmd[3] = {(adcs_real)NAN, 0.01, 0}; /* non-finite: coils off */
    CHECK(!control_saturate(nan_cmd, max_dipole));
    CHECK(nan_cmd[0] == 0 && nan_cmd[1] == 0 && nan_cmd[2] == 0);
}

static adcs_config valid_config(void)
{
    adcs_config c;
    memset(&c, 0, sizeof c);
    c.variant = ADCS_VARIANT_MEKF;
    c.dt = 0.1;
    c.process_dt = 0.1;
    c.inertia[0] = c.inertia[4] = 0.04;
    c.inertia[8] = 0.007;
    c.max_dipole[0] = c.max_dipole[1] = c.max_dipole[2] = 0.1;
    c.flae_weight_sun = c.flae_weight_mag = 1;
    c.flae_tolerance = 1e-3;
    c.flae_max_iterations = 20;
    c.ukf_alpha = 1;
    for (int i = 0; i < ADCS_NX; i++) {
        c.p_x0[i] = c.v[i] = 1e-4;
    }
    for (int i = 0; i < ADCS_NB; i++) {
        c.p_b0[i] = 1e-6;
    }
    c.sigma_gyro_noise = 1e-4;
    c.sigma_mag_noise = 0.1;
    c.sigma_sun_noise = 1e-3;
    c.p0_attitude = c.p0_gyro_bias = c.p0_mag_bias = 0.1;
    c.fault_nis_threshold = 25;
    c.fault_steps = 3;
    c.fault_gyro_bias_sigma = c.fault_mag_bias_sigma = c.fault_attitude_sigma = 0.1;
    c.sigma_outage_accel = 1e-3;
    c.q_target[3] = 1;
    return c;
}

/* The configuration rules shared with the Python reference (config_problems). */
static void test_config_rules(void)
{
    void *mem = aligned_alloc(alignof(max_align_t), (adcs_state_size() + 15) / 16 * 16);
    adcs_state *s = (adcs_state *)mem;
    adcs_config c = valid_config();
    CHECK(adcs_init(s, &c) == 0);
    c.variant = ADCS_VARIANT_THESIS;
    CHECK(adcs_init(s, &c) == 0);
#define REJECTS(field_change)                                                                \
    do {                                                                                     \
        c = valid_config();                                                                  \
        field_change;                                                                        \
        CHECK(adcs_init(s, &c) == -3);                                                       \
    } while (0)
    REJECTS(c.dt = 0);
    REJECTS(c.dt = 20);
    REJECTS(c.inertia[8] = -0.007);
    REJECTS(c.inertia[1] = 0.01); /* not symmetric */
    REJECTS(c.max_dipole[1] = 0);
    REJECTS(c.flae_max_iterations = 0);
    REJECTS(c.flae_max_iterations = 5000);
    REJECTS(c.kp = (adcs_real)1e306);
    REJECTS(c.detumble_exit_rate = 1; c.detumble_enter_rate = (adcs_real)0.5);
    REJECTS(c.q_target[3] = 2);
    REJECTS(c.fault_steps = 0);
    REJECTS(c.fault_gyro_bias_sigma = 0);
    REJECTS(c.sigma_gyro_noise = 0);
    REJECTS(c.sigma_outage_accel = -1);
    REJECTS(c.variant = ADCS_VARIANT_THESIS; c.process_dt = (adcs_real)1e-6);
    REJECTS(c.variant = ADCS_VARIANT_THESIS; c.ukf_alpha = (adcs_real)1e-6);
    REJECTS(c.variant = ADCS_VARIANT_THESIS; c.v[2] = 0);
#undef REJECTS
    free(mem);
}

/* Newton from Σ a|b||r| finds the largest eigenvalue for non-unit vectors too (from λ = 1 it
 * converged to a smaller root above length ~1.5 and returned a 180°-wrong attitude). */
static void test_flae_long_vectors(void)
{
    adcs_real q[4] = {0.3, -0.5, 0.1, 0.8};
    quat_normalize(q);
    const adcs_real ref[6] = {0.6, 0.8, 0, 0, 0, 1};
    adcs_real body[6];
    quat_rotate(q, &ref[0], &body[0]);
    quat_rotate(q, &ref[3], &body[3]);
    const adcs_real scales[4] = {1.5, 1.8, 2.0, 0.5};
    for (int k = 0; k < 4; k++) {
        adcs_real longer[6];
        for (int i = 0; i < 3; i++) {
            longer[i] = body[i] * scales[k];
            longer[3 + i] = body[3 + i];
        }
        flae_result res;
        CHECK(flae2(longer, ref, (const adcs_real[2]){1, 1}, NULL, NEWTON_TOL, 50, &res) == 0);
        const adcs_real d =
            res.q[0] * q[0] + res.q[1] * q[1] + res.q[2] * q[2] + res.q[3] * q[3];
        CHECK_NEAR(fabs(d), 1, TOL9);
    }
}

static void test_abi_precision(void)
{
    CHECK(adcs_real_size() == sizeof(adcs_real));
    CHECK(adcs_check_abi());
}

static void test_api_misuse(void)
{
    void *mem = aligned_alloc(alignof(max_align_t), (adcs_state_size() + 15) / 16 * 16);
    adcs_state *s = (adcs_state *)mem;
    adcs_config c = valid_config();
    adcs_input in;
    adcs_output out;
    memset(&in, 0, sizeof in);
    memset(mem, 0, adcs_state_size());

    CHECK(adcs_init(NULL, &c) != 0);
    CHECK(adcs_init(s, NULL) != 0);
    CHECK(adcs_step(s, &in, &out) != 0); /* never initialised */
    CHECK(out.status == ADCS_STATUS_NOT_INITIALISED && out.mode == ADCS_MODE_OFF);
    CHECK(isnan(out.q_est[0]));
    CHECK(adcs_step(s, &in, NULL) != 0);

    c.variant = 7;
    CHECK(adcs_init(s, &c) != 0);
    c = valid_config();
    c.dt = (adcs_real)NAN;
    CHECK(adcs_init(s, &c) != 0);
    c = valid_config();
    c.inertia[8] = 0;
    CHECK(adcs_init(s, &c) != 0);
    c = valid_config();
    c.variant = ADCS_VARIANT_THESIS;
    c.v[3] = 0;
    CHECK(adcs_init(s, &c) != 0);
    CHECK(adcs_step(s, &in, &out) != 0); /* a failed init leaves the state unusable */

    for (uint32_t variant = 0; variant < 2; variant++) {
        c = valid_config();
        c.variant = variant;
        CHECK(adcs_init(s, &c) == 0);
        /* no valid data: no estimate, no crash */
        CHECK(adcs_step(s, &in, &out) == 0);
        CHECK(out.mode == ADCS_MODE_OFF);
        CHECK(out.status & ADCS_STATUS_NO_ATTITUDE_FIX);
        /* NaN everywhere, all marked valid */
        for (int i = 0; i < 3; i++) {
            in.mag_b[i] = in.gyro_b[i] = in.sun_b[i] = in.sun_ref_i[i] = in.mag_ref_i[i] = (adcs_real)NAN;
        }
        in.mag_valid = in.sun_valid = 1;
        in.gyro_valid = 2; /* any non-zero byte means valid */
        CHECK(adcs_step(s, &in, &out) == 0);
        CHECK(out.status & ADCS_STATUS_BAD_INPUT);
        memset(&in, 0, sizeof in);
    }
    free(mem);
}

/* ---- TSUKF recovery after a failed Cholesky (same tests as tsukf.rs) ------------------ */

#ifdef ADCS_USE_FLOAT
#define TS_TOL 2e-4
#else
#define TS_TOL 1e-12
#endif

static void ts_identity(adcs_real x[TS_NX], const void *ctx)
{
    (void)x;
    (void)ctx;
}

static void ts_fill(adcs_real *v, size_t n, adcs_real value)
{
    for (size_t i = 0; i < n; i++) {
        v[i] = value;
    }
}

static void ts_y(adcs_real y[TS_NY])
{
    static const adcs_real mag[3] = {(adcs_real)0.3, (adcs_real)-0.2, (adcs_real)0.1};
    static const adcs_real gyro[3] = {(adcs_real)0.02, (adcs_real)-0.01, (adcs_real)0.015};
    ts_fill(y, TS_NY, 0);
    y[3] = 1;
    y[TS_SUN + 2] = 1;
    for (size_t i = 0; i < 3; i++) {
        y[TS_MAG + i] = mag[i];
        y[TS_W + i] = gyro[i];
    }
}

static adcs_real ts_wx[TS_NX], ts_wb[TS_NB], ts_v[TS_NY], ts_px0[TS_NX], ts_pb0[TS_NB];
static bool ts_all[TS_NY];
static tsukf ts_a, ts_b; /* ~25 KB each: static, not on the stack */

/* A filter after a few steps, so that β and Pᵇ are no longer their initial values. */
static void ts_warmed(tsukf *f)
{
    adcs_real x0[TS_NX], b0[TS_NB], y[TS_NY];
    ts_fill(ts_wx, TS_NX, (adcs_real)1e-6);
    ts_fill(ts_wb, TS_NB, (adcs_real)1e-6);
    ts_fill(ts_v, TS_NY, (adcs_real)1e-3);
    ts_fill(ts_px0, TS_NX, (adcs_real)1e-2);
    ts_fill(ts_pb0, TS_NB, (adcs_real)1e-2);
    for (size_t i = 0; i < TS_NY; i++) {
        ts_all[i] = true;
    }
    ts_fill(x0, TS_NX, 0);
    x0[3] = 1;
    x0[TS_SUN + 2] = 1;
    ts_fill(b0, TS_NB, 0);
    ts_y(y);
    tsukf_configure(f, 1, ts_wx, ts_wb, ts_v);
    tsukf_reset(f, x0, ts_px0, b0, ts_pb0);
    for (int k = 0; k < 5; k++) {
        CHECK(tsukf_step(f, ts_identity, NULL, y, ts_all));
    }
    bool coupled = false;
    for (size_t i = 0; i < TS_NX * TS_NB; i++) {
        coupled = coupled || f->beta[i] != 0;
    }
    CHECK(coupled);
}

static void ts_check_same(const tsukf *a, const tsukf *b)
{
    for (size_t i = 0; i < TS_NX; i++) {
        CHECK_NEAR(a->x[i], b->x[i], TS_TOL);
    }
    for (size_t i = 0; i < TS_NX * TS_NX; i++) {
        CHECK_NEAR(a->px[i], b->px[i], TS_TOL);
    }
    for (size_t i = 0; i < TS_NB; i++) {
        CHECK_NEAR(a->b[i], b->b[i], TS_TOL);
    }
    for (size_t i = 0; i < TS_NB * TS_NB; i++) {
        CHECK_NEAR(a->pb[i], b->pb[i], TS_TOL);
    }
}

/* A failed prediction factorisation restarts Pˣ, Pᵇ and β together: the step equals one
 * from a filter freshly reset to P₀ with the same estimate. */
static void test_tsukf_failed_prediction_cholesky(void)
{
    adcs_real y[TS_NY];
    ts_y(y);
    ts_warmed(&ts_a);
    ts_b = ts_a;
    tsukf_reset(&ts_b, ts_a.x, ts_px0, ts_a.b, ts_pb0);
    ts_a.px[0] = -1;
    CHECK(!tsukf_step(&ts_a, ts_identity, NULL, y, ts_all));
    CHECK(tsukf_step(&ts_b, ts_identity, NULL, y, ts_all));
    ts_check_same(&ts_a, &ts_b);
}

/* A failed measurement factorisation resets the predicted Pˣ and drops the predicted β,
 * keeping Pᵇ: the step equals one from a filter with Pˣ = P₀ˣ, β = 0, the same Pᵇ and no
 * process noise on x (an identity process then predicts exactly that). */
static void test_tsukf_failed_measurement_cholesky(void)
{
    adcs_real y[TS_NY], zero[TS_NX];
    ts_y(y);
    ts_fill(zero, TS_NX, 0);
    ts_warmed(&ts_a);
    ts_b = ts_a;
    la_zero(TS_NX * TS_NX, ts_b.px);
    for (size_t i = 0; i < TS_NX; i++) {
        LA_AT(ts_b.px, TS_NX, i, i) = ts_px0[i];
    }
    la_zero(TS_NX * TS_NB, ts_b.beta);
    tsukf_configure(&ts_b, 1, zero, ts_wb, ts_v);
    ts_wx[0] = -1; /* makes the predicted Pˣ indefinite */
    tsukf_configure(&ts_a, 1, ts_wx, ts_wb, ts_v);
    CHECK(!tsukf_step(&ts_a, ts_identity, NULL, y, ts_all));
    CHECK(tsukf_step(&ts_b, ts_identity, NULL, y, ts_all));
    ts_check_same(&ts_a, &ts_b);
}

int main(void)
{
    RUN(test_abi_version);
    RUN(test_mul_and_transpose);
    RUN(test_cholesky);
    RUN(test_solve_inverse_det);
    RUN(test_quaternions);
    RUN(test_flae_recovers_attitude);
    RUN(test_flae_scalar_part_near_zero);
    RUN(test_flae_rejects_degenerate_input);
    RUN(test_control_laws);
    RUN(test_api_misuse);
    RUN(test_config_rules);
    RUN(test_flae_long_vectors);
    RUN(test_abi_precision);
    RUN(test_tsukf_failed_prediction_cholesky);
    RUN(test_tsukf_failed_measurement_cholesky);
    return check_summary();
}
