/*
 * adcs.c: the adcs.h API. Port of src/adcssim/adcs_ref/adcs.py (ThesisAdcs) and
 * adcs_improved.py (ImprovedAdcs with the MEKF); the golden vectors in testdata/golden
 * check that the two match.
 */
#include "adcs.h"

#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

#include "control.h"
#include "flae.h"
#include "linalg.h"
#include "mekf.h"
#include "quat.h"
#include "tsukf.h"

#define ADCS_MAGIC 0x41444353u /* "ADCS" */
#define UT ((adcs_real)1e6)    /* tesla -> microtesla, the filters' unit for the field */

typedef struct {
    tsukf filter;
    adcs_real inertia[9];
    adcs_real inertia_inv[9];
    adcs_real m_prev[3];
    adcs_real last_flae_q[4];
    bool has_last_flae_q;
} thesis_state;

typedef struct {
    mekf filter;
    adcs_real gyro_prev[3];
    uint32_t suspect_steps;
    adcs_real inertia[9];
    adcs_real inertia_inv[9];
    adcs_real outage_time; /* s since the gyro became unavailable, 0 while it works */
} mekf_state;

struct adcs_state {
    uint32_t magic;
    adcs_config cfg;
    bool initialised; /* the filter has started (first attitude fix) */
    control_state control;
    union {
        thesis_state thesis;
        mekf_state mekf;
    } v;
};

uint32_t adcs_abi_version(void)
{
    return ADCS_ABI_VERSION;
}

size_t adcs_real_size(void)
{
    return sizeof(adcs_real);
}

size_t adcs_state_size(void)
{
    return sizeof(struct adcs_state);
}

/* ---- helpers ------------------------------------------------------------------------- */

/* Plausibility limits (src/adcssim/adcs_ref/limits.py): a reading outside them is treated
 * like a NaN, so absurd values cannot overflow the filter state. */
#define GYRO_MAX ((adcs_real)50)     /* rad/s */
#define FIELD_MAX ((adcs_real)1e-3)  /* T */
#define UNIT_MIN ((adcs_real)0.5)    /* norm of a direction that should be unit */
#define UNIT_MAX ((adcs_real)2)

static bool gyro_plausible(const adcs_real v[3])
{
    return la_all_finite(3, v) && v3_norm(v) <= GYRO_MAX;
}

static bool field_plausible(const adcs_real v[3])
{
    if (!la_all_finite(3, v)) {
        return false;
    }
    const adcs_real n = v3_norm(v);
    return n > (adcs_real)0 && n <= FIELD_MAX;
}

static bool unit_plausible(const adcs_real v[3])
{
    if (!la_all_finite(3, v)) {
        return false;
    }
    const adcs_real n = v3_norm(v);
    return n >= UNIT_MIN && n <= UNIT_MAX;
}

static void scale3(const adcs_real v[3], adcs_real s, adcs_real out[3])
{
    for (size_t i = 0; i < 3; i++) {
        out[i] = v[i] * s;
    }
}

static void div3(const adcs_real v[3], adcs_real d, adcs_real out[3])
{
    for (size_t i = 0; i < 3; i++) {
        out[i] = v[i] / d;
    }
}

/* Output when there is no estimate yet: NaN estimate, coils off. */
static void output_empty(adcs_output *out, uint32_t mode, uint32_t status)
{
    for (size_t i = 0; i < 4; i++) {
        out->q_est[i] = (adcs_real)NAN;
    }
    for (size_t i = 0; i < 3; i++) {
        out->w_est[i] = (adcs_real)NAN;
        out->bias_mag[i] = (adcs_real)NAN;
        out->bias_gyro[i] = (adcs_real)NAN;
        out->m_cmd[i] = (adcs_real)0;
    }
    for (size_t i = 0; i < ADCS_NX + ADCS_NB; i++) {
        out->p_diag[i] = (adcs_real)NAN;
    }
    out->mode = mode;
    out->status = status;
}

static bool all_positive(size_t n, const adcs_real *v)
{
    for (size_t i = 0; i < n; i++) {
        if (!(v[i] > (adcs_real)0) || !isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

static bool all_nonnegative(size_t n, const adcs_real *v)
{
    for (size_t i = 0; i < n; i++) {
        if (!(v[i] >= (adcs_real)0) || !isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

/* Configuration rules, the same in every implementation (config_problems in
 * src/adcssim/backends/config.py; docs/adcs.md, "Configuration rules"). */
#define MAX_DT ((adcs_real)10)
#define MAX_PROCESS_STEPS ((adcs_real)1000) /* bounds the RK4 steps, and so the run time */
#define MAX_FLAE_ITERATIONS 1000u
#define ALPHA_MIN ((adcs_real)1e-2) /* smaller: the UT weights lose precision */
#define ALPHA_MAX ((adcs_real)1e2)
#define MAX_BDOT_GAIN ((adcs_real)1e9)
#define MAX_PD_GAIN ((adcs_real)1e3)

static bool in_range(adcs_real x, adcs_real lo, adcs_real hi)
{
    return isfinite(x) && x >= lo && x <= hi;
}

static bool inertia_valid(const adcs_real m[9])
{
    if (!la_all_finite(9, m)) {
        return false;
    }
    adcs_real scale = (adcs_real)0, asym = (adcs_real)0, lower[9];
    for (size_t i = 0; i < 9; i++) {
        scale = LA_FABS(m[i]) > scale ? LA_FABS(m[i]) : scale;
    }
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            const adcs_real d = LA_FABS(LA_AT(m, 3, i, j) - LA_AT(m, 3, j, i));
            asym = d > asym ? d : asym;
        }
    }
    return asym <= (adcs_real)1e-9 * scale && la_cholesky(3, m, lower) == 0;
}

static bool config_valid(const adcs_config *c)
{
    const adcs_real wsum = c->flae_weight_sun + c->flae_weight_mag;
    const adcs_real qn = quat_norm(c->q_target);
    if (!in_range(c->dt, (adcs_real)0, MAX_DT) || !(c->dt > (adcs_real)0) ||
        !inertia_valid(c->inertia) || !all_positive(3, c->max_dipole) ||
        !all_nonnegative(1, &c->flae_weight_sun) || !all_nonnegative(1, &c->flae_weight_mag) ||
        !all_positive(1, &wsum) || !all_positive(1, &c->flae_tolerance) ||
        c->flae_max_iterations < 1u || c->flae_max_iterations > MAX_FLAE_ITERATIONS ||
        !in_range(c->bdot_gain, (adcs_real)0, MAX_BDOT_GAIN) ||
        !in_range(c->kp, (adcs_real)0, MAX_PD_GAIN) || !in_range(c->kd, (adcs_real)0, MAX_PD_GAIN) ||
        !all_nonnegative(1, &c->detumble_exit_rate) ||
        !in_range(c->detumble_enter_rate, c->detumble_exit_rate, (adcs_real)INFINITY) ||
        !all_nonnegative(1, &c->settle_time) || !la_all_finite(4, c->q_target) ||
        !(LA_FABS(qn - (adcs_real)1) <= (adcs_real)1e-6)) {
        return false;
    }
    if (c->variant == ADCS_VARIANT_THESIS) {
        return all_positive(1, &c->process_dt) && c->dt / c->process_dt <= MAX_PROCESS_STEPS &&
               in_range(c->ukf_alpha, ALPHA_MIN, ALPHA_MAX) && all_positive(ADCS_NX, c->p_x0) &&
               all_positive(ADCS_NB, c->p_b0) && all_positive(ADCS_NX, c->v) &&
               all_nonnegative(ADCS_NX, c->w_x) && all_nonnegative(ADCS_NB, c->w_b);
    }
    if (c->variant == ADCS_VARIANT_MEKF) {
        const adcs_real positive[] = {
            c->sigma_gyro_noise,     c->sigma_mag_noise,       c->sigma_sun_noise,
            c->p0_attitude,          c->p0_gyro_bias,          c->p0_mag_bias,
            c->fault_nis_threshold,  c->fault_gyro_bias_sigma, c->fault_mag_bias_sigma,
            c->fault_attitude_sigma,
        };
        const adcs_real nonnegative[] = {c->sigma_gyro_bias_walk, c->sigma_mag_bias_walk,
                                         c->sigma_outage_accel};
        return all_positive(sizeof positive / sizeof positive[0], positive) &&
               all_nonnegative(sizeof nonnegative / sizeof nonnegative[0], nonnegative) &&
               c->fault_steps >= 1u;
    }
    return false;
}

/* ---- thesis variant: FLAE + two-stage UKF -------------------------------------------- */

typedef struct {
    adcs_real dt;
    adcs_real process_dt;
    adcs_real torque[3];
    const adcs_real *inertia;
    const adcs_real *inertia_inv;
} process_ctx;

/* Eq. 126, except that the body vectors rotate as −ω × v using the state's own vectors
 * (docs/adcs.md explains why the thesis form diverges in eclipse). */
static void xdot(const adcs_real x[TS_NX], const process_ctx *c, adcs_real out[TS_NX])
{
    adcs_real q[4];
    la_copy(4, q, &x[TS_Q]);
    const adcs_real qn = quat_norm(q);
    for (size_t i = 0; i < 4; i++) {
        q[i] /= qn;
    }
    const adcs_real *w = &x[TS_W];
    adcs_real t[3];
    v3_cross(q, w, t);
    for (size_t i = 0; i < 3; i++) {
        out[TS_Q + i] = (adcs_real)0.5 * (q[3] * w[i] + t[i]);
    }
    out[TS_Q + 3] = (adcs_real)-0.5 * (q[0] * w[0] + q[1] * w[1] + q[2] * w[2]);
    v3_cross(w, &x[TS_SUN], t);
    for (size_t i = 0; i < 3; i++) {
        out[TS_SUN + i] = -t[i];
    }
    v3_cross(w, &x[TS_MAG], t);
    for (size_t i = 0; i < 3; i++) {
        out[TS_MAG + i] = -t[i];
    }
    /* ω̇ = I⁻¹ (−ω × Iω + τ) */
    adcs_real iw[3], g[3];
    const adcs_real nw[3] = {-w[0], -w[1], -w[2]};
    la_mul_vec(3, 3, c->inertia, w, iw);
    v3_cross(nw, iw, g);
    for (size_t i = 0; i < 3; i++) {
        g[i] += c->torque[i];
    }
    la_mul_vec(3, 3, c->inertia_inv, g, &out[TS_W]);
}

/* RK4 over dt in steps of at most process_dt, applied to one sigma point in place. */
static void process_model(adcs_real x[TS_NX], const void *ctx)
{
    const process_ctx *c = (const process_ctx *)ctx;
    adcs_real steps = LA_RINT(c->dt / c->process_dt); /* round half to even, like Python */
    if (steps < (adcs_real)1) {
        steps = (adcs_real)1;
    }
    const adcs_real h = c->dt / steps;
    const uint32_t n = (uint32_t)steps;
    adcs_real k1[TS_NX], k2[TS_NX], k3[TS_NX], k4[TS_NX], xt[TS_NX];
    for (uint32_t s = 0; s < n; s++) {
        xdot(x, c, k1);
        for (size_t i = 0; i < TS_NX; i++) {
            k1[i] *= h;
            xt[i] = x[i] + (adcs_real)0.5 * k1[i];
        }
        xdot(xt, c, k2);
        for (size_t i = 0; i < TS_NX; i++) {
            k2[i] *= h;
            xt[i] = x[i] + (adcs_real)0.5 * k2[i];
        }
        xdot(xt, c, k3);
        for (size_t i = 0; i < TS_NX; i++) {
            k3[i] *= h;
            xt[i] = x[i] + k3[i];
        }
        xdot(xt, c, k4);
        for (size_t i = 0; i < TS_NX; i++) {
            k4[i] *= h;
            x[i] = x[i] +
                   (k1[i] + (adcs_real)2 * k2[i] + (adcs_real)2 * k3[i] + k4[i]) / (adcs_real)6;
        }
    }
}

/* A fixed inertial direction seen from a body rotating at w for dt (Rodrigues), unit. */
static void propagate_body_vector(const adcs_real v[3], const adcs_real w[3], adcs_real dt,
                                  adcs_real out[3])
{
    const adcs_real rate = v3_norm(w);
    const adcs_real angle = rate * dt;
    if (angle < (adcs_real)1e-12) {
        la_copy(3, out, v);
    } else {
        adcs_real k[3], kv[3];
        div3(w, rate, k);
        const adcs_real c = LA_COS(angle), s = LA_SIN(angle);
        v3_cross(k, v, kv);
        const adcs_real kd = v3_dot(k, v);
        for (size_t i = 0; i < 3; i++) {
            out[i] = v[i] * c - kv[i] * s + k[i] * kd * ((adcs_real)1 - c);
        }
    }
    const adcs_real n = v3_norm(out);
    if (n > (adcs_real)0) {
        div3(out, n, out);
    }
}

static void thesis_finish(adcs_state *s, uint32_t status, adcs_output *out)
{
    thesis_state *t = &s->v.thesis;
    const adcs_real *x = t->filter.x;
    adcs_real b_body[3], m[3];
    div3(&x[TS_MAG], UT, b_body);
    const uint32_t mode =
        control_update(&s->control, &s->cfg, &x[TS_Q], &x[TS_W], b_body, NULL, &status, m);
    la_copy(3, t->m_prev, m);
    la_copy(4, out->q_est, &x[TS_Q]);
    la_copy(3, out->w_est, &x[TS_W]);
    div3(&t->filter.b[0], UT, out->bias_mag);
    la_copy(3, out->bias_gyro, &t->filter.b[3]);
    for (size_t i = 0; i < ADCS_NX; i++) {
        out->p_diag[i] = LA_AT(t->filter.px, ADCS_NX, i, i);
    }
    for (size_t i = 0; i < ADCS_NB; i++) {
        out->p_diag[ADCS_NX + i] = LA_AT(t->filter.pb, ADCS_NB, i, i);
    }
    la_copy(3, out->m_cmd, m);
    out->mode = mode;
    out->status = status;
}

static void thesis_step(adcs_state *s, const adcs_input *in, adcs_output *out)
{
    const adcs_config *cfg = &s->cfg;
    thesis_state *t = &s->v.thesis;
    uint32_t status = 0;

    /* Sensor readings and on-board reference models are checked separately: in eclipse the
     * sun sensor has no reading but the ephemeris still gives the Sun direction. */
    const bool mag_ref_ok = field_plausible(in->mag_ref_i);
    const bool sun_ref_ok = unit_plausible(in->sun_ref_i);
    const bool mag_ok = in->mag_valid && field_plausible(in->mag_b);
    const bool sun_ok = in->sun_valid && unit_plausible(in->sun_b);
    const bool gyro_ok = in->gyro_valid && gyro_plausible(in->gyro_b);
    if ((in->mag_valid && !mag_ok) || (in->sun_valid && !sun_ok) || (in->gyro_valid && !gyro_ok)) {
        status |= ADCS_STATUS_BAD_INPUT;
    }
    adcs_real mag_ut[3] = {0, 0, 0}, mag_ref_ut[3] = {0, 0, 0};
    adcs_real sun_b[3] = {0, 0, 0}, sun_ref[3] = {0, 0, 0}, gyro[3] = {0, 0, 0};
    if (mag_ok) {
        scale3(in->mag_b, UT, mag_ut);
    }
    if (mag_ref_ok) {
        scale3(in->mag_ref_i, UT, mag_ref_ut);
    }
    /* Directions are normalised: the plausibility check accepts lengths 0.5–2. */
    if (sun_ok) {
        div3(in->sun_b, v3_norm(in->sun_b), sun_b);
    }
    if (sun_ref_ok) {
        div3(in->sun_ref_i, v3_norm(in->sun_ref_i), sun_ref);
    }
    if (gyro_ok) {
        la_copy(3, gyro, in->gyro_b);
    }

    /* Attitude measurement from FLAE (needs two vectors). Without a sun reading, the
     * filter's sun vector, propagated over this step, stands in for it. */
    bool have_q = false;
    adcs_real q_meas[4] = {0, 0, 0, 0};
    adcs_real sun_for_flae[3] = {0, 0, 0};
    bool have_sun = sun_ok;
    if (sun_ok) {
        la_copy(3, sun_for_flae, sun_b);
    } else if (s->initialised) {
        propagate_body_vector(&t->filter.x[TS_SUN], &t->filter.x[TS_W], cfg->dt, sun_for_flae);
        have_sun = true;
    }
    if (mag_ok && mag_ref_ok && sun_ref_ok && have_sun) {
        adcs_real vb[6], vr[6];
        la_copy(3, &vb[0], sun_for_flae);
        div3(mag_ut, v3_norm(mag_ut), &vb[3]);
        la_copy(3, &vr[0], sun_ref);
        div3(mag_ref_ut, v3_norm(mag_ref_ut), &vr[3]);
        const adcs_real weights[2] = {cfg->flae_weight_sun, cfg->flae_weight_mag};
        const adcs_real *prev = s->initialised       ? &t->filter.x[TS_Q]
                                : t->has_last_flae_q ? t->last_flae_q
                                                     : NULL;
        flae_result res;
        if (flae2(vb, vr, weights, prev, cfg->flae_tolerance, cfg->flae_max_iterations, &res) ==
            0) {
            la_copy(4, q_meas, res.q);
            la_copy(4, t->last_flae_q, res.q);
            t->has_last_flae_q = true;
            have_q = true;
        } else {
            status |= ADCS_STATUS_FLAE_FAILED;
        }
    } else {
        status |= ADCS_STATUS_NO_ATTITUDE_FIX;
    }

    if (!s->initialised) {
        /* never start from an attitude that has no real sun reading */
        if (!sun_ok || !have_q || !gyro_ok) {
            output_empty(out, ADCS_MODE_OFF, status);
            return;
        }
        adcs_real x0[TS_NX];
        const adcs_real b0[TS_NB] = {0, 0, 0, 0, 0, 0};
        la_copy(4, &x0[TS_Q], q_meas);
        la_copy(3, &x0[TS_SUN], sun_b);
        la_copy(3, &x0[TS_MAG], mag_ut);
        la_copy(3, &x0[TS_W], gyro);
        tsukf_reset(&t->filter, x0, cfg->p_x0, b0, cfg->p_b0);
        s->initialised = true;
        s->control.has_b_prev = false;
        thesis_finish(s, status, out);
        return;
    }

    process_ctx ctx = {cfg->dt, cfg->process_dt, {0, 0, 0}, t->inertia, t->inertia_inv};
    if (mag_ok) {
        adcs_real b_t[3];
        div3(mag_ut, UT, b_t);
        v3_cross(t->m_prev, b_t, ctx.torque);
    }
    adcs_real y[TS_NY];
    bool available[TS_NY];
    la_copy(4, &y[TS_Q], q_meas);
    la_copy(3, &y[TS_SUN], sun_b);
    la_copy(3, &y[TS_MAG], mag_ut);
    la_copy(3, &y[TS_W], gyro);
    for (size_t i = 0; i < TS_NY; i++) {
        available[i] = i < TS_SUN ? have_q : i < TS_MAG ? sun_ok : i < TS_W ? mag_ok : gyro_ok;
    }
    if (!tsukf_step(&t->filter, process_model, &ctx, y, available)) {
        status |= ADCS_STATUS_COVARIANCE_RESET;
    }
    thesis_finish(s, status, out);
}

/* ---- MEKF variant ---------------------------------------------------------------------- */

/* Control and output. mag_ut/mag_ok: the magnetometer reading, usable for B-dot whether or
 * not the field model is available. */
static void mekf_finish(adcs_state *s, const adcs_input *in, uint32_t status,
                        const adcs_real gyro[3], const adcs_real mag_ut[3], bool mag_ok,
                        adcs_output *out)
{
    const adcs_config *cfg = &s->cfg;
    const mekf *f = &s->v.mekf.filter;
    adcs_real w[3], b_body[3], m[3];
    for (size_t i = 0; i < 3; i++) {
        w[i] = gyro[i] - f->x[MEKF_GYRO + i];
    }
    if (mag_ok) {
        for (size_t i = 0; i < 3; i++) {
            b_body[i] = (mag_ut[i] - f->x[MEKF_MAG + i]) / UT;
        }
    } else if (field_plausible(in->mag_ref_i)) {
        quat_rotate(f->q_ref, in->mag_ref_i, b_body);
    } else {
        la_zero(3, b_body);
    }
    /* B-dot from Ḃ = −ω̂ × B: a fixed inertial field seen from the rotating body. */
    adcs_real b_dot[3];
    v3_cross(w, b_body, b_dot);
    for (size_t i = 0; i < 3; i++) {
        b_dot[i] = -b_dot[i];
    }
    const uint32_t mode =
        control_update(&s->control, cfg, f->q_ref, w, b_body, b_dot, &status, m);
    const adcs_real rate_outage = cfg->sigma_outage_accel * s->v.mekf.outage_time;
    la_copy(4, out->q_est, f->q_ref);
    la_copy(3, out->w_est, w);
    div3(&f->x[MEKF_MAG], UT, out->bias_mag);
    la_copy(3, out->bias_gyro, &f->x[MEKF_GYRO]);
    for (size_t i = 0; i < ADCS_NX + ADCS_NB; i++) {
        out->p_diag[i] = (adcs_real)NAN;
    }
    for (size_t i = 0; i < 3; i++) {
        const adcs_real pg = LA_AT(f->p, MEKF_NX, MEKF_GYRO + i, MEKF_GYRO + i);
        out->p_diag[i] = LA_AT(f->p, MEKF_NX, MEKF_ATT + i, MEKF_ATT + i); /* rad² */
        /* rate = gyro − bias (+ the propagation uncertainty during a gyro outage) */
        out->p_diag[10 + i] =
            cfg->sigma_gyro_noise * cfg->sigma_gyro_noise + pg + rate_outage * rate_outage;
        out->p_diag[13 + i] = LA_AT(f->p, MEKF_NX, MEKF_MAG + i, MEKF_MAG + i);
        out->p_diag[16 + i] = pg;
    }
    la_copy(3, out->m_cmd, m);
    out->mode = mode;
    out->status = status;
}

/* The outage growth stops once an attitude variance reaches this σ² (rad): beyond ~15° the
 * attitude is effectively unknown, and more would only break the linearisation. */
#define OUTAGE_MAX_ATTITUDE_SIGMA ((adcs_real)0.25)

/* Torque-free Euler equations, ω̇ = I⁻¹ (−ω × Iω). */
static void euler_rate(const adcs_real w[3], const adcs_real *inertia,
                       const adcs_real *inertia_inv, adcs_real out[3])
{
    adcs_real iw[3], g[3];
    const adcs_real nw[3] = {-w[0], -w[1], -w[2]};
    la_mul_vec(3, 3, inertia, w, iw);
    v3_cross(nw, iw, g);
    la_mul_vec(3, 3, inertia_inv, g, out);
}

/* Body rate after dt without torques (one RK4 step). */
static void propagate_rate(const adcs_real w[3], adcs_real dt, const adcs_real *inertia,
                           const adcs_real *inertia_inv, adcs_real out[3])
{
    adcs_real k1[3], k2[3], k3[3], k4[3], xt[3];
    euler_rate(w, inertia, inertia_inv, k1);
    for (size_t i = 0; i < 3; i++) {
        xt[i] = w[i] + (adcs_real)0.5 * dt * k1[i];
    }
    euler_rate(xt, inertia, inertia_inv, k2);
    for (size_t i = 0; i < 3; i++) {
        xt[i] = w[i] + (adcs_real)0.5 * dt * k2[i];
    }
    euler_rate(xt, inertia, inertia_inv, k3);
    for (size_t i = 0; i < 3; i++) {
        xt[i] = w[i] + dt * k3[i];
    }
    euler_rate(xt, inertia, inertia_inv, k4);
    for (size_t i = 0; i < 3; i++) {
        out[i] = w[i] + dt * (k1[i] + (adcs_real)2 * k2[i] + (adcs_real)2 * k3[i] + k4[i]) /
                            (adcs_real)6;
    }
}

/* Declare a bias fault when the sun or magnetometer innovation stays too large: inflate the
 * attitude and both bias uncertainties and let the measurements re-estimate them. Only in
 * sunlight, where the unbiased sun sensor can arbitrate. */
static bool detect_fault(adcs_state *s, bool sun_ok)
{
    const adcs_config *cfg = &s->cfg;
    mekf_state *m = &s->v.mekf;
    const adcs_real nis_sun = mekf_nis(&m->filter, 0);
    const adcs_real nis_mag = mekf_nis(&m->filter, 1);
    const adcs_real thr = cfg->fault_nis_threshold;
    const bool suspect = sun_ok && (nis_sun > thr || nis_mag > thr);
    m->suspect_steps = suspect ? m->suspect_steps + 1u : 0u;
    if (m->suspect_steps >= cfg->fault_steps) {
        mekf_inflate(&m->filter, cfg->fault_attitude_sigma, cfg->fault_gyro_bias_sigma,
                     cfg->fault_mag_bias_sigma);
        m->suspect_steps = 0;
        return true;
    }
    return false;
}

static void mekf_variant_step(adcs_state *s, const adcs_input *in, adcs_output *out)
{
    const adcs_config *cfg = &s->cfg;
    mekf_state *m = &s->v.mekf;
    uint32_t status = 0;
    m->filter.has_update = false;

    const bool mag_ref_ok = field_plausible(in->mag_ref_i);
    const bool sun_ref_ok = unit_plausible(in->sun_ref_i);
    const bool mag_plausible = field_plausible(in->mag_b);
    const bool sun_plausible = unit_plausible(in->sun_b);
    const bool mag_ok = in->mag_valid && mag_plausible && mag_ref_ok;
    const bool sun_ok = in->sun_valid && sun_plausible && sun_ref_ok;
    const bool gyro_ok = in->gyro_valid && gyro_plausible(in->gyro_b);
    if ((in->mag_valid && !mag_plausible) || (in->sun_valid && !sun_plausible) ||
        (in->gyro_valid && !gyro_ok)) {
        status |= ADCS_STATUS_BAD_INPUT;
    }
    /* The magnetometer reading itself (for B-dot) needs no reference model. */
    const bool mag_reading_ok = in->mag_valid && mag_plausible;
    adcs_real mag_ut[3] = {0, 0, 0}, mag_meas_ut[3] = {0, 0, 0}, mag_ref_ut[3] = {0, 0, 0};
    adcs_real sun_b[3] = {0, 0, 0}, sun_ref[3] = {0, 0, 0}, gyro[3];
    if (mag_ok) {
        scale3(in->mag_b, UT, mag_ut);
    }
    if (mag_reading_ok) {
        scale3(in->mag_b, UT, mag_meas_ut);
    }
    if (mag_ref_ok) {
        scale3(in->mag_ref_i, UT, mag_ref_ut);
    }
    /* Directions are normalised: the plausibility check accepts lengths 0.5–2. */
    if (sun_ok) {
        div3(in->sun_b, v3_norm(in->sun_b), sun_b);
    }
    if (sun_ref_ok) {
        div3(in->sun_ref_i, v3_norm(in->sun_ref_i), sun_ref);
    }
    la_copy(3, gyro, gyro_ok ? in->gyro_b : m->gyro_prev);
    if (!(sun_ok && mag_ok)) {
        status |= ADCS_STATUS_NO_ATTITUDE_FIX;
    }

    if (!s->initialised) {
        if (!(sun_ok && mag_ok && gyro_ok)) {
            output_empty(out, ADCS_MODE_OFF, status);
            return;
        }
        adcs_real vb[6], vr[6];
        la_copy(3, &vb[0], sun_b);
        div3(mag_ut, v3_norm(mag_ut), &vb[3]);
        la_copy(3, &vr[0], sun_ref);
        div3(mag_ref_ut, v3_norm(mag_ref_ut), &vr[3]);
        const adcs_real weights[2] = {cfg->flae_weight_sun, cfg->flae_weight_mag};
        flae_result res;
        if (flae2(vb, vr, weights, NULL, cfg->flae_tolerance, cfg->flae_max_iterations, &res) !=
            0) {
            output_empty(out, ADCS_MODE_OFF, status | ADCS_STATUS_FLAE_FAILED);
            return;
        }
        adcs_real p0[MEKF_NX];
        for (size_t i = 0; i < 3; i++) {
            p0[MEKF_ATT + i] = cfg->p0_attitude * cfg->p0_attitude;
            p0[MEKF_GYRO + i] = cfg->p0_gyro_bias * cfg->p0_gyro_bias;
            p0[MEKF_MAG + i] = cfg->p0_mag_bias * cfg->p0_mag_bias;
        }
        mekf_reset(&m->filter, res.q, p0);
        s->initialised = true;
        la_copy(3, m->gyro_prev, gyro);
        mekf_finish(s, in, status, gyro, mag_meas_ut, mag_reading_ok, out);
        return;
    }

    /* Without the gyro, propagate the rate with the dynamics, grow the attitude uncertainty
     * with what they leave out, and freeze the (unobservable) gyro-bias estimate. */
    const adcs_real dt = cfg->dt;
    adcs_real outage_var = (adcs_real)0;
    if (gyro_ok) {
        m->outage_time = (adcs_real)0;
    } else {
        adcs_real w_prev[3], w_new[3];
        for (size_t i = 0; i < 3; i++) {
            w_prev[i] = m->gyro_prev[i] - m->filter.x[MEKF_GYRO + i];
        }
        propagate_rate(w_prev, dt, m->inertia, m->inertia_inv, w_new);
        for (size_t i = 0; i < 3; i++) {
            gyro[i] = w_new[i] + m->filter.x[MEKF_GYRO + i];
        }
        const adcs_real t0 = m->outage_time;
        m->outage_time = t0 + dt;
        const adcs_real t1 = m->outage_time, sa = cfg->sigma_outage_accel;
        /* rate error σ_α t → attitude error σ_α t²/2 → variance (σ_α²/4) t⁴ */
        outage_var = (adcs_real)0.25 * sa * sa * (t1 * t1 * t1 * t1 - t0 * t0 * t0 * t0);
        const adcs_real *p = m->filter.p;
        adcs_real largest = LA_AT(p, MEKF_NX, 0, 0);
        for (size_t i = 1; i < 3; i++) {
            largest = LA_AT(p, MEKF_NX, i, i) > largest ? LA_AT(p, MEKF_NX, i, i) : largest;
        }
        const adcs_real room =
            OUTAGE_MAX_ATTITUDE_SIGMA * OUTAGE_MAX_ATTITUDE_SIGMA - largest;
        const adcs_real cap = room > (adcs_real)0 ? room : (adcs_real)0;
        outage_var = outage_var < cap ? outage_var : cap;
        status |= ADCS_STATUS_RATE_PROPAGATED;
    }

    /* Propagate with the mean of the previous and current rate readings: the body rate
     * changes within a step, and holding the previous reading leaves an O(ω̇ dt²) error. */
    adcs_real rate[3];
    for (size_t i = 0; i < 3; i++) {
        rate[i] = (adcs_real)0.5 * (m->gyro_prev[i] + gyro[i]);
    }
    if (!mekf_step(&m->filter, rate, dt, sun_b, sun_ref, sun_ok, mag_ut, mag_ref_ut, mag_ok,
                   outage_var, !gyro_ok)) {
        status |= ADCS_STATUS_COVARIANCE_RESET;
    }
    if (detect_fault(s, sun_ok)) {
        status |= ADCS_STATUS_BIAS_FAULT;
    }
    la_copy(3, m->gyro_prev, gyro);
    mekf_finish(s, in, status, gyro, mag_meas_ut, mag_reading_ok, out);
}

/* ---- API ------------------------------------------------------------------------------- */

static bool aligned(const adcs_state *state)
{
    return (uintptr_t)state % alignof(struct adcs_state) == 0;
}

int adcs_init(adcs_state *state, const adcs_config *config)
{
    if (state == NULL) {
        return -1;
    }
    if (!aligned(state)) {
        return -2;
    }
    state->magic = 0; /* a later adcs_step on a failed init fails cleanly */
    if (config == NULL) {
        return -1;
    }
    if (!config_valid(config)) {
        return -3;
    }
    state->cfg = *config;
    state->initialised = false;
    control_reset(&state->control);
    if (config->variant == ADCS_VARIANT_THESIS) {
        thesis_state *t = &state->v.thesis;
        adcs_real work[9];
        la_copy(9, t->inertia, config->inertia);
        if (la_inverse(3, config->inertia, t->inertia_inv, work) != 0) {
            return -3;
        }
        la_zero(3, t->m_prev);
        la_zero(4, t->last_flae_q);
        t->has_last_flae_q = false;
        tsukf_configure(&t->filter, config->ukf_alpha, config->w_x, config->w_b, config->v);
        const adcs_real x0[ADCS_NX] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const adcs_real b0[ADCS_NB] = {0, 0, 0, 0, 0, 0};
        tsukf_reset(&t->filter, x0, config->p_x0, b0, config->p_b0);
    } else {
        mekf_state *m = &state->v.mekf;
        mekf_configure(&m->filter, config->sigma_gyro_noise, config->sigma_gyro_bias_walk,
                       config->sigma_mag_noise, config->sigma_mag_bias_walk,
                       config->sigma_sun_noise);
        la_zero(3, m->gyro_prev);
        m->suspect_steps = 0;
        m->outage_time = (adcs_real)0;
        adcs_real work[9];
        la_copy(9, m->inertia, config->inertia);
        if (la_inverse(3, config->inertia, m->inertia_inv, work) != 0) {
            return -3;
        }
    }
    state->magic = ADCS_MAGIC;
    return 0;
}

int adcs_step(adcs_state *state, const adcs_input *in, adcs_output *out)
{
    if (out == NULL) {
        return -1;
    }
    if (state == NULL || in == NULL || !aligned(state) || state->magic != ADCS_MAGIC ||
        (state->cfg.variant != ADCS_VARIANT_THESIS && state->cfg.variant != ADCS_VARIANT_MEKF)) {
        output_empty(out, ADCS_MODE_OFF, ADCS_STATUS_NOT_INITIALISED);
        return -1;
    }
    if (state->cfg.variant == ADCS_VARIANT_THESIS) {
        thesis_step(state, in, out);
    } else {
        mekf_variant_step(state, in, out);
    }
    return 0;
}
