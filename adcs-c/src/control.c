#include "control.h"

#include <math.h>

#include "linalg.h"
#include "quat.h"

void control_reset(control_state *c)
{
    c->mode = ADCS_MODE_ESTIMATE;
    c->mode_time = (adcs_real)0;
    la_zero(3, c->b_prev);
    c->has_b_prev = false;
}

/* eq. 119: τ = −kp sign(δq4) δq1:3 − kd ω, with δq = q ⊗ q_d⁻¹ (eq. 116) */
void control_quaternion_feedback(const adcs_real q[4], const adcs_real q_target[4],
                                 const adcs_real w[3], adcs_real kp, adcs_real kd,
                                 adcs_real tau[3])
{
    adcs_real qc[4], dq[4];
    quat_conj(q_target, qc);
    quat_mul(q, qc, dq);
    const adcs_real sign = dq[3] >= (adcs_real)0 ? (adcs_real)1 : (adcs_real)-1;
    for (int i = 0; i < 3; i++) {
        tau[i] = -kp * sign * dq[i] - kd * w[i];
    }
}

/* eq. 121: m = (B × τ) / |B|² */
void control_torque_to_dipole(const adcs_real tau[3], const adcs_real b[3], adcs_real m[3])
{
    const adcs_real b2 = v3_dot(b, b);
    if (b2 == (adcs_real)0) {
        la_zero(3, m);
        return;
    }
    v3_cross(b, tau, m);
    for (int i = 0; i < 3; i++) {
        m[i] /= b2;
    }
}

/* Scale the whole vector so no coil exceeds its limit, keeping the direction. A NaN
 * command propagates (and is reported as saturated), like the reference. */
bool control_saturate(adcs_real m[3], const adcs_real max_dipole[3])
{
    adcs_real ratio = LA_FABS(m[0]) / max_dipole[0];
    for (int i = 1; i < 3; i++) {
        const adcs_real r = LA_FABS(m[i]) / max_dipole[i];
        if (isnan(r) || r > ratio) {
            ratio = isnan(ratio) ? ratio : r;
        }
    }
    if (ratio <= (adcs_real)1) {
        return false;
    }
    for (int i = 0; i < 3; i++) {
        m[i] /= ratio;
    }
    return true;
}

uint32_t control_update(control_state *c, const adcs_config *cfg, const adcs_real q[4],
                        const adcs_real w[3], const adcs_real b_body[3], uint32_t *status,
                        adcs_real m_out[3])
{
    const adcs_real rate = v3_norm(w);
    uint32_t mode;
    if (*status & (ADCS_STATUS_BAD_INPUT | ADCS_STATUS_COVARIANCE_RESET)) {
        mode = ADCS_MODE_SAFE;
    } else if (!cfg->control_enabled) {
        mode = ADCS_MODE_ESTIMATE;
    } else if (c->mode != ADCS_MODE_DETUMBLE) {
        if (rate > cfg->detumble_enter_rate) {
            mode = ADCS_MODE_DETUMBLE;
        } else if (c->mode == ADCS_MODE_POINT ||
                   (c->mode == ADCS_MODE_ESTIMATE && c->mode_time >= cfg->settle_time)) {
            mode = ADCS_MODE_POINT;
        } else {
            mode = ADCS_MODE_ESTIMATE;
        }
    } else {
        mode = rate < cfg->detumble_exit_rate ? ADCS_MODE_ESTIMATE : ADCS_MODE_DETUMBLE;
    }
    c->mode_time = mode == c->mode ? c->mode_time + cfg->dt : (adcs_real)0;
    c->mode = mode;

    la_zero(3, m_out);
    if (mode == ADCS_MODE_DETUMBLE && c->has_b_prev) {
        /* B-dot, eq. 114: m = −k dB/dt */
        for (int i = 0; i < 3; i++) {
            m_out[i] = -cfg->bdot_gain * ((b_body[i] - c->b_prev[i]) / cfg->dt);
        }
    } else if (mode == ADCS_MODE_POINT) {
        adcs_real tau[3];
        control_quaternion_feedback(q, cfg->q_target, w, cfg->kp, cfg->kd, tau);
        control_torque_to_dipole(tau, b_body, m_out);
    }
    if (control_saturate(m_out, cfg->max_dipole)) {
        *status |= ADCS_STATUS_DIPOLE_SATURATED;
    }
    la_copy(3, c->b_prev, b_body);
    c->has_b_prev = true;
    return mode;
}
