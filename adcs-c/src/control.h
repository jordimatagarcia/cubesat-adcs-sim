/*
 * control.h: mode manager and magnetorquer control laws (internal), shared by both
 * estimator variants. Port of src/adcssim/adcs_ref/control.py (thesis §4.2.3).
 */
#ifndef ADCS_CONTROL_H
#define ADCS_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

#include "adcs.h"

typedef struct {
    uint32_t mode; /* adcs_mode */
    adcs_real mode_time;
    adcs_real b_prev[3];
    bool has_b_prev;
} control_state;

void control_reset(control_state *c);

/* New mode and dipole command from the current estimate (b_body in tesla). Returns the
 * mode; adds ADCS_STATUS_DIPOLE_SATURATED to *status when the command was scaled down. */
uint32_t control_update(control_state *c, const adcs_config *cfg, const adcs_real q[4],
                        const adcs_real w[3], const adcs_real b_body[3], uint32_t *status,
                        adcs_real m_out[3]);

/* Exposed for the unit tests. */
void control_quaternion_feedback(const adcs_real q[4], const adcs_real q_target[4],
                                 const adcs_real w[3], adcs_real kp, adcs_real kd,
                                 adcs_real tau[3]);
void control_torque_to_dipole(const adcs_real tau[3], const adcs_real b[3], adcs_real m[3]);
bool control_saturate(adcs_real m[3], const adcs_real max_dipole[3]);

#endif /* ADCS_CONTROL_H */
