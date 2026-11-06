/*
 * quat.h: quaternion helpers (internal). Scalar-last, A(q ⊗ p) = A(q) A(p), A(q) maps
 * inertial to body coordinates (docs/conventions.md).
 */
#ifndef ADCS_QUAT_H
#define ADCS_QUAT_H

#include "adcs.h"

void quat_mul(const adcs_real q[4], const adcs_real p[4], adcs_real out[4]);
void quat_conj(const adcs_real q[4], adcs_real out[4]);
adcs_real quat_norm(const adcs_real q[4]);
/* Divides by the norm; leaves q unchanged if the norm is zero. */
void quat_normalize(adcs_real q[4]);
/* Attitude matrix A(q), row-major. */
void quat_to_dcm(const adcs_real q[4], adcs_real a[9]);
/* A(q) v */
void quat_rotate(const adcs_real q[4], const adcs_real v[3], adcs_real out[3]);
/* Quaternion of a body rotating at the constant rate w for dt (exact). */
void quat_from_rate(const adcs_real w[3], adcs_real dt, adcs_real out[4]);

#endif /* ADCS_QUAT_H */
