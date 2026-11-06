/*
 * flae.h: Fast Linear Attitude Estimator, J. Wu et al. (2018), for two vector pairs
 * (internal). Port of src/adcssim/adcs_ref/flae.py; see that file for the derivation and the
 * two deviations from the thesis text (full-pivoting elimination, normalised weights).
 */
#ifndef ADCS_FLAE_H
#define ADCS_FLAE_H

#include <stdint.h>

#include "adcs.h"

typedef struct {
    adcs_real q[4];       /* scalar-last, inertial -> body */
    adcs_real lambda_max; /* largest eigenvalue of W */
    uint32_t iterations;  /* Newton iterations used */
} flae_result;

/*
 * Attitude from two observations b_i = A r_i (v_body and v_ref hold the two vectors one
 * after the other). The sign of q is
 * chosen closest to q_prev when it is given and finite, otherwise with q4 >= 0.
 * Returns 0, or -1 if the geometry is degenerate or Newton's method did not converge.
 */
int flae2(const adcs_real v_body[6], const adcs_real v_ref[6], const adcs_real weights[2],
          const adcs_real *q_prev, adcs_real tolerance, uint32_t max_iterations,
          flae_result *out);

/* Non-zero solution of n x = 0 for a rank-3 4×4 matrix (full pivoting). Returns 0, or -1 if
 * the rank is lower. n is overwritten. Exposed for the unit tests. */
int flae_null_vector(adcs_real n[16], adcs_real out[4]);

#endif /* ADCS_FLAE_H */
