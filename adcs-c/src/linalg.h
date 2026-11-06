/*
 * linalg.h: fixed-size dense linear algebra for the ADCS (internal).
 *
 * Matrices are row-major arrays of adcs_real with their dimensions passed explicitly. No
 * function allocates memory; callers provide every output and any workspace. Sizes are
 * bounded by LA_MAX (the largest matrix in the library is the thesis filter's 19×19).
 */
#ifndef ADCS_LINALG_H
#define ADCS_LINALG_H

#include <stddef.h>

#include "adcs.h"

#define LA_MAX 27 /* largest dimension handled (sigma-point count of the thesis filter) */

#ifdef ADCS_USE_FLOAT
#define LA_SQRT sqrtf
#define LA_FABS fabsf
#define LA_SIN sinf
#define LA_COS cosf
#define LA_RINT rintf
#define LA_EPS 1.1920929e-7f
#else
#define LA_SQRT sqrt
#define LA_FABS fabs
#define LA_SIN sin
#define LA_COS cos
#define LA_RINT rint
#define LA_EPS 2.220446049250313e-16
#endif

#define LA_AT(a, cols, i, j) ((a)[(size_t)(i) * (size_t)(cols) + (size_t)(j)])

void la_copy(size_t n, adcs_real *dst, const adcs_real *src);
void la_zero(size_t n, adcs_real *dst);
void la_eye(size_t n, adcs_real *dst);
int la_all_finite(size_t n, const adcs_real *a);

/* c = a·b, a is m×k, b is k×n. c must not alias a or b. */
void la_mul(size_t m, size_t k, size_t n, const adcs_real *a, const adcs_real *b, adcs_real *c);
/* c = a·bᵀ, a is m×k, b is n×k. */
void la_mul_bt(size_t m, size_t k, size_t n, const adcs_real *a, const adcs_real *b,
               adcs_real *c);
/* y = a·x, a is m×n. */
void la_mul_vec(size_t m, size_t n, const adcs_real *a, const adcs_real *x, adcs_real *y);
/* aᵀ (m×n → n×m). */
void la_transpose(size_t m, size_t n, const adcs_real *a, adcs_real *at);
/* a ← (a + aᵀ) / 2 for an n×n matrix. */
void la_symmetrize(size_t n, adcs_real *a);

/* Lower Cholesky factor of an n×n matrix. Returns 0, or -1 if it is not positive definite. */
int la_cholesky(size_t n, const adcs_real *a, adcs_real *lower);

/* Solve a·x = b for n×n a and n×m b by Gaussian elimination with partial pivoting.
 * a and b are overwritten (b becomes x). Returns 0, or -1 if a is singular. */
int la_solve(size_t n, size_t m, adcs_real *a, adcs_real *b);

/* x = b·a⁻¹ for an n×n a and m×n b ("right division", like MATLAB b/a), using la_solve on
 * the transposes. work needs n*n + n*m entries. */
int la_right_divide(size_t m, size_t n, const adcs_real *b, const adcs_real *a, adcs_real *x,
                    adcs_real *work);

/* Inverse of an n×n matrix; work needs n*n entries. Returns 0, or -1 if singular. */
int la_inverse(size_t n, const adcs_real *a, adcs_real *inv, adcs_real *work);

/* Determinant of an n×n matrix (LU with partial pivoting); work needs n*n entries. */
adcs_real la_det(size_t n, const adcs_real *a, adcs_real *work);

/* 3-vectors */
void v3_cross(const adcs_real a[3], const adcs_real b[3], adcs_real out[3]);
adcs_real v3_dot(const adcs_real a[3], const adcs_real b[3]);
adcs_real v3_norm(const adcs_real a[3]);
void v3_skew(const adcs_real v[3], adcs_real m[9]);

#endif /* ADCS_LINALG_H */
