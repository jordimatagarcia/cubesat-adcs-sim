#include "quat.h"

#include <math.h>

#include "linalg.h"

void quat_mul(const adcs_real q[4], const adcs_real p[4], adcs_real out[4])
{
    adcs_real c[3];
    v3_cross(q, p, c);
    const adcs_real x = q[3] * p[0] + p[3] * q[0] - c[0];
    const adcs_real y = q[3] * p[1] + p[3] * q[1] - c[1];
    const adcs_real z = q[3] * p[2] + p[3] * q[2] - c[2];
    const adcs_real s = q[3] * p[3] - v3_dot(q, p);
    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = s;
}

void quat_conj(const adcs_real q[4], adcs_real out[4])
{
    out[0] = -q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = q[3];
}

adcs_real quat_norm(const adcs_real q[4])
{
    return LA_SQRT(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

void quat_normalize(adcs_real q[4])
{
    const adcs_real n = quat_norm(q);
    if (n > (adcs_real)0) {
        for (int i = 0; i < 4; i++) {
            q[i] /= n;
        }
    }
}

void quat_to_dcm(const adcs_real q[4], adcs_real a[9])
{
    const adcs_real q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
    const adcs_real two = (adcs_real)2;
    a[0] = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
    a[1] = two * (q1 * q2 + q3 * q4);
    a[2] = two * (q1 * q3 - q2 * q4);
    a[3] = two * (q1 * q2 - q3 * q4);
    a[4] = -q1 * q1 + q2 * q2 - q3 * q3 + q4 * q4;
    a[5] = two * (q2 * q3 + q1 * q4);
    a[6] = two * (q1 * q3 + q2 * q4);
    a[7] = two * (q2 * q3 - q1 * q4);
    a[8] = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4;
}

void quat_rotate(const adcs_real q[4], const adcs_real v[3], adcs_real out[3])
{
    adcs_real a[9];
    quat_to_dcm(q, a);
    la_mul_vec(3, 3, a, v, out);
}

void quat_from_rate(const adcs_real w[3], adcs_real dt, adcs_real out[4])
{
    const adcs_real rate = v3_norm(w);
    const adcs_real half = (adcs_real)0.5 * rate * dt;
    /* sin(x)/x expanded near 0 to avoid dividing by zero */
    const adcs_real sinc = rate > (adcs_real)1e-12 ? LA_SIN(half) / rate : (adcs_real)0.5 * dt;
    out[0] = w[0] * sinc;
    out[1] = w[1] * sinc;
    out[2] = w[2] * sinc;
    out[3] = LA_COS(half);
}
