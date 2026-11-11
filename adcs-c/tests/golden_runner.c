/*
 * golden_runner: replays golden vectors (testdata/golden/<variant>/<name>.json + .f64)
 * through the C library and compares every output with the recorded Python reference.
 *
 *     golden_runner path/to/vector.json [more.json ...]
 *
 * Tolerances are the ones in src/adcssim/golden.py (TOLERANCES).
 */
#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "adcs.h"
#include "golden_io.h"

#define INPUT_SIZE GOLDEN_INPUT_SIZE
#define OUTPUT_SIZE GOLDEN_OUTPUT_SIZE
#define RECORD_SIZE GOLDEN_RECORD_SIZE

/* ---- comparison -------------------------------------------------------------------- */

#define N_FIELDS 6 /* compared output fields */
#define Q_OFFSET 0 /* q_est within the output record */

#ifndef ADCS_USE_FLOAT
/* |actual − expected| ≤ atol + rtol·|expected|: the TOLERANCES of src/adcssim/golden.py. */
typedef struct {
    const char *name;
    size_t offset; /* within the output record */
    size_t count;
    double atol, rtol;
} output_field;

static const output_field OUTPUT_FIELDS[] = {
    {"q_est", 0, 4, 1e-9, 0},          {"w_est", 4, 3, 1e-9, 0},
    {"bias_mag", 7, 3, 1e-16, 1e-6},   {"bias_gyro", 10, 3, 1e-9, 0},
    {"p_diag", 13, 19, 1e-18, 1e-6},   {"m_cmd", 32, 3, 1e-9, 0},
};
#endif

/* Single-precision builds cannot match the double-precision reference to these tolerances,
 * nor exactly in mode and status; they are checked on the attitude alone. */
#define FLOAT_MAX_ATTITUDE_DEG 0.1

#ifdef ADCS_USE_FLOAT
/* Angle (deg) between two scalar-last quaternions, from the relative quaternion. */
static double attitude_difference_deg(const double *q, const double *r)
{
    const double v0 = q[3] * -r[0] + r[3] * q[0] - (q[1] * -r[2] - q[2] * -r[1]);
    const double v1 = q[3] * -r[1] + r[3] * q[1] - (q[2] * -r[0] - q[0] * -r[2]);
    const double v2 = q[3] * -r[2] + r[3] * q[2] - (q[0] * -r[1] - q[1] * -r[0]);
    const double w = q[3] * r[3] + (q[0] * r[0] + q[1] * r[1] + q[2] * r[2]);
    return 2.0 * atan2(sqrt(v0 * v0 + v1 * v1 + v2 * v2), fabs(w)) * 57.29577951308232;
}

#else
/* 0 if the value is within tolerance (equal values, NaN and infinities included). */
static int value_differs(double e, double a, const output_field *of)
{
    if (e == a || (isnan(e) && isnan(a))) {
        return 0;
    }
    if (!isfinite(e) || !isfinite(a)) {
        return 1;
    }
    return !(fabs(e - a) <= of->atol + of->rtol * fabs(e));
}
#endif

static int run_vector(const char *json_path)
{
    golden_vector g;
    if (golden_load(json_path, &g) != 0) {
        return 1;
    }
    const adcs_config cfg = g.config;
    const size_t steps = g.steps;
    const double *data = g.records;

    void *mem = aligned_alloc(alignof(max_align_t), (adcs_state_size() + 63) / 64 * 64);
    adcs_state *state = (adcs_state *)mem;
    int failures = 0;
    double max_diff[N_FIELDS] = {0};
    if (mem == NULL || adcs_init(state, &cfg) != 0) {
        fprintf(stderr, "%s: adcs_init failed\n", json_path);
        failures = 1;
    }
    for (size_t k = 0; k < steps && failures == 0; k++) {
        const double *rec = data + k * RECORD_SIZE;
        const double *expected = rec + INPUT_SIZE;
        adcs_input in;
        adcs_output out;
        double actual[OUTPUT_SIZE];
        golden_record_to_input(rec, &in);
        if (adcs_step(state, &in, &out) != 0) {
            fprintf(stderr, "%s: adcs_step failed at step %zu\n", json_path, k);
            failures++;
            break;
        }
        golden_output_to_record(&out, actual);
#ifdef ADCS_USE_FLOAT
        const size_t qo = Q_OFFSET;
        const int nan_e = isnan(expected[qo]), nan_a = isnan(actual[qo]);
        if (nan_e != nan_a) {
            if (failures < 10) {
                fprintf(stderr, "%s: step %zu: estimate present %d, expected %d\n", json_path, k,
                        !nan_a, !nan_e);
            }
            failures++;
        } else if (!nan_e) {
            const double angle = attitude_difference_deg(actual + qo, expected + qo);
            max_diff[0] = angle > max_diff[0] ? angle : max_diff[0];
            if (!(angle <= FLOAT_MAX_ATTITUDE_DEG)) {
                if (failures < 10) {
                    fprintf(stderr, "%s: step %zu (t = %.1f s): attitude %.3g deg off\n",
                            json_path, k, rec[0], angle);
                }
                failures++;
            }
        }
#else
        for (size_t fi = 0; fi < sizeof OUTPUT_FIELDS / sizeof OUTPUT_FIELDS[0]; fi++) {
            const output_field *of = &OUTPUT_FIELDS[fi];
            for (size_t i = of->offset; i < of->offset + of->count; i++) {
                if (value_differs(expected[i], actual[i], of)) {
                    if (failures < 10) {
                        fprintf(stderr, "%s: step %zu (t = %.1f s): %s[%zu] = %.17g, expected %.17g\n",
                                json_path, k, rec[0], of->name, i - of->offset, actual[i],
                                expected[i]);
                    }
                    failures++;
                } else if (isfinite(expected[i]) && fabs(expected[i] - actual[i]) > max_diff[fi]) {
                    max_diff[fi] = fabs(expected[i] - actual[i]);
                }
            }
        }
        if (actual[35] != expected[35] || actual[36] != expected[36]) {
            if (failures < 10) {
                fprintf(stderr, "%s: step %zu (t = %.1f s): mode/status %g/%#x, expected %g/%#x\n",
                        json_path, k, rec[0], actual[35], (unsigned)actual[36], expected[35],
                        (unsigned)expected[36]);
            }
            failures++;
        }
#endif
    }
    printf("%-50s %s", json_path, failures == 0 ? "ok" : "FAILED");
#ifdef ADCS_USE_FLOAT
    printf("  attitude %.1e deg (float build)", max_diff[0]);
#else
    for (size_t fi = 0; fi < sizeof OUTPUT_FIELDS / sizeof OUTPUT_FIELDS[0]; fi++) {
        printf("  %s %.1e", OUTPUT_FIELDS[fi].name, max_diff[fi]);
    }
#endif
    printf("\n");
    free(mem);
    golden_free(&g);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s vector.json [...]\n", argv[0]);
        return 2;
    }
    int failed = 0;
    for (int i = 1; i < argc; i++) {
        failed += run_vector(argv[i]);
    }
    return failed == 0 ? 0 : 1;
}
