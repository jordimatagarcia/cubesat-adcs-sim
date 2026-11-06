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

typedef struct {
    const char *name;
    size_t offset; /* within the output record */
    size_t count;
    double tolerance;
} output_field;

static const output_field OUTPUT_FIELDS[] = {
    {"q_est", 0, 4, 1e-9},     {"w_est", 4, 3, 1e-9},    {"bias_mag", 7, 3, 1e-13},
    {"bias_gyro", 10, 3, 1e-9}, {"p_diag", 13, 19, 1e-9}, {"m_cmd", 32, 3, 1e-9},
};

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
    double max_diff[sizeof OUTPUT_FIELDS / sizeof OUTPUT_FIELDS[0]] = {0};
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
        for (size_t fi = 0; fi < sizeof OUTPUT_FIELDS / sizeof OUTPUT_FIELDS[0]; fi++) {
            const output_field *of = &OUTPUT_FIELDS[fi];
            for (size_t i = of->offset; i < of->offset + of->count; i++) {
                const int nan_e = isnan(expected[i]), nan_a = isnan(actual[i]);
                const double d = fabs(expected[i] - actual[i]);
                if (nan_e != nan_a || (!nan_e && !(d <= of->tolerance))) {
                    if (failures < 10) {
                        fprintf(stderr, "%s: step %zu (t = %.1f s): %s[%zu] = %.17g, expected %.17g\n",
                                json_path, k, rec[0], of->name, i - of->offset, actual[i],
                                expected[i]);
                    }
                    failures++;
                } else if (!nan_e && d > max_diff[fi]) {
                    max_diff[fi] = d;
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
    }
    printf("%-50s %s", json_path, failures == 0 ? "ok" : "FAILED");
    for (size_t fi = 0; fi < sizeof OUTPUT_FIELDS / sizeof OUTPUT_FIELDS[0]; fi++) {
        printf("  %s %.1e", OUTPUT_FIELDS[fi].name, max_diff[fi]);
    }
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
