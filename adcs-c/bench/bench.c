/*
 * bench: one benchmark driver for every ADCS library that implements adcs.h. Linked against
 * libadcs_c.a or the Rust libadcs_rs.a, so timing code, inputs and the driver's compiler are
 * identical and only the library differs (docs/results/c_vs_rust.md).
 *
 *     bench --label NAME [--reps N] vector.json [...]
 *
 * For the thesis and MEKF vectors separately, prints one JSON object per line:
 * - accuracy against the recorded Python reference (double precision): largest and RMS
 *   attitude difference, largest rate difference, and an FNV-1a hash of every output byte
 *   (equal hashes mean bit-identical outputs);
 * - time per adcs_step over N repetitions of every vector after a warm-up: mean, p50, p99,
 *   p99.9, max;
 * - peak stack used by adcs_init and by adcs_step, measured on a thread whose stack is
 *   pre-filled with a pattern (0 in sanitizer builds, whose locals live elsewhere);
 * - the largest attitude difference per vector.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "adcs.h"
#include "golden_io.h"

#define MAX_VECTORS 32
#define WARMUP_REPS 20
#define STACK_BYTES (1u << 20)
#define STACK_FILL 0xA5

typedef struct {
    golden_vector g;
    adcs_input *inputs;
    const char *name;
} vector;

static void *state_memory(void)
{
    void *mem = aligned_alloc(alignof(max_align_t), (adcs_state_size() + 63) / 64 * 64);
    if (mem == NULL) {
        abort();
    }
    return mem;
}

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ---- accuracy ---------------------------------------------------------------------- */

typedef struct {
    double max_att_deg, sum_sq_att, max_rate;
    size_t n_att;
    uint64_t hash;
} accuracy;

static void fnv1a(uint64_t *h, const void *data, size_t n)
{
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++) {
        *h = (*h ^ p[i]) * 0x100000001b3ull;
    }
}

static void run_accuracy(const vector *v, accuracy *acc)
{
    adcs_state *s = state_memory();
    if (adcs_init(s, &v->g.config) != 0) {
        abort();
    }
    for (size_t k = 0; k < v->g.steps; k++) {
        adcs_output out;
        adcs_step(s, &v->inputs[k], &out);
        fnv1a(&acc->hash, out.q_est, sizeof out.q_est);
        fnv1a(&acc->hash, out.w_est, sizeof out.w_est);
        fnv1a(&acc->hash, out.bias_mag, sizeof out.bias_mag);
        fnv1a(&acc->hash, out.bias_gyro, sizeof out.bias_gyro);
        fnv1a(&acc->hash, out.p_diag, sizeof out.p_diag);
        fnv1a(&acc->hash, out.m_cmd, sizeof out.m_cmd);
        fnv1a(&acc->hash, &out.mode, sizeof out.mode);
        fnv1a(&acc->hash, &out.status, sizeof out.status);
        const double *ref = v->g.records + k * GOLDEN_RECORD_SIZE + GOLDEN_INPUT_SIZE;
        if (isnan(ref[0]) || isnan((double)out.q_est[0])) {
            continue;
        }
        /* angle of q_est ⊗ q_ref⁻¹, by atan2: acos(q·q_ref) loses all precision near 0 */
        const double q[4] = {(double)out.q_est[0], (double)out.q_est[1], (double)out.q_est[2],
                             (double)out.q_est[3]};
        const double r[4] = {-ref[0], -ref[1], -ref[2], ref[3]};
        const double qv[3] = {
            q[3] * r[0] + r[3] * q[0] - (q[1] * r[2] - q[2] * r[1]),
            q[3] * r[1] + r[3] * q[1] - (q[2] * r[0] - q[0] * r[2]),
            q[3] * r[2] + r[3] * q[2] - (q[0] * r[1] - q[1] * r[0]),
        };
        const double w = q[3] * r[3] - (q[0] * r[0] + q[1] * r[1] + q[2] * r[2]);
        const double angle =
            2.0 * atan2(sqrt(qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2]), fabs(w)) *
                             180.0 / 3.14159265358979323846;
        acc->max_att_deg = fmax(acc->max_att_deg, angle);
        acc->sum_sq_att += angle * angle;
        acc->n_att++;
        for (int i = 0; i < 3; i++) {
            acc->max_rate = fmax(acc->max_rate, fabs(ref[4 + i] - (double)out.w_est[i]));
        }
    }
    free(s);
}

/* ---- timing ------------------------------------------------------------------------ */

static size_t run_timing(const vector *v, double *samples)
{
    adcs_state *s = state_memory();
    if (adcs_init(s, &v->g.config) != 0) {
        abort();
    }
    adcs_output out;
    for (size_t k = 0; k < v->g.steps; k++) {
        const double t0 = now_ns();
        adcs_step(s, &v->inputs[k], &out);
        const double t1 = now_ns();
        if (samples != NULL) {
            samples[k] = t1 - t0;
        }
    }
    free(s);
    return v->g.steps;
}

/* ---- stack ------------------------------------------------------------------------ */

/*
 * Peak stack of adcs_init and of adcs_step, measured on a thread whose stack is a buffer
 * filled with a pattern: the deepest overwritten byte, relative to the thread function's
 * own frame. Between init and the steps the unused part of the stack (below the current
 * frame, minus a margin for the x86-64 red zone) is filled again, so the two are separate.
 * The state memory is allocated outside the thread, so malloc's own stack use is excluded.
 */
typedef struct {
    const vector *v;
    size_t count;
    adcs_state *state;
    unsigned char *stack;
    size_t init_bytes, step_bytes;
} stack_job;

#define RED_ZONE 512u

/* Bytes used below offset `frame` of the stack buffer (the stack grows down). */
static size_t deepest_use(const unsigned char *stack, size_t frame)
{
    size_t untouched = 0;
    while (untouched < frame && stack[untouched] == STACK_FILL) {
        untouched++;
    }
    return frame - untouched;
}

static void *stack_thread(void *arg)
{
    stack_job *job = arg;
    volatile unsigned char marker = 0;
    const uintptr_t at = (uintptr_t)&marker, lo = (uintptr_t)job->stack;
    if (at < lo + RED_ZONE || at >= lo + STACK_BYTES) {
        /* locals are not on this stack (AddressSanitizer's fake stack): cannot measure */
        job->init_bytes = job->step_bytes = 0;
        return NULL;
    }
    const size_t frame = (size_t)(at - lo) - RED_ZONE;
    if (adcs_init(job->state, &job->v[0].g.config) != 0) {
        abort();
    }
    job->init_bytes = deepest_use(job->stack, frame);
    memset(job->stack, STACK_FILL, frame);
    for (size_t i = 0; i < job->count; i++) {
        if (i > 0) {
            /* re-initialising is init again: measure only the steps */
            adcs_init(job->state, &job->v[i].g.config);
            memset(job->stack, STACK_FILL, frame);
        }
        for (size_t k = 0; k < job->v[i].g.steps; k++) {
            adcs_output out;
            adcs_step(job->state, &job->v[i].inputs[k], &out);
        }
        const size_t used = deepest_use(job->stack, frame);
        if (used > job->step_bytes) {
            job->step_bytes = used;
        }
    }
    return NULL;
}

static void stack_used(stack_job *job)
{
    job->stack = aligned_alloc(4096, STACK_BYTES);
    job->state = state_memory();
    if (job->stack == NULL) {
        abort();
    }
    memset(job->stack, STACK_FILL, STACK_BYTES);
    pthread_attr_t attr;
    pthread_t th;
    pthread_attr_init(&attr);
    pthread_attr_setstack(&attr, job->stack, STACK_BYTES);
    if (pthread_create(&th, &attr, stack_thread, job) != 0) {
        abort();
    }
    pthread_join(th, NULL);
    pthread_attr_destroy(&attr);
    free(job->stack);
    free(job->state);
}

/* ---- main -------------------------------------------------------------------------- */

static void report(const char *label, const char *variant, const vector *vs, size_t count,
                   int reps)
{
    if (count == 0) {
        return;
    }
    accuracy acc = {0.0, 0.0, 0.0, 0, 0xcbf29ce484222325ull};
    size_t total_steps = 0;
    char per_vector[2048] = "";
    for (size_t i = 0; i < count; i++) {
        accuracy one = {0.0, 0.0, 0.0, 0, 0};
        run_accuracy(&vs[i], &one);
        run_accuracy(&vs[i], &acc);
        char item[128];
        snprintf(item, sizeof item, "%s\"%s\": %.3e", i ? ", " : "", vs[i].name, one.max_att_deg);
        strncat(per_vector, item, sizeof per_vector - strlen(per_vector) - 1);
        total_steps += vs[i].g.steps;
    }
    for (int r = 0; r < WARMUP_REPS; r++) {
        for (size_t i = 0; i < count; i++) {
            run_timing(&vs[i], NULL);
        }
    }
    const size_t n = total_steps * (size_t)reps;
    double *samples = malloc(n * sizeof(double));
    if (samples == NULL) {
        abort();
    }
    size_t at = 0;
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i < count; i++) {
            at += run_timing(&vs[i], samples + at);
        }
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += samples[i];
    }
    qsort(samples, n, sizeof(double), compare_double);
    stack_job job = {vs, count, NULL, NULL, 0, 0};
    stack_used(&job);

    printf("{\"label\": \"%s\", \"variant\": \"%s\", \"steps\": %zu, \"state_bytes\": %zu, "
           "\"mean_us\": %.4f, \"p50_us\": %.4f, \"p99_us\": %.4f, \"p999_us\": %.4f, "
           "\"max_us\": %.4f, \"stack_init_bytes\": %zu, \"stack_step_bytes\": %zu, "
           "\"max_att_deg\": %.3e, "
           "\"rms_att_deg\": %.3e, \"max_rate\": %.3e, \"hash\": \"%016llx\", "
           "\"max_att_deg_per_vector\": {%s}}\n",
           label, variant, n, adcs_state_size(), sum / (double)n * 1e-3,
           samples[n / 2] * 1e-3, samples[(size_t)((double)n * 0.99)] * 1e-3,
           samples[(size_t)((double)n * 0.999)] * 1e-3, samples[n - 1] * 1e-3,
           job.init_bytes, job.step_bytes, acc.max_att_deg,
           acc.n_att ? sqrt(acc.sum_sq_att / (double)acc.n_att) : 0.0, acc.max_rate,
           (unsigned long long)acc.hash, per_vector);
    fflush(stdout);
    free(samples);
}

int main(int argc, char **argv)
{
    const char *label = "unnamed";
    int reps = 100;
    static vector thesis[MAX_VECTORS], mekf[MAX_VECTORS];
    size_t n_thesis = 0, n_mekf = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--label") == 0 && i + 1 < argc) {
            label = argv[++i];
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else {
            vector v;
            const char *slash = strrchr(argv[i], '/');
            v.name = slash ? slash + 1 : argv[i];
            if (golden_load(argv[i], &v.g) != 0) {
                return 1;
            }
            v.inputs = malloc(v.g.steps * sizeof(adcs_input));
            if (v.inputs == NULL) {
                return 1;
            }
            for (size_t k = 0; k < v.g.steps; k++) {
                golden_record_to_input(v.g.records + k * GOLDEN_RECORD_SIZE, &v.inputs[k]);
            }
            const int is_mekf = v.g.config.variant == ADCS_VARIANT_MEKF;
            size_t *n = is_mekf ? &n_mekf : &n_thesis;
            if (*n == MAX_VECTORS) {
                free(v.inputs);
                golden_free(&v.g);
                return 1;
            }
            (is_mekf ? mekf : thesis)[(*n)++] = v;
        }
    }
    if (adcs_abi_version() != ADCS_ABI_VERSION || reps < 1) {
        fprintf(stderr, "ABI mismatch or bad --reps\n");
        return 1;
    }
    report(label, "thesis", thesis, n_thesis, reps);
    report(label, "mekf", mekf, n_mekf, reps);
    return 0;
}
