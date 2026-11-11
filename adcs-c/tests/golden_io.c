/*
 * golden_io.c: reads golden vectors (testdata/golden/<variant>/<name>.json + .f64), for the
 * golden runner and the benchmark. The JSON reader only handles what
 * src/adcssim/golden.py writes: it looks up each config key and parses a number, a
 * boolean, a string or an array of numbers after it.
 */
#include "golden_io.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- minimal JSON lookup ----------------------------------------------------------- */

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = size >= 0 ? malloc((size_t)size + 1) : NULL;
    if (buf != NULL && fread(buf, 1, (size_t)size, f) == (size_t)size) {
        buf[size] = '\0';
    } else {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    return buf;
}

/* Pointer to the value after "key": (first occurrence), or NULL. */
static const char *json_value(const char *text, const char *key)
{
    char pattern[96];
    snprintf(pattern, sizeof pattern, "\"%s\"", key);
    for (const char *p = strstr(text, pattern); p != NULL; p = strstr(p + 1, pattern)) {
        const char *v = p + strlen(pattern);
        while (*v == ' ' || *v == '\n' || *v == '\t' || *v == '\r') {
            v++;
        }
        if (*v == ':') {
            v++;
            while (*v == ' ' || *v == '\n' || *v == '\t' || *v == '\r') {
                v++;
            }
            return v;
        }
    }
    return NULL;
}

/* Parses `count` numbers (a scalar, or an array of exactly `count`). Returns 0 on success. */
static int json_numbers(const char *text, const char *key, double *out, size_t count)
{
    const char *v = json_value(text, key);
    if (v == NULL) {
        return -1;
    }
    const int array = *v == '[';
    if (array) {
        v++;
    } else if (count != 1) {
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        while (*v == ' ' || *v == '\n' || *v == ',' || *v == '\r' || *v == '\t') {
            v++;
        }
        if (strncmp(v, "true", 4) == 0 || strncmp(v, "false", 5) == 0) {
            out[i] = *v == 't' ? 1.0 : 0.0;
            v += *v == 't' ? 4 : 5;
            continue;
        }
        char *end;
        out[i] = strtod(v, &end);
        if (end == v) {
            return -1;
        }
        v = end;
    }
    if (array) {
        while (*v == ' ' || *v == '\n' || *v == '\r' || *v == '\t') {
            v++;
        }
        if (*v != ']') {
            return -1;
        }
    }
    return 0;
}

typedef enum { REAL, U32, BOOL } field_kind;

typedef struct {
    const char *name;
    size_t offset;
    size_t count;
    field_kind kind;
    bool optional; /* MEKF settings, absent from the thesis set (recorded before them) */
} field;

#define REAL_FIELD(f, n) {#f, offsetof(adcs_config, f), n, REAL, false}
#define MEKF_FIELD(f) {#f, offsetof(adcs_config, f), 1, REAL, true}
static const field CONFIG_FIELDS[] = {
    REAL_FIELD(dt, 1),
    REAL_FIELD(process_dt, 1),
    REAL_FIELD(inertia, 9),
    REAL_FIELD(max_dipole, 3),
    REAL_FIELD(flae_weight_sun, 1),
    REAL_FIELD(flae_weight_mag, 1),
    REAL_FIELD(flae_tolerance, 1),
    {"flae_max_iterations", offsetof(adcs_config, flae_max_iterations), 1, U32, false},
    REAL_FIELD(ukf_alpha, 1),
    REAL_FIELD(p_x0, ADCS_NX),
    REAL_FIELD(p_b0, ADCS_NB),
    REAL_FIELD(w_x, ADCS_NX),
    REAL_FIELD(w_b, ADCS_NB),
    REAL_FIELD(v, ADCS_NX),
    MEKF_FIELD(sigma_gyro_noise),
    MEKF_FIELD(sigma_gyro_bias_walk),
    MEKF_FIELD(sigma_mag_noise),
    MEKF_FIELD(sigma_mag_bias_walk),
    MEKF_FIELD(sigma_sun_noise),
    MEKF_FIELD(p0_attitude),
    MEKF_FIELD(p0_gyro_bias),
    MEKF_FIELD(p0_mag_bias),
    MEKF_FIELD(fault_nis_threshold),
    {"fault_steps", offsetof(adcs_config, fault_steps), 1, U32, true},
    MEKF_FIELD(fault_gyro_bias_sigma),
    MEKF_FIELD(fault_mag_bias_sigma),
    MEKF_FIELD(fault_attitude_sigma),
    MEKF_FIELD(sigma_outage_accel),
    {"control_enabled", offsetof(adcs_config, control_enabled), 1, BOOL, false},
    REAL_FIELD(bdot_gain, 1),
    REAL_FIELD(kp, 1),
    REAL_FIELD(kd, 1),
    REAL_FIELD(q_target, 4),
    REAL_FIELD(detumble_enter_rate, 1),
    REAL_FIELD(detumble_exit_rate, 1),
    REAL_FIELD(settle_time, 1),
};

static int read_config(const char *text, adcs_config *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    const char *variant = json_value(text, "variant");
    if (variant == NULL || strncmp(variant, "\"thesis\"", 8) == 0) {
        cfg->variant = ADCS_VARIANT_THESIS;
    } else if (strncmp(variant, "\"mekf\"", 6) == 0) {
        cfg->variant = ADCS_VARIANT_MEKF;
    } else {
        return -1;
    }
    unsigned char *base = (unsigned char *)cfg;
    for (size_t i = 0; i < sizeof CONFIG_FIELDS / sizeof CONFIG_FIELDS[0]; i++) {
        const field *fd = &CONFIG_FIELDS[i];
        double values[16];
        if (fd->optional && json_value(text, fd->name) == NULL) {
            continue;
        }
        if (json_numbers(text, fd->name, values, fd->count) != 0) {
            fprintf(stderr, "config field %s missing or malformed\n", fd->name);
            return -1;
        }
        for (size_t k = 0; k < fd->count; k++) {
            if (fd->kind == REAL) {
                ((adcs_real *)(void *)(base + fd->offset))[k] = (adcs_real)values[k];
            } else if (fd->kind == U32) {
                *(uint32_t *)(void *)(base + fd->offset) = (uint32_t)values[0];
            } else {
                *(uint8_t *)(void *)(base + fd->offset) = values[0] != 0.0 ? 1u : 0u;
            }
        }
    }
    return 0;
}

void golden_output_to_record(const adcs_output *o, double rec[GOLDEN_OUTPUT_SIZE])
{
    size_t k = 0;
    for (size_t i = 0; i < 4; i++) {
        rec[k++] = (double)o->q_est[i];
    }
    for (size_t i = 0; i < 3; i++) {
        rec[k++] = (double)o->w_est[i];
    }
    for (size_t i = 0; i < 3; i++) {
        rec[k++] = (double)o->bias_mag[i];
    }
    for (size_t i = 0; i < 3; i++) {
        rec[k++] = (double)o->bias_gyro[i];
    }
    for (size_t i = 0; i < ADCS_NX + ADCS_NB; i++) {
        rec[k++] = (double)o->p_diag[i];
    }
    for (size_t i = 0; i < 3; i++) {
        rec[k++] = (double)o->m_cmd[i];
    }
    rec[k++] = (double)o->mode;
    rec[k] = (double)o->status;
}

void golden_record_to_input(const double *r, adcs_input *in)
{
    memset(in, 0, sizeof *in);
    in->t = (adcs_real)r[0];
    for (size_t i = 0; i < 3; i++) {
        in->mag_b[i] = (adcs_real)r[1 + i];
        in->gyro_b[i] = (adcs_real)r[5 + i];
        in->sun_b[i] = (adcs_real)r[9 + i];
        in->sun_ref_i[i] = (adcs_real)r[13 + i];
        in->mag_ref_i[i] = (adcs_real)r[16 + i];
    }
    in->mag_valid = r[4] != 0.0;
    in->gyro_valid = r[8] != 0.0;
    in->sun_valid = r[12] != 0.0;
}

int golden_load(const char *json_path, golden_vector *g)
{
    memset(g, 0, sizeof *g);
    char *text = read_text(json_path);
    if (text == NULL) {
        fprintf(stderr, "%s: cannot read\n", json_path);
        return -1;
    }
    double steps_d;
    const int bad =
        read_config(text, &g->config) != 0 || json_numbers(text, "steps", &steps_d, 1) != 0;
    free(text);
    if (bad) {
        fprintf(stderr, "%s: cannot parse\n", json_path);
        return -1;
    }
    const size_t steps = (size_t)steps_d;

    char data_path[1024];
    const size_t len = strlen(json_path);
    if (len < 5 || len + 1 > sizeof data_path) {
        return -1;
    }
    memcpy(data_path, json_path, len - 5);
    strcpy(data_path + len - 5, ".f64");
    FILE *f = fopen(data_path, "rb");
    double *data = malloc(steps * GOLDEN_RECORD_SIZE * sizeof(double));
    const size_t got =
        (f != NULL && data != NULL) ? fread(data, sizeof(double), steps * GOLDEN_RECORD_SIZE, f) : 0;
    if (f != NULL) {
        fclose(f);
    }
    if (got != steps * GOLDEN_RECORD_SIZE) {
        fprintf(stderr, "%s: expected %zu records\n", data_path, steps);
        free(data);
        return -1;
    }
    g->steps = steps;
    g->records = data;
    return 0;
}

void golden_free(golden_vector *g)
{
    free(g->records);
    g->records = NULL;
}
