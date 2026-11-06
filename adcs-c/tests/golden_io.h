/*
 * golden_io.h: golden-vector reader shared by the golden runner and the benchmark.
 */
#ifndef ADCS_GOLDEN_IO_H
#define ADCS_GOLDEN_IO_H

#include <stddef.h>

#include "adcs.h"

#define GOLDEN_INPUT_SIZE 19  /* doubles per record: the adcs_input fields */
#define GOLDEN_OUTPUT_SIZE 37 /* then the adcs_output fields */
#define GOLDEN_RECORD_SIZE (GOLDEN_INPUT_SIZE + GOLDEN_OUTPUT_SIZE)

typedef struct {
    adcs_config config;
    size_t steps;
    double *records; /* steps × GOLDEN_RECORD_SIZE, malloc'ed */
} golden_vector;

/* Reads <name>.json and <name>.f64. Returns 0, or -1 with a message on stderr. */
int golden_load(const char *json_path, golden_vector *g);
void golden_free(golden_vector *g);

void golden_record_to_input(const double *record, adcs_input *in);
void golden_output_to_record(const adcs_output *o, double record[GOLDEN_OUTPUT_SIZE]);

#endif /* ADCS_GOLDEN_IO_H */
