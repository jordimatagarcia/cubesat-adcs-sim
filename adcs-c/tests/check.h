/*
 * check.h: minimal test harness (no dependencies). A test is a void function using CHECK*;
 * RUN() executes it and main returns check_summary().
 */
#ifndef ADCS_CHECK_H
#define ADCS_CHECK_H

#include <math.h>
#include <stdio.h>

static int check_failures;
static int check_count;
static const char *check_current;

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        check_count++;                                                                       \
        if (!(cond)) {                                                                       \
            check_failures++;                                                                \
            fprintf(stderr, "%s:%d: %s: CHECK(%s) failed\n", __FILE__, __LINE__,             \
                    check_current, #cond);                                                   \
        }                                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                \
    do {                                                                                     \
        const double check_a_ = (double)(a), check_b_ = (double)(b);                         \
        check_count++;                                                                       \
        if (!(fabs(check_a_ - check_b_) <= (double)(tol))) {                                 \
            check_failures++;                                                                \
            fprintf(stderr, "%s:%d: %s: %s = %.17g, expected %s = %.17g (tol %g)\n",         \
                    __FILE__, __LINE__, check_current, #a, check_a_, #b, check_b_,           \
                    (double)(tol));                                                          \
        }                                                                                    \
    } while (0)

#define RUN(test)                                                                            \
    do {                                                                                     \
        check_current = #test;                                                               \
        test();                                                                              \
    } while (0)

static inline int check_summary(void)
{
    printf("%d checks, %d failed\n", check_count, check_failures);
    return check_failures == 0 ? 0 : 1;
}

#endif /* ADCS_CHECK_H */
