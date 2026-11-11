/*
 * adcs.h: public interface of the ADCS library.
 *
 * This header is the contract shared by the C and Rust implementations (and mirrored by the
 * Python reference in src/adcssim/adcs_ref). All three export exactly these symbols and
 * structs, so the simulator can load any of them.
 *
 * Conventions (docs/conventions.md): SI units at the interface, quaternions scalar-last and
 * mapping inertial to body coordinates, matrices row-major.
 */
#ifndef ADCS_H
#define ADCS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Symbols exported from the shared library (everything else is hidden). */
#if defined(ADCS_BUILDING_LIBRARY) && (defined(__GNUC__) || defined(__clang__))
#define ADCS_API __attribute__((visibility("default")))
#else
#define ADCS_API
#endif

/* Version of this interface. Incremented on any change to the exported structs or functions.
 * The precision is not part of it: a caller also checks adcs_real_size() == sizeof(adcs_real)
 * (see adcs_check_abi below). */
#define ADCS_ABI_VERSION 4u

#ifdef ADCS_USE_FLOAT
typedef float adcs_real;
#else
typedef double adcs_real;
#endif

/* Sizes of the thesis estimator: state x = [q(4), sun(3), B(3), w(3)], bias b = [mag(3),
 * gyro(3)]. The MEKF uses 9 states (attitude error, gyro bias, magnetometer bias). */
#define ADCS_NX 13
#define ADCS_NB 6
#define ADCS_MEKF_NX 9

/* Estimator selected in adcs_config.variant (docs/adcs.md). */
typedef enum {
    ADCS_VARIANT_THESIS = 0, /* FLAE + two-stage UKF, as reproduced from the thesis */
    ADCS_VARIANT_MEKF = 1    /* multiplicative EKF with vector measurements (flight choice) */
} adcs_variant;

typedef enum {
    ADCS_MODE_OFF = 0,
    ADCS_MODE_DETUMBLE = 1, /* B-dot */
    ADCS_MODE_ESTIMATE = 2, /* estimation only, coils off */
    ADCS_MODE_POINT = 3,    /* quaternion-feedback pointing */
    ADCS_MODE_SAFE = 4      /* invalid data or filter failure: coils off; the interrupted mode
                               resumes, with its timer, once the inputs are good again */
} adcs_mode;

/* Status flags (bit mask) of the last step. 0 means nominal. */
#define ADCS_STATUS_BAD_INPUT (1u << 0) /* NaN/Inf or implausible value in an input marked valid */
#define ADCS_STATUS_NO_ATTITUDE_FIX (1u << 1) /* fewer than two vectors: FLAE not run */
#define ADCS_STATUS_FLAE_FAILED (1u << 2)     /* FLAE did not converge or was degenerate */
#define ADCS_STATUS_COVARIANCE_RESET (1u << 3) /* covariance lost positive definiteness */
#define ADCS_STATUS_DIPOLE_SATURATED (1u << 4)
#define ADCS_STATUS_NOT_INITIALISED (1u << 5) /* adcs_step called before adcs_init */
#define ADCS_STATUS_BIAS_FAULT (1u << 6)      /* MEKF: a sensor-bias fault was declared */
#define ADCS_STATUS_RATE_PROPAGATED (1u << 7) /* MEKF: no gyro; rate propagated by dynamics */

typedef struct {
    uint32_t variant;     /* adcs_variant */
    adcs_real dt;         /* control period (s) */
    adcs_real process_dt; /* RK4 step of the filter's process model (s); dt is split into steps */
    adcs_real inertia[9]; /* inertia tensor as known on board, row-major (kg m^2) */
    adcs_real max_dipole[3]; /* magnetorquer limits (A m^2) */

    /* FLAE (Wu et al. 2018) */
    adcs_real flae_weight_sun;
    adcs_real flae_weight_mag;
    adcs_real flae_tolerance; /* Newton iteration stop criterion on lambda */
    uint32_t flae_max_iterations;

    /* Two-stage UKF. The magnetic field is handled in microtesla inside the filter, the unit
     * the thesis tuning was made in; all other quantities are SI. */
    adcs_real ukf_alpha;
    adcs_real p_x0[ADCS_NX]; /* initial state covariance (diagonal) */
    adcs_real p_b0[ADCS_NB]; /* initial bias covariance (diagonal) */
    adcs_real w_x[ADCS_NX];  /* state process noise per filter step (diagonal) */
    adcs_real w_b[ADCS_NB];  /* bias process noise per filter step (diagonal) */
    adcs_real v[ADCS_NX];    /* measurement noise (diagonal) */

    /* MEKF. Noise values are standard deviations; the magnetic field is in microtesla. */
    adcs_real sigma_gyro_noise;     /* gyro white noise per sample (rad/s) */
    adcs_real sigma_gyro_bias_walk; /* gyro bias random walk (rad/s/sqrt(s)) */
    adcs_real sigma_mag_noise;      /* magnetometer noise (uT) */
    adcs_real sigma_mag_bias_walk;  /* magnetometer bias random walk (uT/sqrt(s)) */
    adcs_real sigma_sun_noise;      /* sun sensor noise per component (rad) */
    adcs_real p0_attitude;          /* initial attitude uncertainty (rad) */
    adcs_real p0_gyro_bias;         /* initial gyro bias uncertainty (rad/s) */
    adcs_real p0_mag_bias;          /* initial magnetometer bias uncertainty (uT) */
    adcs_real fault_nis_threshold;  /* per-step sun or magnetometer NIS that is suspect */
    uint32_t fault_steps;           /* consecutive suspect steps before a fault is declared */
    adcs_real fault_gyro_bias_sigma;  /* uncertainties added on a fault */
    adcs_real fault_mag_bias_sigma;
    adcs_real fault_attitude_sigma;
    adcs_real sigma_outage_accel; /* rad/s^2: rate uncertainty growth while the gyro is out */

    /* Control */
    uint8_t control_enabled; /* 0: estimation only (the thesis cases); flags are bytes, not
                                bool, so any value a foreign caller passes is valid */
    adcs_real bdot_gain;  /* m = -k dB/dt (A m^2 s / T) */
    adcs_real kp;         /* quaternion-feedback gains, tau = -kp sign(dq4) dq - kd w */
    adcs_real kd;
    adcs_real q_target[4];         /* target attitude for POINT mode */
    adcs_real detumble_enter_rate; /* rad/s: enter DETUMBLE above this body rate */
    adcs_real detumble_exit_rate;  /* rad/s: leave DETUMBLE below this body rate */
    adcs_real settle_time;         /* s in ESTIMATE before POINT */
} adcs_config;

typedef struct {
    adcs_real t;          /* s since start */
    adcs_real mag_b[3];   /* magnetometer reading, body frame (T) */
    uint8_t mag_valid;    /* non-zero: the reading is usable */
    adcs_real gyro_b[3];  /* gyroscope reading, body frame (rad/s) */
    uint8_t gyro_valid;
    adcs_real sun_b[3];   /* sun direction from the sun sensors, body frame (unit vector) */
    uint8_t sun_valid;
    adcs_real sun_ref_i[3]; /* sun direction from the on-board ephemeris, inertial (unit) */
    adcs_real mag_ref_i[3]; /* geomagnetic field from the on-board model, inertial (T) */
} adcs_input;

typedef struct {
    adcs_real q_est[4];     /* estimated attitude, scalar-last, inertial -> body */
    adcs_real w_est[3];     /* estimated body rate (rad/s) */
    adcs_real bias_mag[3];  /* estimated magnetometer bias (T) */
    adcs_real bias_gyro[3]; /* estimated gyroscope bias (rad/s) */
    /* Covariance diagonal in filter units. Thesis: [q(4), sun(3), B(3), w(3), b_mag(3),
     * b_gyro(3)]. MEKF: [attitude error (3, rad^2), NaN x7, rate (3), b_mag (3, uT^2),
     * b_gyro (3)], so the same indices mean the same quantities where both have them. */
    adcs_real p_diag[ADCS_NX + ADCS_NB];
    adcs_real m_cmd[3];     /* magnetorquer dipole command (A m^2) */
    uint32_t mode;          /* adcs_mode */
    uint32_t status;        /* ADCS_STATUS_* flags */
} adcs_output;

/* Opaque state; the caller allocates adcs_state_size() bytes, aligned for any type (as
 * malloc returns), and keeps them for the life of the ADCS. No heap use inside. */
typedef struct adcs_state adcs_state;

/* Returns ADCS_ABI_VERSION as compiled into the library. */
ADCS_API uint32_t adcs_abi_version(void);

/* sizeof(adcs_real) in the library: 8 (double) or 4 (float, ADCS_USE_FLOAT). */
ADCS_API size_t adcs_real_size(void);

/* Size in bytes of adcs_state. */
ADCS_API size_t adcs_state_size(void);

/* Initialises the state from a configuration (the rules are in docs/adcs.md, "Configuration
 * rules"). Returns 0 on success; -1 for a null pointer, -2 for misaligned state memory, -3 for
 * an invalid configuration. Whenever `state` is non-null and aligned it is marked
 * uninitialised first, so a later adcs_step on it fails cleanly. */
ADCS_API int adcs_init(adcs_state *state, const adcs_config *config);

/* Processes one control period. Returns 0 on success, nonzero on a programming error
 * (null pointers, misaligned or not initialised state). Data problems are reported in
 * out->status instead. */
ADCS_API int adcs_step(adcs_state *state, const adcs_input *in, adcs_output *out);

/* Whether the linked library matches this header (version and precision). */
static inline bool adcs_check_abi(void)
{
    return adcs_abi_version() == ADCS_ABI_VERSION && adcs_real_size() == sizeof(adcs_real);
}

#ifdef __cplusplus
}
#endif

#endif /* ADCS_H */
