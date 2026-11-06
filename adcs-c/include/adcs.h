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

/* Version of this interface. Incremented on any change to the exported structs or functions. */
#define ADCS_ABI_VERSION 2u

#ifdef ADCS_USE_FLOAT
typedef float adcs_real;
#else
typedef double adcs_real;
#endif

/* Sizes of the estimator: state x = [q(4), sun(3), B(3), w(3)], bias b = [mag(3), gyro(3)]. */
#define ADCS_NX 13
#define ADCS_NB 6

typedef enum {
    ADCS_MODE_OFF = 0,
    ADCS_MODE_DETUMBLE = 1, /* B-dot */
    ADCS_MODE_ESTIMATE = 2, /* estimation only, coils off */
    ADCS_MODE_POINT = 3,    /* quaternion-feedback pointing */
    ADCS_MODE_SAFE = 4      /* invalid data or filter failure: coils off, filter restarts */
} adcs_mode;

/* Status flags (bit mask) of the last step. 0 means nominal. */
#define ADCS_STATUS_BAD_INPUT (1u << 0)       /* NaN/Inf in an input marked valid */
#define ADCS_STATUS_NO_ATTITUDE_FIX (1u << 1) /* fewer than two vectors: FLAE not run */
#define ADCS_STATUS_FLAE_FAILED (1u << 2)     /* FLAE did not converge or was degenerate */
#define ADCS_STATUS_COVARIANCE_RESET (1u << 3) /* covariance lost positive definiteness */
#define ADCS_STATUS_DIPOLE_SATURATED (1u << 4)
#define ADCS_STATUS_NOT_INITIALISED (1u << 5) /* adcs_step called before adcs_init */

typedef struct {
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

    /* Control */
    bool control_enabled; /* false: estimation only (the thesis cases) */
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
    bool mag_valid;
    adcs_real gyro_b[3];  /* gyroscope reading, body frame (rad/s) */
    bool gyro_valid;
    adcs_real sun_b[3];   /* sun direction from the sun sensors, body frame (unit vector) */
    bool sun_valid;
    adcs_real sun_ref_i[3]; /* sun direction from the on-board ephemeris, inertial (unit) */
    adcs_real mag_ref_i[3]; /* geomagnetic field from the on-board model, inertial (T) */
} adcs_input;

typedef struct {
    adcs_real q_est[4];     /* estimated attitude, scalar-last, inertial -> body */
    adcs_real w_est[3];     /* estimated body rate (rad/s) */
    adcs_real bias_mag[3];  /* estimated magnetometer bias (T) */
    adcs_real bias_gyro[3]; /* estimated gyroscope bias (rad/s) */
    adcs_real p_diag[ADCS_NX + ADCS_NB]; /* covariance diagonal: state then bias (filter units) */
    adcs_real m_cmd[3];     /* magnetorquer dipole command (A m^2) */
    uint32_t mode;          /* adcs_mode */
    uint32_t status;        /* ADCS_STATUS_* flags */
} adcs_output;

/* Opaque state; the caller allocates adcs_state_size() bytes (no heap use inside). */
typedef struct adcs_state adcs_state;

/* Returns ADCS_ABI_VERSION as compiled into the library. */
uint32_t adcs_abi_version(void);

/* Size in bytes of adcs_state. */
size_t adcs_state_size(void);

/* Initialises the state from a configuration. Returns 0 on success, nonzero if the
 * configuration is invalid. */
int adcs_init(adcs_state *state, const adcs_config *config);

/* Processes one control period. Returns 0 on success, nonzero on a programming error
 * (null pointers, not initialised). Data problems are reported in out->status instead. */
int adcs_step(adcs_state *state, const adcs_input *in, adcs_output *out);

#ifdef __cplusplus
}
#endif

#endif /* ADCS_H */
