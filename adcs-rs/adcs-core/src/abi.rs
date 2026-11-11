//! The structs and constants of `adcs-c/include/adcs.h`, with the same memory layout
//! (`#[repr(C)]`). The `adcs-ffi` crate passes them across the C interface unchanged; a test
//! there checks every offset against the C compiler's.
//!
//! The flag fields are bytes (`uint8_t` in the header, non-zero is true), so any value a
//! foreign caller passes is valid.

use crate::real::Real;

pub const NX: usize = 13;
pub const NB: usize = 6;
pub const MEKF_NX: usize = 9;

pub const VARIANT_THESIS: u32 = 0;
pub const VARIANT_MEKF: u32 = 1;

pub const MODE_OFF: u32 = 0;
pub const MODE_DETUMBLE: u32 = 1;
pub const MODE_ESTIMATE: u32 = 2;
pub const MODE_POINT: u32 = 3;
pub const MODE_SAFE: u32 = 4;

pub const STATUS_BAD_INPUT: u32 = 1 << 0;
pub const STATUS_NO_ATTITUDE_FIX: u32 = 1 << 1;
pub const STATUS_FLAE_FAILED: u32 = 1 << 2;
pub const STATUS_COVARIANCE_RESET: u32 = 1 << 3;
pub const STATUS_DIPOLE_SATURATED: u32 = 1 << 4;
pub const STATUS_NOT_INITIALISED: u32 = 1 << 5;
pub const STATUS_BIAS_FAULT: u32 = 1 << 6;
pub const STATUS_RATE_PROPAGATED: u32 = 1 << 7;

/// `adcs_config`: see the header for the meaning of each field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub variant: u32,
    pub dt: Real,
    pub process_dt: Real,
    pub inertia: [Real; 9],
    pub max_dipole: [Real; 3],
    pub flae_weight_sun: Real,
    pub flae_weight_mag: Real,
    pub flae_tolerance: Real,
    pub flae_max_iterations: u32,
    pub ukf_alpha: Real,
    pub p_x0: [Real; NX],
    pub p_b0: [Real; NB],
    pub w_x: [Real; NX],
    pub w_b: [Real; NB],
    pub v: [Real; NX],
    pub sigma_gyro_noise: Real,
    pub sigma_gyro_bias_walk: Real,
    pub sigma_mag_noise: Real,
    pub sigma_mag_bias_walk: Real,
    pub sigma_sun_noise: Real,
    pub p0_attitude: Real,
    pub p0_gyro_bias: Real,
    pub p0_mag_bias: Real,
    pub fault_nis_threshold: Real,
    pub fault_steps: u32,
    pub fault_gyro_bias_sigma: Real,
    pub fault_mag_bias_sigma: Real,
    pub fault_attitude_sigma: Real,
    pub sigma_outage_accel: Real,
    pub control_enabled: u8,
    pub bdot_gain: Real,
    pub kp: Real,
    pub kd: Real,
    pub q_target: [Real; 4],
    pub detumble_enter_rate: Real,
    pub detumble_exit_rate: Real,
    pub settle_time: Real,
}

impl Config {
    /// All zero, which is not a valid configuration: fill in the fields before use.
    pub const ZERO: Self = Self {
        variant: 0,
        dt: 0.0,
        process_dt: 0.0,
        inertia: [0.0; 9],
        max_dipole: [0.0; 3],
        flae_weight_sun: 0.0,
        flae_weight_mag: 0.0,
        flae_tolerance: 0.0,
        flae_max_iterations: 0,
        ukf_alpha: 0.0,
        p_x0: [0.0; NX],
        p_b0: [0.0; NB],
        w_x: [0.0; NX],
        w_b: [0.0; NB],
        v: [0.0; NX],
        sigma_gyro_noise: 0.0,
        sigma_gyro_bias_walk: 0.0,
        sigma_mag_noise: 0.0,
        sigma_mag_bias_walk: 0.0,
        sigma_sun_noise: 0.0,
        p0_attitude: 0.0,
        p0_gyro_bias: 0.0,
        p0_mag_bias: 0.0,
        fault_nis_threshold: 0.0,
        fault_steps: 0,
        fault_gyro_bias_sigma: 0.0,
        fault_mag_bias_sigma: 0.0,
        fault_attitude_sigma: 0.0,
        sigma_outage_accel: 0.0,
        control_enabled: 0,
        bdot_gain: 0.0,
        kp: 0.0,
        kd: 0.0,
        q_target: [0.0; 4],
        detumble_enter_rate: 0.0,
        detumble_exit_rate: 0.0,
        settle_time: 0.0,
    };
}

/// `adcs_input`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Input {
    pub t: Real,
    pub mag_b: [Real; 3],
    pub mag_valid: u8,
    pub gyro_b: [Real; 3],
    pub gyro_valid: u8,
    pub sun_b: [Real; 3],
    pub sun_valid: u8,
    pub sun_ref_i: [Real; 3],
    pub mag_ref_i: [Real; 3],
}

/// `adcs_output`
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    pub q_est: [Real; 4],
    pub w_est: [Real; 3],
    pub bias_mag: [Real; 3],
    pub bias_gyro: [Real; 3],
    pub p_diag: [Real; NX + NB],
    pub m_cmd: [Real; 3],
    pub mode: u32,
    pub status: u32,
}

impl Output {
    /// No estimate yet: NaN estimate, coils off.
    #[must_use]
    pub fn empty(mode: u32, status: u32) -> Self {
        let nan = Real::NAN;
        Self {
            q_est: [nan; 4],
            w_est: [nan; 3],
            bias_mag: [nan; 3],
            bias_gyro: [nan; 3],
            p_diag: [nan; NX + NB],
            m_cmd: [0.0; 3],
            mode,
            status,
        }
    }
}
