//! Fuzzes `Adcs::step` (cargo fuzz run step), like `adcs-c/tests/fuzz_step.c`.
//!
//! Starts from a realistic configuration and input (golden vector "nominal", step 0). Byte 0
//! selects the variant (bit 0) and control (bit 1); then records of 9 bytes, one per step:
//! an op byte whose value mod 20 picks an input field to overwrite with the next 8 bytes as
//! a raw f64 (19 = no change), after which the ADCS steps once. Checks that every step
//! succeeds, the mode and status are in range, the attitude is unit-norm or all NaN, and the
//! dipole command is finite and within the coil limits.
#![no_main]

use adcs_core::Adcs;
use adcs_core::abi::{Config, Input, MODE_SAFE, VARIANT_MEKF, VARIANT_THESIS};
use libfuzzer_sys::fuzz_target;

const BASE_INPUT: [f64; 19] = [
    0.0,
    -1.450_141_663_940_126e-5,
    2.708_610_337_796_873_7e-5,
    1.659_670_587_303_675e-5,
    1.0,
    0.052_361_120_400_379_29,
    0.087_267_015_572_057_24,
    -0.069_813_797_952_736_16,
    1.0,
    0.341_067_451_638_774_4,
    0.530_452_642_799_764,
    0.776_075_374_676_57,
    1.0,
    -0.930_701_755_323_723_8,
    -0.335_620_057_187_648_8,
    -0.145_442_152_936_136_77,
    -2.519_617_593_206_360_3e-5,
    1.240_078_485_858_844_8e-5,
    -2.075_504_644_853_297_5e-5,
];

fn config(variant: u32, control: bool) -> Config {
    let mut c = Config::ZERO;
    c.variant = variant;
    c.dt = 0.1;
    c.process_dt = 0.1;
    c.inertia[0] = 0.041_866_6;
    c.inertia[4] = 0.041_866_6;
    c.inertia[8] = 0.006_666_6;
    c.max_dipole = [0.13, 0.13, 0.082];
    c.flae_weight_sun = 1.0;
    c.flae_weight_mag = 1.0;
    c.flae_tolerance = 1e-3;
    c.flae_max_iterations = 20;
    c.ukf_alpha = 1.0;
    c.p_x0 = [
        1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1e-7, 1e-7, 1e-7, 5e-7, 5e-7, 5e-7,
    ];
    c.w_x = [
        1e-8, 1e-8, 1e-8, 1e-8, 1e-4, 1e-4, 1e-4, 2e-2, 2e-2, 2e-2, 1e-9, 1e-9, 1e-9,
    ];
    c.v = [
        1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1.0, 1.0, 1.0, 5e-6, 5e-6, 5e-6,
    ];
    c.p_b0 = [1e-12, 1e-12, 1e-12, 1e-7, 1e-7, 1e-7];
    c.w_b = [1e-11, 1e-11, 1e-11, 1e-6, 1e-6, 1e-6];
    c.sigma_gyro_noise = 1e-5;
    c.sigma_gyro_bias_walk = 3e-6;
    c.sigma_mag_noise = 0.1;
    c.sigma_mag_bias_walk = 1e-3;
    c.sigma_sun_noise = 2e-4;
    c.p0_attitude = 0.05;
    c.p0_gyro_bias = 5e-3;
    c.p0_mag_bias = 2.0;
    c.fault_nis_threshold = 25.0;
    c.fault_steps = 3;
    c.fault_gyro_bias_sigma = 0.1;
    c.fault_mag_bias_sigma = 50.0;
    c.fault_attitude_sigma = 0.05;
    c.control_enabled = u8::from(control);
    c.bdot_gain = 4e4;
    c.kp = 2e-5;
    c.kd = 2e-3;
    c.q_target[3] = 1.0;
    c.detumble_enter_rate = 0.0698;
    c.detumble_exit_rate = 0.0175;
    c.settle_time = 1.0;
    c
}

fn to_input(r: &[f64; 19]) -> Input {
    let v3 = |i: usize| [r[i], r[i + 1], r[i + 2]];
    Input {
        t: r[0],
        mag_b: v3(1),
        mag_valid: u8::from(r[4] != 0.0),
        gyro_b: v3(5),
        gyro_valid: u8::from(r[8] != 0.0),
        sun_b: v3(9),
        sun_valid: u8::from(r[12] != 0.0),
        sun_ref_i: v3(13),
        mag_ref_i: v3(16),
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&flags, rest)) = data.split_first() else {
        return;
    };
    let variant = if flags & 1 == 0 {
        VARIANT_THESIS
    } else {
        VARIANT_MEKF
    };
    let cfg = config(variant, flags & 2 != 0);
    let mut adcs = Box::new(Adcs::new());
    adcs.init(&cfg).expect("base configuration is valid");
    let mut fields = BASE_INPUT;
    for rec in rest.chunks_exact(9) {
        let op = usize::from(rec[0] % 20);
        if op < 19 {
            fields[op] = f64::from_le_bytes(rec[1..9].try_into().unwrap());
        }
        let out = adcs.step(&to_input(&fields)).expect("initialised");
        assert!(out.mode <= MODE_SAFE && out.status & !0x7f == 0);
        let n2: f64 = out.q_est.iter().map(|x| x * x).sum();
        assert!(out.q_est.iter().all(|x| x.is_nan()) || (n2 - 1.0).abs() < 1e-6);
        for (m, max) in out.m_cmd.iter().zip(cfg.max_dipole) {
            assert!(
                m.abs() <= max * (1.0 + 1e-12),
                "dipole {m} beyond limit {max}"
            );
        }
        fields[0] += 0.1;
    }
});
