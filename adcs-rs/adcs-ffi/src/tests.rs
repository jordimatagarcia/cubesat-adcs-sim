use super::*;
use adcs_core::abi::{
    MODE_OFF, STATUS_BAD_INPUT, STATUS_NO_ATTITUDE_FIX, STATUS_NOT_INITIALISED, VARIANT_MEKF,
    VARIANT_THESIS,
};
use core::fmt::Write;
use core::mem::offset_of;
use std::process::Command;

#[test]
fn exports_core_abi_version() {
    assert_eq!(adcs_abi_version(), adcs_core::ABI_VERSION);
}

/// The `#[repr(C)]` structs must match the C compiler's layout of adcs.h, field by field.
#[test]
#[cfg_attr(miri, ignore = "runs the C compiler")]
#[allow(clippy::too_many_lines)]
fn struct_layout_matches_c_header() {
    macro_rules! fields {
        ($ty:ty, $cname:literal, [$($f:ident),* $(,)?]) => {{
            let mut v = vec![(format!("{} sizeof", $cname), size_of::<$ty>())];
            $( v.push((format!("{} {}", $cname, stringify!($f)), offset_of!($ty, $f))); )*
            v
        }};
    }
    let mut expected = fields!(
        Config,
        "adcs_config",
        [
            variant,
            dt,
            process_dt,
            inertia,
            max_dipole,
            flae_weight_sun,
            flae_weight_mag,
            flae_tolerance,
            flae_max_iterations,
            ukf_alpha,
            p_x0,
            p_b0,
            w_x,
            w_b,
            v,
            sigma_gyro_noise,
            sigma_gyro_bias_walk,
            sigma_mag_noise,
            sigma_mag_bias_walk,
            sigma_sun_noise,
            p0_attitude,
            p0_gyro_bias,
            p0_mag_bias,
            fault_nis_threshold,
            fault_steps,
            fault_gyro_bias_sigma,
            fault_mag_bias_sigma,
            fault_attitude_sigma,
            control_enabled,
            bdot_gain,
            kp,
            kd,
            q_target,
            detumble_enter_rate,
            detumble_exit_rate,
            settle_time,
        ]
    );
    expected.extend(fields!(
        Input,
        "adcs_input",
        [
            t, mag_b, mag_valid, gyro_b, gyro_valid, sun_b, sun_valid, sun_ref_i, mag_ref_i,
        ]
    ));
    expected.extend(fields!(
        Output,
        "adcs_output",
        [
            q_est, w_est, bias_mag, bias_gyro, p_diag, m_cmd, mode, status,
        ]
    ));

    let dir = std::env::temp_dir().join(format!("adcs-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut src = String::from(
        "#include <stdio.h>\n#include <stddef.h>\n#include \"adcs.h\"\nint main(void) {\n",
    );
    for (key, _) in &expected {
        let (ty, field) = key.split_once(' ').unwrap();
        if field == "sizeof" {
            writeln!(src, "  printf(\"%zu\\n\", sizeof({ty}));").unwrap();
        } else {
            writeln!(src, "  printf(\"%zu\\n\", offsetof({ty}, {field}));").unwrap();
        }
    }
    src += "  return 0;\n}\n";
    std::fs::write(dir.join("layout.c"), src).unwrap();
    let include = concat!(env!("CARGO_MANIFEST_DIR"), "/../../adcs-c/include");
    let mut cc = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()));
    cc.args(["-std=c11", "-I", include]);
    if cfg!(feature = "f32") {
        cc.arg("-DADCS_USE_FLOAT");
    }
    let status = cc
        .arg(dir.join("layout.c"))
        .arg("-o")
        .arg(dir.join("layout"))
        .status()
        .expect("C compiler not found");
    assert!(status.success());
    let out = Command::new(dir.join("layout")).output().unwrap();
    let c_values: Vec<usize> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();
    for ((key, rust), c) in expected.iter().zip(&c_values) {
        assert_eq!(rust, c, "{key}: Rust {rust}, C {c}");
    }
    assert_eq!(c_values.len(), expected.len());
}

/// State memory as a C caller would allocate it: 8-byte aligned, zeroed.
fn state_buffer() -> Vec<u64> {
    vec![0u64; adcs_state_size().div_ceil(8)]
}

fn config(variant: u32) -> Config {
    let mut c = Config::ZERO;
    c.variant = variant;
    c.dt = 0.1;
    c.process_dt = 0.1;
    c.inertia[0] = 0.04;
    c.inertia[4] = 0.04;
    c.inertia[8] = 0.007;
    c.max_dipole = [0.1; 3];
    c.flae_weight_sun = 1.0;
    c.flae_weight_mag = 1.0;
    c.flae_tolerance = 1e-3;
    c.flae_max_iterations = 20;
    c.ukf_alpha = 1.0;
    c.p_x0 = [1e-4; 13];
    c.v = [1e-4; 13];
    c.p_b0 = [1e-6; 6];
    c.sigma_mag_noise = 0.1;
    c.sigma_sun_noise = 1e-3;
    c.p0_attitude = 0.1;
    c.p0_gyro_bias = 0.1;
    c.p0_mag_bias = 0.1;
    c.fault_nis_threshold = 25.0;
    c.fault_steps = 3;
    c.q_target[3] = 1.0;
    c
}

#[test]
fn misuse_is_reported_not_undefined() {
    let mut mem = state_buffer();
    let state = mem.as_mut_ptr().cast::<c_void>();
    let cfg = config(VARIANT_MEKF);
    let mut input = Input::default();
    let mut out = Output::empty(0, 0);
    unsafe {
        assert_eq!(adcs_init(core::ptr::null_mut(), &raw const cfg), -1);
        assert_eq!(adcs_init(state, core::ptr::null()), -1);
        assert_eq!(
            adcs_init(state.cast::<u8>().add(1).cast(), &raw const cfg),
            -2
        );
        // zeroed memory, never initialised
        assert_eq!(adcs_step(state, &raw const input, &raw mut out), -1);
        assert_eq!(out.status, STATUS_NOT_INITIALISED);
        assert_eq!(out.mode, MODE_OFF);
        assert!(out.q_est[0].is_nan());
        assert_eq!(
            adcs_step(state, &raw const input, core::ptr::null_mut()),
            -1
        );

        let mut bad = cfg;
        bad.variant = 7;
        assert_eq!(adcs_init(state, &raw const bad), -3);
        bad = config(VARIANT_THESIS);
        bad.v[3] = 0.0;
        assert_eq!(adcs_init(state, &raw const bad), -3);
        assert_eq!(adcs_step(state, &raw const input, &raw mut out), -1); // a failed init leaves it unusable

        for variant in [VARIANT_THESIS, VARIANT_MEKF] {
            let variant_cfg = config(variant);
            assert_eq!(adcs_init(state, &raw const variant_cfg), 0);
            assert_eq!(adcs_step(state, &raw const input, &raw mut out), 0);
            assert_eq!(out.mode, MODE_OFF);
            assert_ne!(out.status & STATUS_NO_ATTITUDE_FIX, 0);
            input.mag_b = [adcs_core::Real::NAN; 3];
            input.gyro_b = [adcs_core::Real::NAN; 3];
            input.sun_b = [adcs_core::Real::NAN; 3];
            input.mag_valid = 1;
            input.gyro_valid = 2; // any non-zero byte is true, without undefined behaviour
            input.sun_valid = 1;
            assert_eq!(adcs_step(state, &raw const input, &raw mut out), 0);
            assert_ne!(out.status & STATUS_BAD_INPUT, 0);
            input = Input::default();
        }
    }
}
