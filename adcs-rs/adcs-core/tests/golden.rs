//! Golden vectors (testdata/golden/<variant>/<name>.json + .f64): the Rust ADCS must
//! reproduce every output recorded from the Python reference, within the tolerances of
//! `src/adcssim/golden.py`. With the `f32` feature, which cannot match a double-precision
//! reference to those tolerances, only the attitude is checked (within 0.1°), like the C
//! runner's float build.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast,
    clippy::useless_conversion, // f64::from is needed for the f32 build
    clippy::float_cmp           // exact equality is the point (equal values always pass)
)]

use std::path::{Path, PathBuf};

use adcs_core::abi::{Config, Input, NB, NX, Output, VARIANT_MEKF, VARIANT_THESIS};
use adcs_core::{Adcs, Real};
use serde_json::Value;

const INPUT_SIZE: usize = 19;
const OUTPUT_SIZE: usize = 37;
const RECORD_SIZE: usize = INPUT_SIZE + OUTPUT_SIZE;
/// (name, offset in the output record, width, absolute and relative tolerance): a value
/// passes if |actual − expected| ≤ abs + rel·|expected|
const FIELDS: [(&str, usize, usize, f64, f64); 6] = [
    ("q_est", 0, 4, 1e-9, 0.0),
    ("w_est", 4, 3, 1e-9, 0.0),
    ("bias_mag", 7, 3, 1e-16, 1e-6),
    ("bias_gyro", 10, 3, 1e-9, 0.0),
    ("p_diag", 13, 19, 1e-18, 1e-6),
    ("m_cmd", 32, 3, 1e-9, 0.0),
];
const FLOAT_MAX_ATTITUDE_DEG: f64 = 0.1;

fn r(x: f64) -> Real {
    x as Real
}

/// Angle (deg) between two scalar-last quaternions, from the relative quaternion.
fn attitude_difference_deg(q: &[f64], p: &[f64]) -> f64 {
    let rv = [-p[0], -p[1], -p[2]];
    let v = [
        q[3] * rv[0] + p[3] * q[0] - (q[1] * rv[2] - q[2] * rv[1]),
        q[3] * rv[1] + p[3] * q[1] - (q[2] * rv[0] - q[0] * rv[2]),
        q[3] * rv[2] + p[3] * q[2] - (q[0] * rv[1] - q[1] * rv[0]),
    ];
    let w = q[3] * p[3] + (q[0] * p[0] + q[1] * p[1] + q[2] * p[2]);
    2.0 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        .sqrt()
        .atan2(w.abs())
        .to_degrees()
}

/// Whether `a` differs from `e` beyond the tolerance (equal values, NaN and infinities
/// included, pass).
fn differs(e: f64, a: f64, atol: f64, rtol: f64) -> bool {
    if e == a || (e.is_nan() && a.is_nan()) {
        return false;
    }
    !e.is_finite() || !a.is_finite() || (e - a).abs() > atol + rtol * e.abs()
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden")
}

fn arr<const N: usize>(cfg: &Value, key: &str) -> [Real; N] {
    let v: Vec<Real> = cfg[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} missing"))
        .iter()
        .map(|x| r(x.as_f64().unwrap()))
        .collect();
    v.try_into().unwrap()
}

/// Missing MEKF keys read as 0: the thesis set was recorded before they existed.
fn num(cfg: &Value, key: &str) -> f64 {
    cfg.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn read_config(meta: &Value) -> Config {
    let c = &meta["config"];
    Config {
        variant: match c.get("variant").and_then(Value::as_str) {
            None | Some("thesis") => VARIANT_THESIS,
            Some("mekf") => VARIANT_MEKF,
            Some(other) => panic!("unknown variant {other}"),
        },
        dt: r(meta["dt"].as_f64().unwrap()),
        process_dt: r(num(c, "process_dt")),
        inertia: arr(meta, "inertia"),
        max_dipole: arr(meta, "max_dipole"),
        flae_weight_sun: r(num(c, "flae_weight_sun")),
        flae_weight_mag: r(num(c, "flae_weight_mag")),
        flae_tolerance: r(num(c, "flae_tolerance")),
        flae_max_iterations: num(c, "flae_max_iterations") as u32,
        ukf_alpha: r(num(c, "ukf_alpha")),
        p_x0: arr::<NX>(c, "p_x0"),
        p_b0: arr::<NB>(c, "p_b0"),
        w_x: arr::<NX>(c, "w_x"),
        w_b: arr::<NB>(c, "w_b"),
        v: arr::<NX>(c, "v"),
        sigma_gyro_noise: r(num(c, "sigma_gyro_noise")),
        sigma_gyro_bias_walk: r(num(c, "sigma_gyro_bias_walk")),
        sigma_mag_noise: r(num(c, "sigma_mag_noise")),
        sigma_mag_bias_walk: r(num(c, "sigma_mag_bias_walk")),
        sigma_sun_noise: r(num(c, "sigma_sun_noise")),
        p0_attitude: r(num(c, "p0_attitude")),
        p0_gyro_bias: r(num(c, "p0_gyro_bias")),
        p0_mag_bias: r(num(c, "p0_mag_bias")),
        fault_nis_threshold: r(num(c, "fault_nis_threshold")),
        fault_steps: num(c, "fault_steps") as u32,
        fault_gyro_bias_sigma: r(num(c, "fault_gyro_bias_sigma")),
        fault_mag_bias_sigma: r(num(c, "fault_mag_bias_sigma")),
        fault_attitude_sigma: r(num(c, "fault_attitude_sigma")),
        sigma_outage_accel: r(num(c, "sigma_outage_accel")),
        control_enabled: u8::from(c["control_enabled"].as_bool().unwrap()),
        bdot_gain: r(num(c, "bdot_gain")),
        kp: r(num(c, "kp")),
        kd: r(num(c, "kd")),
        q_target: arr(c, "q_target"),
        detumble_enter_rate: r(num(c, "detumble_enter_rate")),
        detumble_exit_rate: r(num(c, "detumble_exit_rate")),
        settle_time: r(num(c, "settle_time")),
    }
}

fn record_to_input(rec: &[f64]) -> Input {
    let v3 = |i: usize| [r(rec[i]), r(rec[i + 1]), r(rec[i + 2])];
    Input {
        t: r(rec[0]),
        mag_b: v3(1),
        mag_valid: u8::from(rec[4] != 0.0),
        gyro_b: v3(5),
        gyro_valid: u8::from(rec[8] != 0.0),
        sun_b: v3(9),
        sun_valid: u8::from(rec[12] != 0.0),
        sun_ref_i: v3(13),
        mag_ref_i: v3(16),
    }
}

fn output_to_record(o: &Output) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(OUTPUT_SIZE);
    for part in [
        &o.q_est[..],
        &o.w_est,
        &o.bias_mag,
        &o.bias_gyro,
        &o.p_diag,
        &o.m_cmd,
    ] {
        v.extend(part.iter().map(|&x| f64::from(x)));
    }
    v.push(f64::from(o.mode));
    v.push(f64::from(o.status));
    v
}

/// Replays one vector; returns the largest difference per field.
fn replay(json: &Path) -> Vec<f64> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(json).unwrap()).unwrap();
    let steps = meta["steps"].as_u64().unwrap() as usize;
    let bytes = std::fs::read(json.with_extension("f64")).unwrap();
    let data: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    assert_eq!(data.len(), steps * RECORD_SIZE);

    let mut adcs = Box::new(Adcs::new());
    adcs.init(&read_config(&meta))
        .expect("configuration rejected");
    let mut worst = vec![0.0; FIELDS.len()];
    for (k, rec) in data.as_chunks::<RECORD_SIZE>().0.iter().enumerate() {
        let expected = &rec[INPUT_SIZE..];
        let actual = output_to_record(&adcs.step(&record_to_input(rec)).unwrap());
        let at = |what: &str| format!("{}: step {k} (t = {} s): {what}", json.display(), rec[0]);
        if cfg!(feature = "f32") {
            assert_eq!(
                expected[0].is_nan(),
                actual[0].is_nan(),
                "{}",
                at("estimate present")
            );
            if !expected[0].is_nan() {
                let angle = attitude_difference_deg(&actual[0..4], &expected[0..4]);
                assert!(
                    angle <= FLOAT_MAX_ATTITUDE_DEG,
                    "{}",
                    at(&format!("{angle} deg"))
                );
                worst[0] = f64::max(worst[0], angle);
            }
            continue;
        }
        for (f, &(name, offset, width, atol, rtol)) in FIELDS.iter().enumerate() {
            for i in offset..offset + width {
                let (e, a) = (expected[i], actual[i]);
                assert!(
                    !differs(e, a, atol, rtol),
                    "{}",
                    at(&format!("{name}[{}] = {a:e}, expected {e:e}", i - offset))
                );
                if e.is_finite() {
                    worst[f] = f64::max(worst[f], (e - a).abs());
                }
            }
        }
        assert_eq!(
            (actual[35], actual[36]),
            (expected[35], expected[36]),
            "{}",
            at("mode/status")
        );
    }
    worst
}

#[test]
#[cfg_attr(
    miri,
    ignore = "too slow under miri; the unit tests cover memory safety"
)]
fn reproduces_all_golden_vectors() {
    let mut files: Vec<PathBuf> = ["thesis", "mekf"]
        .iter()
        .flat_map(|v| std::fs::read_dir(golden_dir().join(v)).unwrap())
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 24);
    for f in &files {
        let worst = replay(f);
        let cells: Vec<String> = FIELDS
            .iter()
            .zip(&worst)
            .map(|((n, ..), w)| format!("{n} {w:.1e}"))
            .collect();
        println!(
            "{:<28} {}",
            f.strip_prefix(golden_dir()).unwrap().display(),
            cells.join("  ")
        );
    }
}
