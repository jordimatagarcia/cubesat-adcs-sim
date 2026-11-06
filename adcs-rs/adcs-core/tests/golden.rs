//! Golden vectors (testdata/golden/<variant>/<name>.json + .f64): the Rust ADCS must
//! reproduce every output recorded from the Python reference, within the tolerances of
//! `src/adcssim/golden.py`.
#![cfg(not(feature = "f32"))] // the tolerances are for double precision
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::path::{Path, PathBuf};

use adcs_core::Adcs;
use adcs_core::abi::{Config, Input, NB, NX, Output, VARIANT_MEKF, VARIANT_THESIS};
use serde_json::Value;

const INPUT_SIZE: usize = 19;
const OUTPUT_SIZE: usize = 37;
const RECORD_SIZE: usize = INPUT_SIZE + OUTPUT_SIZE;
/// (name, offset in the output record, width, tolerance)
const FIELDS: [(&str, usize, usize, f64); 6] = [
    ("q_est", 0, 4, 1e-9),
    ("w_est", 4, 3, 1e-9),
    ("bias_mag", 7, 3, 1e-13),
    ("bias_gyro", 10, 3, 1e-9),
    ("p_diag", 13, 19, 1e-9),
    ("m_cmd", 32, 3, 1e-9),
];

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden")
}

fn arr<const N: usize>(cfg: &Value, key: &str) -> [f64; N] {
    let v: Vec<f64> = cfg[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} missing"))
        .iter()
        .map(|x| x.as_f64().unwrap())
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
        dt: meta["dt"].as_f64().unwrap(),
        process_dt: num(c, "process_dt"),
        inertia: arr(meta, "inertia"),
        max_dipole: arr(meta, "max_dipole"),
        flae_weight_sun: num(c, "flae_weight_sun"),
        flae_weight_mag: num(c, "flae_weight_mag"),
        flae_tolerance: num(c, "flae_tolerance"),
        flae_max_iterations: num(c, "flae_max_iterations") as u32,
        ukf_alpha: num(c, "ukf_alpha"),
        p_x0: arr::<NX>(c, "p_x0"),
        p_b0: arr::<NB>(c, "p_b0"),
        w_x: arr::<NX>(c, "w_x"),
        w_b: arr::<NB>(c, "w_b"),
        v: arr::<NX>(c, "v"),
        sigma_gyro_noise: num(c, "sigma_gyro_noise"),
        sigma_gyro_bias_walk: num(c, "sigma_gyro_bias_walk"),
        sigma_mag_noise: num(c, "sigma_mag_noise"),
        sigma_mag_bias_walk: num(c, "sigma_mag_bias_walk"),
        sigma_sun_noise: num(c, "sigma_sun_noise"),
        p0_attitude: num(c, "p0_attitude"),
        p0_gyro_bias: num(c, "p0_gyro_bias"),
        p0_mag_bias: num(c, "p0_mag_bias"),
        fault_nis_threshold: num(c, "fault_nis_threshold"),
        fault_steps: num(c, "fault_steps") as u32,
        fault_gyro_bias_sigma: num(c, "fault_gyro_bias_sigma"),
        fault_mag_bias_sigma: num(c, "fault_mag_bias_sigma"),
        fault_attitude_sigma: num(c, "fault_attitude_sigma"),
        control_enabled: u8::from(c["control_enabled"].as_bool().unwrap()),
        bdot_gain: num(c, "bdot_gain"),
        kp: num(c, "kp"),
        kd: num(c, "kd"),
        q_target: arr(c, "q_target"),
        detumble_enter_rate: num(c, "detumble_enter_rate"),
        detumble_exit_rate: num(c, "detumble_exit_rate"),
        settle_time: num(c, "settle_time"),
    }
}

fn record_to_input(r: &[f64]) -> Input {
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

fn output_to_record(o: &Output) -> Vec<f64> {
    let mut v = Vec::with_capacity(OUTPUT_SIZE);
    v.extend_from_slice(&o.q_est);
    v.extend_from_slice(&o.w_est);
    v.extend_from_slice(&o.bias_mag);
    v.extend_from_slice(&o.bias_gyro);
    v.extend_from_slice(&o.p_diag);
    v.extend_from_slice(&o.m_cmd);
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
        for (f, &(name, offset, width, tol)) in FIELDS.iter().enumerate() {
            for i in offset..offset + width {
                let (e, a) = (expected[i], actual[i]);
                assert_eq!(
                    e.is_nan(),
                    a.is_nan(),
                    "{}: step {k}: {name}[{}] NaN mismatch ({a} vs {e})",
                    json.display(),
                    i - offset
                );
                if !e.is_nan() {
                    let d = (e - a).abs();
                    assert!(
                        d <= tol,
                        "{}: step {k} (t = {} s): {name}[{}] = {a:e}, expected {e:e}",
                        json.display(),
                        rec[0],
                        i - offset
                    );
                    worst[f] = f64::max(worst[f], d);
                }
            }
        }
        assert_eq!(
            (actual[35], actual[36]),
            (expected[35], expected[36]),
            "{}: step {k}: mode/status",
            json.display()
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
    assert_eq!(files.len(), 16);
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
