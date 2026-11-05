# cubesat-adcs-sim

> **Status: work in progress.** The simulator runs (open loop, no ADCS yet). Next: the
> visualization, then the ADCS in C and Rust.

An open-source attitude determination and control system (ADCS) simulator for a 3U CubeSat,
rebuilt from scratch from my MSc thesis
[*Implementation and verification of attitude determination control algorithms for a 3U
CubeSat*](https://aaltodoc.aalto.fi/items/f665d4b5-35c9-4d32-8ee1-40ff1c5f1547)
(Aalto University, 2023).

## What it will contain

- **Simulator (Python):** SGP4 orbit propagation, IGRF magnetic field, Sun position and
  eclipses, rigid-body attitude dynamics, and models of the magnetometer, gyroscope, sun
  sensors and magnetorquers.
- **Visualization:** the satellite over the Earth with its predicted orbit, a ground track,
  and live sensor readings next to what the ADCS corrects in them.
- **ADCS in C:** a Fast Linear Quaternion Attitude Estimator (FLAE) with a two-stage unscented
  Kalman filter, B-dot detumbling and quaternion-feedback control, written like flight
  software: C11, no heap allocation, portable to ARM Cortex-M.
- **ADCS in Rust:** the same algorithms behind the same C interface, and a measured
  comparison of the two implementations.

## Running a scenario

The seven test cases of the thesis are in `scenarios/`:

```bash
uv run adcssim run scenarios/case_1a.toml     # writes runs/case_1a_none_<time>.parquet
```

Each run logs truth, sensor readings and ADCS outputs at every control period to a Parquet
file. What the simulator models, how each model was validated, and its assumptions are in
[docs/simulator.md](docs/simulator.md).

## Repository layout

| Path | Content |
|---|---|
| `src/adcssim/` | Python simulator package |
| `scenarios/` | Scenario files (TOML), including the thesis test cases |
| `data/tle/` | Orbit element sets used by the scenarios |
| `adcs-c/` | ADCS library in C (CMake) |
| `adcs-rs/` | ADCS library in Rust (Cargo workspace) |
| `docs/` | Conventions, maths notes and results |
| `tools/` | Development helper scripts |

## Development

Requirements: Python 3.12+ with [uv](https://docs.astral.sh/uv/), CMake 3.20+ and a C11
compiler, and a stable Rust toolchain.

```bash
uv sync                      # create the virtual environment
uv run pre-commit install    # enable the commit hooks
uv run pytest                # Python tests

cmake -S adcs-c -B build/c && cmake --build build/c && ctest --test-dir build/c

cargo test --manifest-path adcs-rs/Cargo.toml
```

## License

[MIT](LICENSE). The thesis itself is published separately under CC BY-NC-SA 4.0; this
repository re-implements its methods from the published equations.
