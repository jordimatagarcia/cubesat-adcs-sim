# Simulator: models, validation and assumptions

The simulator is the "truth" side of the loop. It propagates the orbit and attitude, models
the space environment and the sensors, and hands the ADCS exactly what flight software would
see. This page lists every model, how it was checked, and what it deliberately leaves out.

## Models

| Module | Model | Source |
|---|---|---|
| `timebase` | UTC → Julian date, decimal year, GMST (IAU-82) | Vallado, *Fundamentals of Astrodynamics*, alg. 14, eq. 3-47 |
| `frames` | Scalar-last quaternions, attitude matrix, TEME ↔ Earth-fixed, WGS-84 geodetic | Markley & Crassidis (2014) ch. 2–3 |
| `orbit` | SGP4 from a TLE or from mean elements; next-orbit prediction | Vallado et al. (2006), `sgp4` package |
| `igrf` | IGRF-14 main field, degree 13, from the official IAGA coefficients | Davis (2004); IAGA (2024) |
| `environment` | Sun position (almanac, ~0.01°), cylindrical Earth shadow, exponential atmosphere | Vallado alg. 29, table 8-4; thesis eqs. 122–125 |
| `disturbances` | Gravity gradient, residual dipole, drag and solar pressure torques | Thesis §3.4; Wertz (1978) |
| `dynamics` | Euler's equations + quaternion kinematics, RK4 | Thesis §3.2–3.3 |
| `sensors` | Magnetometer (eq. 93), gyroscope (eq. 91), sun sensors per face | Thesis §4.1.2; LIS3MDL, L3GD20H and S5990-1 datasheets |
| `actuators` | Magnetorquers with saturation, quantisation and duty cycle | Thesis §4.1.3, table 3 |

## Validation

Every model is tested against something independent of this code. The tests live in `tests/`
and run in CI.

| Check | Reference | Result |
|---|---|---|
| GMST | astropy 8.0 (IAU-82), Vallado example 3-5, `sgp4.gstime` | < 1e-8° |
| Sun direction | astropy `get_sun` in TEME | < 0.006° (model spec 0.01°) |
| TEME → Earth-fixed | astropy (which also applies polar motion) | < 20 m |
| SGP4 | Vallado SGP4-VER satellite 00005 | < 0.1 m |
| IGRF-14 | ppigrf, 18 points incl. near both poles | 1e-11 nT at model epochs, < 0.05 nT between |
| Eclipse | Analytic eclipse fraction for the orbit's β angle | within 1% of the orbit |
| Dynamics | Energy and inertial angular momentum, torque-free, 1 h | relative drift < 1e-9 |
| Dynamics | Analytic axisymmetric precession; closed-form constant-rate kinematics | < 1e-10 |
| RK4 | Error ratio when halving the step | 16 ± 25% (fourth order) |
| Quaternions | Property tests (hypothesis): composition, inverse, DCM round trip | exact to 1e-12 |
| Sensors | Noise statistics, bias windows, random-walk growth, quantisation | as configured |

## Assumptions and simplifications

- **UT1 = UTC.** The difference is under 0.9 s, so the Earth's rotation angle is off by less
  than 0.004°.
- **TEME is the inertial frame.** SGP4 outputs TEME and the thesis treated it as inertial.
  The simulator and the ADCS use the same frame, so attitude estimation is unaffected.
- **No polar motion** in TEME → Earth-fixed (about 10 m on the ground track).
- **Sun position from a low-precision model**, used directly in TEME (nutation ignored,
  under 0.01°).
- **Cylindrical Earth shadow**, as in the thesis: no penumbra, no atmospheric refraction.
- **Exponential atmosphere** with fixed density per altitude band: no solar-activity or
  day/night variation. Drag torque is tiny for a CubeSat with its centre of pressure near the
  centre of mass, so this affects nothing the ADCS could see.
- **Generic spacecraft:** a uniform 4 kg 3U box. Foresail-1's mass properties aren't
  published. A scenario that sets `mass_kg` without an inertia gets the inertia of a uniform
  3U box of that mass.
- **Environment held constant over one dynamics step** (0.1 s), while attitude-dependent
  torques are re-evaluated in every RK4 stage. It does not depend on the attitude, so it is
  evaluated 64 control periods at a time, with one IGRF epoch per batch (the field's secular
  variation over a batch is far below a nanotesla).
- **Coils switch exactly at the duty-cycle boundary.** A dynamics step that straddles the
  switch-off time is split there into two RK4 steps, so the magnetic impulse is exact for
  any duty cycle, not rounded to whole dynamics steps.
- **Sensors sample at the start of each control period.** No sensor latency or filtering.
- **Faults stop the run instead of corrupting it.** A non-finite coil command is applied as
  zero dipole (a real driver can't produce NaN amps), and a non-finite true attitude or rate
  ends the run with an error rather than writing NaN telemetry.
- **Runs must lie within the IGRF model's validity** (1900 to five years after the last
  epoch); this is checked before the run starts.

## Scenario validation

Scenario files are checked when loaded, and overrides (such as `--duration`) are checked
the same way: unknown keys, non-finite numbers, a zero initial quaternion, negative noise,
drag or reflectivity coefficients, a sun-sensor half field of view outside (0°, 180°], a
non-positive mass, and run lengths that aren't a whole number of control periods (which
were previously rounded silently) are all errors. The name may only contain letters,
digits, `_`, `-` and `.`, because it becomes part of the log file name.

## Run metadata

Every log records what produced it: the scenario file's text and name, the backend, the
ADCS variant and its full configuration (as JSON), the seed, duration and time steps, the
start time, the simulator version, the git commit of the source tree (with `+dirty` if it
had local changes), and for the C and Rust backends the path of the library loaded.

## Reproducing the thesis cases

The scenario files in `scenarios/` reproduce the seven thesis cases (§5.5), plus case 2c,
which the thesis mentions but doesn't run. They run the reference ADCS at 10 Hz, as the
thesis filter did, starting from a 130° attitude.

- **TLE:** the Aalto-1 element set printed in the thesis (§2.1.2), in
  `data/tle/aalto-1_2021-287.tle`. pdftotext had broken its spacing; the fixed-column lines
  were rebuilt and confirmed by both TLE checksums.
- **Noise levels:** the thesis gives σ_AWGN per sensor (e.g. 1e-12 for the magnetometer).
  They are read as standard deviations, as the symbol says; this reading reproduces the
  thesis results, while reading them as variances does not (see [adcs.md](adcs.md#findings)).
  The nominal cases therefore have nearly ideal sensors.
- **Eclipse:** the runs start 3816 s after the TLE epoch, so the eclipse begins at
  t = 1087 s as in the thesis. With this date it lasts about 1700 s; the thesis reports
  1167 s, but doesn't give the start date it used, and the eclipse length depends on the
  season.
- **Sensor coverage and quantisation:** the thesis models white noise only and full sun
  sensor coverage, so these files set `quantize = false` and `half_fov_deg = 90`. The code
  defaults (datasheet resolution, 45° per face) are more realistic.
