# The ADCS: algorithms, implementation choices and findings

This page describes the reference ADCS in `src/adcssim/adcs_ref/` (backend `python-ref`). The
C and Rust implementations follow it exactly and are tested against its golden vectors
(`testdata/golden/`). Results for the thesis test cases are in
[results/thesis_cases.md](results/thesis_cases.md).

## Structure

One call of `adcs_step` (one control period, 0.1 s in the thesis cases):

1. **Input checks.** A reading marked valid that contains NaN or Inf is treated as missing,
   flags `BAD_INPUT` and puts the ADCS in `SAFE` mode for that step (coils off).
2. **FLAE** (Wu et al. 2018; thesis §4.2.2) solves Wahba's problem from the sun and
   magnetometer readings and gives an attitude measurement `q`, sign-aligned with the current
   estimate.
3. **Two-stage UKF** (Chen et al.; thesis §2.4.3, §5.4.2) with state
   `x = [q, sun, B, ω]` (13), bias `b = [b_mag, b_gyro]` (6), measurement
   `y = [q_FLAE, sun, B, ω]` and `h(x) = x`. The biases enter only the B and ω measurements
   (`F_s`); `F_a = 0`. The process model integrates the rigid-body kinematics and dynamics with
   RK4 steps of `process_dt`.
4. **Mode manager and control.** `DETUMBLE` (B-dot, eq. 114) above `detumble_enter_rate`,
   `ESTIMATE` (coils off) until `settle_time` has passed, then `POINT` (quaternion-feedback PD,
   eq. 119, converted to a dipole with eq. 121). The dipole is scaled down as a whole vector
   if any coil would saturate. With `control_enabled = false` (the thesis cases) the mode
   stays `ESTIMATE`.

The tuning is the thesis's (§5.4.2), with the magnetic field in microtesla inside the filter,
the unit that tuning was made in.

## Choices where the thesis is silent

| Topic | Choice | Reason |
|---|---|---|
| UKF α | 1.0 | Not stated. The appendix code uses `Xi = √3·α` with κ = 3 − n, which is Julier's original unscented transform when α = 1. |
| Filter rate | 10 Hz | The appendix process model uses a fixed 0.1 s RK4 step. |
| FLAE elimination | Gaussian elimination with full pivoting | The thesis (and Wu's paper) fix q₄ = −1, which divides by zero for attitudes whose q₄ is near 0. Full pivoting fixes whichever component is best conditioned. Tested on 180° rotations. |
| Unavailable measurement | Variance set to 1e12, value replaced by the prediction | Keeps the filter fixed-size (C-friendly) and makes the update ignore that measurement. |
| Covariance hygiene | Symmetrise after updates; Cholesky with growing jitter, reset to P₀ as a last resort (flag `COVARIANCE_RESET`) | The thesis adds ε·I before each Cholesky; this keeps that and adds a defined failure path. |

## Deliberate deviations from the thesis

### 1. Initialisation from the first measurement

The thesis starts the filter at the identity attitude (`x0 = [0 0 0 1 …]`). That is trivial
when the true attitude is also the identity, so the scenarios here start at a 130° rotation
and the filter is initialised from the first FLAE solution and the first readings. The thesis
appendix contains this as a commented-out alternative (`x0 = [0 0 0 1, meas']`).

### 2. Eclipse: FLAE uses the filtered sun vector

FLAE needs two vectors. In eclipse the sun sensor has no reading, so FLAE uses the filter's
own sun vector, rotated forward by the estimated body rate over one step, together with the
real magnetometer reading. This is how the thesis abstract describes the link ("using the
filtered data from the Kalman filter"), and §5.6.5 says the zeroed sun input happened when
*initialising* in eclipse, when no filtered vector exists yet.

When the run starts in eclipse (cases 3a/3b) the ADCS reports no estimate until the first
sun reading, instead of initialising from a zeroed sun vector.

### 3. Process model: body vectors rotate with the state's own vectors

Eq. 126 propagates the body sun and field vectors as `−ω × A(q_FLAE) v_i`, from the FLAE
attitude. Implemented that way, **the estimate diverges in every eclipse** (case 1a: up to
180° error, gyro-bias estimate off by 5 °/s). The mechanism: in eclipse a small attitude
error makes the predicted field wrong, the filter attributes the magnetometer innovation to
ω and the gyro bias (the process model couples them), the corrupted rate corrupts the
attitude, and the loop closes.

Here the vectors rotate as `−ω × v` using the state's own vectors, the physically consistent
form. The magnetometer then constrains the body rate properly in eclipse, and case 1a stays
below 4° through a 1700 s eclipse, with the same pattern the thesis describes (a deviation
that peaks around t ≈ 1500 s and reconverges around t ≈ 1950 s).

| Case 1a, eclipse (t = 1087–2787 s) | Eq. 126 as written | State vectors |
|---|---|---|
| Attitude error at t = 1500 / 2000 / 2500 s | 58° / 64° / 59° | 0.72° / 0.06° / 0.52° |
| Worst error in eclipse | 180° | 3.8° |

## Findings

### Thesis noise levels are standard deviations

The thesis gives σ_AWGN = 1e-12 (magnetometer), 5e-7 (gyroscope) and 1e-4 (sun sensor) for
the nominal cases. Read as variances, they give plausible real-world sensors (1 µT,
0.04 °/s, 0.57°), and that was this project's first reading. Read as standard deviations, as
the symbol σ says, they describe nearly ideal sensors. Two observations decide it:

- With the variance reading, case 1a's eclipse error grows to 20°; with standard deviations
  it shows the thesis's described behaviour (peak near 1500 s, back near 1950 s, < 1° most of
  the time).
- Case 4 ("high noise") read as variances gives a magnetometer noise of 32 µT, about as large
  as the Earth's field itself, which no estimator could turn into the sub-degree accuracy the
  thesis reports. As standard deviations it is 1 nT, 0.29 °/s and 5.7°: plausible "high
  noise" for which sub-degree accuracy is achievable.

The scenario files therefore use the standard-deviation reading.

### The magnetometer-bias failure reproduces

Case 2b reproduces the thesis's main open problem: the magnetometer bias is never estimated
(the estimate stays at zero) and the attitude error reaches ~53° while the bias is active.
The cause is the one the thesis identifies: FLAE uses the raw magnetometer, so the bias
enters the attitude measurement directly and the filter has no way to separate it. Case 2c
shows the side effect: the corrupted attitude also drags the gyro-bias estimate away.
Fixing this is Phase 8.

### Case 4 is worse than reported

With the standard-deviation reading, case 4 converges after ~5 minutes to a median error of
0.6° (worst 2.3° in sunlight, 5.6° in eclipse). The thesis reports below 1° at all times.
The difference is not investigated further here; the thesis's exact noise realisation and
start date are unknown.

### Control: detumbling works, pointing and realistic-sensor eclipses need work

The thesis evaluated estimation only. `scenarios/detumble.toml` (two orbits, realistic sensor
noise and gyro bias, 45° sun-sensor field of view, 80% coil duty cycle) exercises the
controllers:

| t (s) | Body rate (°/s) | Mode | Estimation error (°) | Pointing error (°) |
|---|---|---|---|---|
| 0 | 9.9 | DETUMBLE | 0.5 | 130 |
| 1200 | 3.0 | DETUMBLE | 0.4 | 100 |
| 3000 | 0.9 | POINT | 0.05 | 138 |
| 8000 (eclipse) | 0.6 | POINT | 87 | 164 |
| 11399 | 0.4 | POINT | 0.2 | 74 |

- **B-dot** brings the rate from ~10 °/s to below 1 °/s within the first orbit.
- **Quaternion-feedback pointing** with magnetorquers only does not converge within two
  orbits with the default gains (`kp = 2e-5`, `kd = 2e-3`). Magnetorquers can only produce
  torque perpendicular to the local field, so three-axis pointing relies on the field
  direction changing along the orbit; this needs gain tuning (and possibly a different
  control law).
- **Eclipses with realistic sensors.** With 0.03 °/s gyro noise and an unknown gyro bias,
  coasting through eclipse on the magnetometer alone degrades: a few degrees in the first
  eclipse, up to 87° in the second. With the near-ideal sensors of the thesis cases the same
  filter stays within 4°.

Both are on the Phase 8 list, together with the magnetometer-bias fix and a comparison
against a multiplicative EKF.
