# The ADCS: algorithms, implementation choices and findings

The reference ADCS in `src/adcssim/adcs_ref/` (backend `python-ref`) has two variants,
selected with `variant` in a scenario's `[adcs]` table:

- **`thesis`** (default): the thesis algorithm, reproduced and frozen together with its
  golden vectors (`testdata/golden/`) and results
  ([results/thesis_cases.md](results/thesis_cases.md)). Described in the first half of this
  page.
- **`improved`**: an error-state UKF designed from a measured diagnosis of the thesis
  variant ([results/diagnosis.md](results/diagnosis.md)). Described in
  [The improved variant](#the-improved-variant); compared with the thesis variant in
  [results/variants.md](results/variants.md).
- **`mekf`**: the same model and settings with a multiplicative extended Kalman filter, the
  industry-standard baseline ([MEKF baseline](#mekf-baseline)).

Both share the interface (`adcs-c/include/adcs.h`), the mode manager and the control laws.

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
The improved variant below fixes it.

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

The improved variant and the MEKF baseline below address both.


## The improved variant

Designed from the [diagnosis](results/diagnosis.md) of the thesis variant: every change
answers a measured problem.

### Algorithm

An error-state unscented filter, USQUE (Crassidis & Markley 2003; the thesis's reference
[7] proposes it in §5.7), extended with a magnetometer bias:

- **State (9):** attitude error as generalised Rodrigues parameters (3; equal to the
  rotation angle vector for small errors) around a reference quaternion, gyro bias (3),
  magnetometer bias (3, µT). The quaternion itself is never a UKF state, so its unit-norm
  constraint can't break the filter's assumptions (thesis §5.7).
- **Gyro as the rate input.** Each step propagates every sigma point's attitude with its
  own bias-corrected rate, using the mean of the previous and current gyro readings. No
  rigid-body dynamics are modelled, so unmodelled torques (control, duty cycle,
  disturbances) cannot corrupt the estimate.
- **Vector measurements.** `y_sun = A(q) s_i` and `y_B = A(q) B_i + b_mag`, compared with
  the on-board ephemeris and field model. The magnetometer constrains the attitude directly
  (two axes at any instant, all three as the field direction turns along the orbit), and
  the magnetometer bias is observable because the sun sensor pins the attitude.
- **FLAE** only initialises the attitude.
- **Bias faults.** Both biases are slow random walks, so they don't absorb sensor noise.
  Sudden faults (the thesis test cases) are caught by the innovations: when the sun or
  magnetometer NIS stays above a threshold for a few steps in sunlight, the attitude and
  both bias uncertainties are inflated and the measurements re-estimate them. Which sensor
  failed isn't guessed, because a magnetometer-bias step also shows up in the sun
  innovation once it has pulled the attitude.
- **Noise settings** are matched to the scenario's sensors (`adcs_ref/tuning.py`), with
  floors for the on-board reference models (0.1 µT for the field model, 2e-4 rad for the
  ephemeris, 1e-5 rad/s for the gyro): a filter that trusted a 1 pT magnetometer
  completely would let a bias step drag the attitude with it.

### Results

Full table and figures: [results/variants.md](results/variants.md) (`tools/compare_variants.py`).
Worst attitude error in each scenario's hardest condition:

| Scenario | Condition | Thesis variant | Improved variant |
|---|---|---|---|
| case 1a | eclipse | 3.8° | **0.01°** |
| case 1b (fast tumble) | eclipse | 1.8° | **0.06°** |
| case 2a | gyro bias 5 °/s step | **0.03°** | 1.7° |
| case 2b | magnetometer bias 30 µT step | 53° | **0.6°** |
| case 2c | both biases | 53° | **1.7°** |
| case 4 (high noise) | eclipse | 5.6° | **3.5°** |
| realistic_1a | eclipse, two orbits | 8.6° | **0.43°** |
| detumble (control on) | eclipse | 180° (diverges) | **0.87°** |

- The improved variant's error stays inside its own 3σ bound 99.8–100% of the time in
  every scenario (thesis variant: 43–100%), so its covariance can be trusted, e.g. to
  decide when pointing may start.
- Gyro-bias estimates are 10–400× closer to the truth outside fault windows.
- It costs about half the computation per step (0.5 vs 1.1 ms in Python).
- **Where the thesis variant is better:** a sudden 5 °/s gyro-bias step (case 2a). Its
  bias state absorbs every gyro innovation, so it follows a step instantly; the improved
  variant takes about two seconds to detect the fault and re-estimate, peaking at 1.7°.
  That is the price of a bias estimate that doesn't jitter with the sensor noise.

### Two lessons from building it

- **Classifying faults is fragile; inflating all the uncertainties is not.** The first
  version declared a gyro fault when the sun innovation was high and a magnetometer fault
  when only the magnetometer's was. A magnetometer-bias step pulls the attitude within a
  step, so the sun innovation rose too: case 2b was misread 239 times as a gyro fault and
  peaked at 22°. Inflating everything brought it to 0.7° with exactly two detections (onset
  and removal).
- **Holding the gyro rate constant over a step was the largest error source with good
  sensors.** A tumbling body's rate changes within 0.1 s; holding the previous reading
  left a systematic attitude error each step that the process noise didn't cover. The
  filter was then badly overconfident (its error inside its 3σ bound only 21–35% of the
  time in the thesis cases). Using the mean of the two readings removed it: case 1a went
  from 0.12° to 0.01° worst-case error in sunlight, and the share of errors inside 3σ to
  100% (conservatively: see the README on calibration).

## MEKF baseline

The `mekf` variant (`adcs_ref/mekf.py`, Markley & Crassidis 2014 §6.2) uses exactly the
improved variant's model, tuning, initialisation, fault detector and controller; only the
filter differs: Jacobians (exact discrete transition matrix, Joseph-form covariance update)
instead of sigma points. The question it answers: does the UKF's extra cost buy anything?

**On the ten scenarios** ([results/variants.md](results/variants.md)) the two are
indistinguishable: identical to two decimals in seven, the MEKF better on the gyro-fault
cases (worst 1.3° vs 1.7° in 2a and 2c), and identical while detumbling (eclipse worst 0.87°
since B-dot uses −ω̂ × B; before that review fix, 0.58° vs 0.69° on a different trajectory). The MEKF needs 25–60% less computation per step in Python. Once FLAE
has initialised the filter, errors stay small and the linearisation is essentially exact.

**From large initial errors** (`tools/large_error_experiment.py`; synthetic rotation,
realistic noise, honest initial uncertainty, 20 cases each), where a UKF is usually expected
to win:

| Initial error | UKF: converged (median time) | MEKF: converged (median time) |
|---|---|---|
| 10° | 20/20 (0.1 s) | 20/20 (0.2 s) |
| 30° | 20/20 (13 s) | 20/20 (15 s) |
| 60° | 4/20 (100 s) | 19/20 (30 s) |
| 90° | 1/20 (44 s) | 12/20 (30 s) |
| 150° | 6/20 (58 s) | 0/20 |

The UKF is worse from 60° on. Its sigma points spread to √(n + λ) = √10 ≈ 3.2σ, so with a
60° uncertainty they reach ~190°, beyond the 180° where the 3-parameter attitude error is
well defined; the points wrap around and corrupt the mean. A smaller λ narrows the spread
but makes the centre weight negative. Neither filter is meant to start that far off: FLAE
initialises within a degree and a detected fault inflates the attitude uncertainty by ~3°.

**Conclusion.** In this application the MEKF matches the UKF's accuracy at lower cost, with
simpler code (no 9×9 Cholesky factorisation and no sigma points per step) and better
behaviour from large errors. It is the stronger candidate for the flight implementation in
C and Rust; the UKF stays in the Python reference as the documented comparison.

## Scope of the C and Rust implementations

Frozen on 2026-11-06 as `adcs.h` ABI version 3; version 4 (2026-11-11, "Review fixes"
below) changed the flags to bytes and added `adcs_real_size()`, one configuration field and
one status bit. The C and Rust libraries implement two of the
three variants, selected with `adcs_config.variant`:

| Variant | In C and Rust | Why |
|---|---|---|
| `thesis` (`ADCS_VARIANT_THESIS`) | yes | The original algorithm: the baseline the improvements are measured against, and the heaviest filter (13 + 6 states, 27 sigma points per stage), so the most informative for the C vs Rust benchmark |
| `mekf` (`ADCS_VARIANT_MEKF`) | yes | The flight choice: the improved variant's accuracy at lower cost, simpler code, better behaviour from large errors ([MEKF baseline](#mekf-baseline)) |
| `improved` (USQUE UKF) | no | Kept in Python as the documented comparison |

What each implementation must reproduce, within the tolerances in `golden.py`:

- `testdata/golden/thesis/`: 8 vectors recorded from the thesis variant. Frozen: they predate
  the detumble scenario's move to 50 Hz dynamics and are not regenerated.
- `testdata/golden/mekf/`: the same 8 situations (nominal, bad input, gyro- and
  magnetometer-bias steps, eclipse entry, start in eclipse, high noise, control) with the
  MEKF, its noise settings matched to each scenario's sensors.

The control laws (B-dot, quaternion feedback, saturation, mode manager) are part of both
variants and frozen as they are. Pointing with magnetorquers is not tuned (see "Control"
above); improving it would change the control golden vectors, so it is left for after the
ports, together with the magnetometer duty-cycle handling.

## Input plausibility limits

A sensor reading is used only if it is finite and inside a physical range; otherwise it is
ignored and the step is flagged `BAD_INPUT`, the same as a NaN. An on-board reference vector
outside its range is treated as unavailable.

| Quantity | Accepted | Margin over the real values |
|---|---|---|
| Gyro rate norm | ≤ 50 rad/s | MEMS gyros saturate at about 35 rad/s |
| Magnetometer reading and reference field norm | 0 < \|B\| ≤ 1 mT | The field in LEO stays below 70 µT |
| Sun sensor reading and ephemeris direction norm | 0.5 to 2 | Both should be unit vectors |

Accepted directions are then normalised: FLAE and the vector measurements assume unit
vectors (see "Review fixes").

Added after fuzzing the C library: a finite but absurd gyro reading (−2·10¹⁶⁵ rad/s)
overflowed the thesis filter's state, and normalising the quaternion by an infinite norm
gave a finite all-zero quaternion that passed the filter's finiteness check. The limits are
in all implementations (`adcs_ref/limits.py`), and a quaternion with a zero or overflowed
norm now becomes NaN so the covariance-reset path catches it. Neither change affects the
golden vectors.

## Review fixes (ABI version 4)

A review of all three implementations in November 2026 (five independent passes, each
finding reproduced before fixing) found no errors in the estimators' mathematics, but these
defects, now fixed identically in Python, C and Rust. Each has a regression test in
`tests/test_review_fixes.py`; the rules and limits are named constants in all three.

| Defect | Fix |
|---|---|
| **Gyro outage:** the MEKF and UKF held the last rate reading. 60 s without the gyro while tumbling in eclipse gave 180° error with a reported 1σ of 0.12° (1466σ), and the gyro-bias estimate absorbed the error. | The rate is propagated with Euler's torque-free equations; the attitude variance grows as (σ_α t²/2)² (`sigma_outage_accel`, default 10⁻³ rad/s², about the largest coil torque), up to σ = 0.25 rad, beyond which the filters' linearisation or sigma points fail; the gyro-bias update is frozen (zero gain rows, Joseph form); status `RATE_PROPAGATED`. Same case now: 12° error at 1.3σ, rate correct once the gyro returns. |
| **Non-unit sun vectors:** lengths 0.5–2 pass the plausibility check but were used as is. From about 1.5–1.8, FLAE's Newton iteration (started at λ = 1) converged to a smaller root and returned a 180°-wrong attitude marked as success; a Sun direction of length 0.983 (an ephemeris in AU) gave the MEKF 18° error. | Both directions are normalised after the check; Newton starts at Σaᵢ\|bᵢ\|\|rᵢ\|, an upper bound of λmax, from which it converges monotonically to the largest root. |
| **Unbounded RK4 step count:** `dt/process_dt` had no limit. In C, converting more than 2³² steps to `uint32_t` is undefined behaviour; all three hang (`process_dt = 1e-6` costs seconds per step). | `dt/process_dt ≤ 1000` in the configuration rules; GCC builds also sanitize float-to-integer overflow. |
| **Single precision:** the Cholesky jitter was machine epsilon, which in float (1.2·10⁻⁷) exceeds the process noise on q and ω and acted as extra noise; the float thesis filter drifted up to 42°. | The jitter is 2.2·10⁻¹⁶ in both precisions: the float filter now stays within 0.003° of the double reference ([c_vs_rust.md](results/c_vs_rust.md)). |
| **Configuration validation differed:** C and Rust accepted `fault_steps = 0` (a fault declared every step), `flae_max_iterations = 0` (never initialises), a non-unit target quaternion, gains large enough to overflow to a NaN dipole, and more; Python accepted `dt = 0` and negative gains. | One rule set (below), implemented in all three; `tests/test_native.py` checks that C and Rust reject every configuration Python rejects. |
| **MEKF B-dot** differentiated the raw magnetometer: at 0.3 µT noise 90% of the commands saturated and many pointed the wrong way. | Ḃ = −ω̂ × B with the bias-corrected rate (the field is fixed in inertial space over a step). |
| **B-dot without the field model:** with the on-board field model unavailable, B-dot got B = 0 (a spurious saturated pulse, then no detumbling). | B-dot uses the magnetometer reading, which needs no model. |
| **Mode manager:** one bad input in SAFE aborted detumbling inside the hysteresis band and restarted the 60 s settle time from POINT. | SAFE resumes the mode it interrupted, with its timer. |
| **Non-finite dipole** commands were passed to the coils (flagged as saturated). | They become zero. |
| **Interface:** the ABI version did not encode precision, so a float library loaded as double gave garbage (or, the other way round, overflowed the caller's stack); C `bool` fields made a foreign value of 2 undefined behaviour; `adcs_step` did not check alignment in C; Rust formed a reference to the state before checking its magic number. | ABI v4: `adcs_real_size()` (and `adcs_check_abi()` in the header), byte-sized flags, alignment and variant checks in `adcs_step`, the Rust state is `repr(C)` with the magic first and read through a raw pointer, and `adcs_init` marks the memory uninitialised before any other check. |
| **Rust memory:** `adcs_init` used 8 KB of stack (a non-zero field in a constant made LLVM copy it through the stack); the double state (41.8 KB) did not fit the bare-metal example's 40 KB buffer, which CI only linked; the firmware panic handler was a silent loop. | The constant is all zero (init stack now small); the example's buffer is 48 KB and the example now runs on the host against both libraries in CI; a panic traps (`udf`) so the watchdog resets the board. |

What the review found correct, and not changed: frames, signs and units in the simulator;
FLAE, the two-stage UKF equations, the MEKF's Φ, Q and Joseph update, USQUE's GRPs; the
control-law signs; C and Rust matching each other line by line; memory safety under the
sanitizers.

### Configuration rules

`adcs_init` (and `AdcsConfig.validate` in Python) accepts a configuration only if:

| Applies to | Rule |
|---|---|
| All | 0 < `dt` ≤ 10 s; inertia finite, symmetric (to 10⁻⁹ relative) and positive definite; every `max_dipole` > 0 |
| All | FLAE weights ≥ 0 with a positive sum; `flae_tolerance` > 0; 1 ≤ `flae_max_iterations` ≤ 1000 |
| All | 0 ≤ `bdot_gain` ≤ 10⁹; 0 ≤ `kp`, `kd` ≤ 10³; 0 ≤ exit rate ≤ enter rate; `settle_time` ≥ 0; `q_target` unit to 10⁻⁶ |
| Thesis | `process_dt` > 0 and `dt/process_dt` ≤ 1000; 10⁻² ≤ `ukf_alpha` ≤ 10²; `p_x0`, `p_b0`, `v` > 0; `w_x`, `w_b` ≥ 0 |
| MEKF | noise σ, initial σ, fault σ and `fault_nis_threshold` > 0; bias walks and `sigma_outage_accel` ≥ 0; `fault_steps` ≥ 1 |

All values must be finite.

## The C library

`adcs-c/` implements `adcs.h` for the `thesis` and `mekf` variants, as a line-by-line port
of the Python reference.

- **Flight-style C11.** No heap, no recursion, no variable-length arrays, no I/O. The caller
  allocates the state (`adcs_state_size()` bytes: 36 KB in double precision, 18 KB in
  float; almost all of it the thesis UKF's sigma points and matrices). Every temporary lives
  in that state, so no function uses more than about 1.5 KB of stack (checked with
  `-Wstack-usage`; a whole step, about 3 KB). Only the four `adcs.h` functions are exported from the shared library.
- **Precision.** `adcs_real` is `double`, or `float` with `-DADCS_USE_FLOAT`. The float
  build compiles with `-Wdouble-promotion` so that no double arithmetic slips in.
- **Linear algebra** is written for the purpose (`linalg.c`): fixed-size row-major matrices,
  Cholesky, LU with partial pivoting, right division `B A⁻¹`. The filters use the same
  factorisations as the NumPy reference.
- **Configuration is validated** by `adcs_init` (positive periods and noise values,
  invertible inertia, positive coil limits). `adcs_step` before a successful `adcs_init`
  returns an error and `NOT_INITIALISED`.

Verification:

| Check | How |
|---|---|
| Matches the reference | All 16 golden vectors, by a C runner (`tests/golden_runner.c`, under ctest) and through the simulator's ctypes backend. Largest difference: 3·10⁻¹¹ (rad/s, MEKF gyro-bias case right after a fault inflates the covariance), elsewhere ~10⁻¹⁴. Tolerance 10⁻⁹. |
| Closed loop | The same 60 s scenarios (B-dot detumbling, magnetometer-bias case, realistic sensors) with the Python and C backends give identical attitude, rate, commands, modes and status to 10⁻⁸ (`tests/test_native.py`). |
| Full scenarios | `tools/crosscheck.py` runs all 10 scenarios to the end with both variants, Python and C (rerun after the review fixes). 18 of 20 runs agree to 2·10⁻¹¹ or better over up to 114 000 closed-loop steps. Two detumble runs drift: with the thesis variant the filter repeatedly loses lock while its estimate drives the coils (errors up to 180°); with the MEKF, rate-based B-dot now finishes detumbling early and the run spends hours in POINT, where the untuned pointing loop amplifies rounding by about 10× per 1000 s (10⁻¹⁴ at 1800 s, 10⁻⁶ at 11 000 s). Both grow smoothly with no discrete event first, so they are sensitivity of the closed loop, not porting errors; C and Rust stay bit-identical in the thesis detumble run. |
| Memory and undefined behaviour | Unit and golden tests under AddressSanitizer and UndefinedBehaviorSanitizer, with GCC and Clang. |
| Robustness | libFuzzer harness (`tests/fuzz_step.c`): arbitrary input sequences must never crash, the attitude must be unit-norm or NaN, and the dipole command finite and within the coil limits. 2 minutes per CI run. |
| Static analysis | cppcheck (warning, style, performance, portability) for both precisions. |
| Bare metal | `tools/cross_build.sh` builds `examples/bare_metal.c` for a Cortex-M4F (float) and a Cortex-M7 (double) with newlib's `nosys` stubs: about 24 KB of code for either target at `-O2`, no `malloc`. |

Speed, on the development PC (x86-64, one core): the C thesis variant takes about 20 µs per
step and the MEKF about 2 µs, against 755 µs and 288 µs for the NumPy reference. Through
the simulator's ctypes backend the C steps take 48 µs and 28 µs, most of it call overhead.
[c_vs_rust.md](results/c_vs_rust.md) compares C and Rust properly.

Running a scenario on the C library:

```bash
cmake -S adcs-c -B adcs-c/build -DCMAKE_BUILD_TYPE=Release && cmake --build adcs-c/build
uv run adcssim run scenarios/case_1a.toml --backend c
uv run adcssim view scenarios/detumble.toml --backend c
```

The backend loads `adcs-c/build/libadcs_c.so`, or the library named in `ADCSSIM_C_LIB`.

## The Rust library

`adcs-rs/` implements the same `adcs.h` interface, as a port of the C library: the same
modules, algorithms and operation order (so both round the same way), with the
hand-written linear algebra kept rather than replaced by a crate, so that the comparison
with C ([c_vs_rust.md](results/c_vs_rust.md)) measures the languages and not the libraries.

- **`adcs-core`** holds the algorithms. It is `no_std`, has `#![forbid(unsafe_code)]`, and
  takes its maths functions from `libm`. `Adcs` can be used directly from Rust (for example
  in a `static`), with `Result` errors instead of status codes. `#[repr(C)]` structs mirror
  the header. The flag fields are bytes in both languages (`uint8_t`/`u8`), so any value a
  foreign caller passes is valid.
- **`adcs-ffi`** exports the four C functions and is the only crate with `unsafe` code:
  turning the caller's pointers into references, each block with a `SAFETY:` comment
  stating the `adcs.h` precondition it relies on. With the default `std` feature it builds
  the shared library the simulator loads; without it, a `no_std` static library for
  firmware. Panics abort (none are expected: there is no allocation and all indices are
  fixed by array sizes).
- **Precision**: `f64`, or `f32` with the `f32` feature, like `ADCS_USE_FLOAT`.

Verification:

| Check | How |
|---|---|
| Matches the reference | All 16 golden vectors, by a Rust test (`adcs-core/tests/golden.rs`) and through the simulator's ctypes backend. The differences from the Python reference are the C library's, to the digit, in all but one vector (high noise, where they are smaller). |
| Closed loop | The 60 s closed-loop scenarios in `tests/test_native.py` run against both libraries and match the Python reference to 10⁻⁸. |
| Full scenarios | `tools/crosscheck.py` (all 10 scenarios, both variants, to the end): Rust differs from the Python reference as much as C does. Rust and C are bit-identical in 7 of the 10 thesis-variant runs, including the chaotic thesis detumble run (114 000 steps), so its drift from the Python reference comes from NumPy's rounding. Elsewhere Rust and C differ by at most 3·10⁻¹², except MEKF detumble (10⁻³, the POINT-mode sensitivity above, seeded by `libm` vs the system maths library in sine and cosine). |
| Interface | A test compiles a C program against `adcs.h` and checks every struct size and field offset against the Rust structs, in both precisions. |
| Undefined behaviour | Miri runs the unit tests, including the FFI crate's misuse tests (null and misaligned pointers, a zeroed never-initialised state, non-0/1 flag bytes); it does not exercise arbitrary garbage state. |
| Robustness | The same libFuzzer harness as the C library (`fuzz/fuzz_targets/step.rs`), built with the stable toolchain and run with debug assertions and overflow checks, 2 minutes per CI run (`tools/fuzz.sh`). |
| Lints | `clippy::pedantic` with warnings as errors, both precisions. |
| Bare metal | `tools/cross_build.sh` builds the `no_std` static library for a Cortex-M4F (float) and a Cortex-M7 (double) and links the C example `examples/bare_metal.c` against it, unchanged. No `malloc`. The Cortex-M builds can only be linked; `tools/host_example.sh` builds the same example for the host against the Rust library and runs it, in both precisions (the state fits the 48 KB buffer and the ADCS runs). |

First look at code size: with the size-optimised `firmware` profile (`opt-level = "s"`) the
firmware image has 35 KB of code (M4F) and 33 KB (M7), against about 21 KB for the C
library at `-Os`. At `opt-level = 2` it is about 90 KB: LLVM unrolls the fixed-size 13×13
loops of the UKF. [c_vs_rust.md](results/c_vs_rust.md) compares size and speed at matching
optimisation levels.

```bash
cargo build --release --manifest-path adcs-rs/Cargo.toml
uv run adcssim run scenarios/case_1a.toml --backend rust
```

The backend loads `adcs-rs/target/release/libadcs_rs.so`, or the library named in
`ADCSSIM_RUST_LIB`.
