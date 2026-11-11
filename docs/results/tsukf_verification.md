# TSUKF vs Chen et al. (2018): is the Rust filter the paper's algorithm?

Scope: `adcs-rs/adcs-core/src/tsukf.rs` checked against Chen et al., *Two-stage unscented
Kalman filter algorithm for fault estimation in spacecraft attitude control system*, IET
Control Theory Appl. 12(13), 2018 (Theorem 1, eqs. 1–9), and the derivation the code cites,
thesis §2.4.3 (eqs. 49–68) and its MATLAB listing (Appendix A.1). §5 extends the check to
the C (`adcs-c/src/tsukf.c`) and Python (`src/adcssim/adcs_ref/tsukf.py`) versions.

**Verdict: all three implementations (Rust, C, Python) implement the paper correctly.** The
Rust matches the equations term by term, and on synthetic data it agrees with an independent
full (state + bias) UKF to ~1e-12 relative, which for this model is the exact answer. Python
agrees with the same reference to ~1e-12 and with Rust to ~1e-14; C is statement-for-statement
the Rust and reproduces the Python golden vectors. One low-severity robustness issue, shared by
all three, was found in a recovery path and is now fixed (§6).

## 1. Paper → thesis

Thesis eqs. 49–68 reproduce the paper's Theorem 1, including the parts most often got wrong:
the `1/(2√3α)` factor in M and N (valid because κ = 3 − n gives n + λ = 3α² for any n), the
sign of `−NβPᵇβᵀNᵀ` in P̃ʸʸ, the `−MβPᵇβᵀMᵀ + RPᵇRᵀ` terms in Pˣ, the β prediction and
update, and the final combination.

Errors found in the thesis (none reached the code):

| Where | Error | Correct |
|---|---|---|
| eq. 64 | P̃ᵇ (never defined) in Pᵇʸ and Pʸ | Pᵇₖ₊₁\|ₖ |
| §2.4.2 eq. 44 | adds Pˣₖ\|ₖ | process noise W |
| §2.4.2 eq. 46 | adds Pʸₖ\|ₖ | measurement noise V |
| §2.4.2 eq. 46 | Pˣʸ built from f(χₖ\|ₖ) | the redrawn points χₖ₊₁\|ₖ |
| §2.4.2 eq. 45 | sigma points without the √(n+λ) scale | scaled, as in eq. 34 |
| p. 25 | "ith **row**" of the matrix square root | ith **column** of the lower Cholesky factor |

Paper typo: eq. 59 has Cₖ₊₁ where Nₖ₊₁ is meant (derivation only).

## 2. Thesis → Rust

Every equation maps to the code, and the code agrees with the thesis MATLAB line for line:

| Equations | Code | |
|---|---|---|
| 35: λ = 3α² − n, weights, ξ = √(n+λ) | `configure` | ✓ |
| 34/45: ±ξ·(column i of lower Cholesky) | `sigma_points` | ✓ |
| 50/57: M, N = [g(χᵢ) − g(χᵢ₊ₙ)]ᵢ · L⁻¹ / (2ξ) | `transform` | ✓ |
| 51–56: R, m, x̂, Pˣ, Pᵇ, β prediction | `step`, prediction | ✓ |
| 58–59: H = Nβ + Fˢ, n | `step`, measurement | ✓ |
| 60–63: bias-free filter, β update | `step` | ✓ |
| 64–66: bias filter (with Pᵇ, not P̃ᵇ) | `step` | ✓ |
| 67–68: x̂ = x̃ + βb̂, Pˣ = P̃ˣ + βPᵇβᵀ | `step` | ✓ |

With Fᵃ = 0 the `−MβPᵇβᵀMᵀ` and `+RPᵇRᵀ` terms cancel exactly, and with h(x) = x, N = I.
The code still computes both in full, which costs time but not accuracy.

## 3. Behaviour not in the paper

| Addition | Assessment |
|---|---|
| Unavailable channel: y := ŷ, variance 1e12 | Correct: zero full innovation, the channel is effectively dropped (confirmed numerically, §4). |
| Quaternion measurement sign | FLAE aligns q_meas to the filter's q, so no ±q jumps. |
| ε·I before each Cholesky, then growing jitter | From the thesis; sigma points and M/N use the same factor, so it stays consistent. |
| q renormalised after predict; q, sun at output | Also in the thesis MATLAB. Covariance is not projected: the usual additive-quaternion compromise. |
| Body vectors rotate as −ω×v on the state's own vectors | Dynamics model change, not a filter change; justified in docs/adcs.md ("Eclipse"). |
| **Cholesky-failure branches** (prediction and measurement side, in all three implementations) | **Finding, low severity, fixed (§6).** They reset Pˣ to P₀ˣ but kept β and Pᵇ, so P̃ˣ = P₀ˣ − βPᵇβᵀ could be indefinite. `reset_covariance` resets all three; these branches now do the equivalent. |

## 4. Numerical check

For this model (Fᵃ = 0, linear h) the two-stage filter is *algebraically identical* to a plain
UKF on the augmented state [x; b] (paper eqs. 10–22), even for nonlinear f: the augmented
sigma set's x-components are the two-stage set plus points at x̂, which carry the same total
weight because κ = 3 − dim, and the cross-covariance it produces is exactly MβPᵇ. So a correct
implementation must agree with the augmented UKF to rounding.

Setup: a Rust driver runs `Tsukf` on a synthetic truth trajectory (drifting biases, noisy
measurements, 300 steps of 0.1 s) and dumps every input and output; a separate numpy UKF on
the 19-dimensional augmented state replays the same inputs. The repo's normalisation steps
are mirrored on the mean only. Values are the largest relative difference over the run.

| Case | x | b | Pˣ | Pᵇ |
|---|---|---|---|---|
| Linear f, α = 1 | 8.5e-13 | 1.2e-12 | 3.7e-13 | 3.4e-13 |
| Linear f, α = 0.3 | 8.3e-13 | 1.4e-12 | 3.8e-13 | 3.5e-13 |
| Nonlinear f (attitude kinematics + Euler), α = 1 | 8.9e-13 | 2.8e-12 | 9.7e-13 | 9.1e-13 |
| Nonlinear f, α = 0.3 | 8.9e-13 | 2.8e-12 | 9.8e-13 | 9.2e-13 |
| Nonlinear f, 30 % of sensor groups dropped | 7.8e-13 | 3.0e-12 | 1.3e-12 | 1.2e-12 |
| Linear f, 30 % dropped | 7.3e-13 | 1.5e-12 | 3.7e-13 | 3.3e-13 |
| Repo default tuning, nonlinear f (ε·I mirrored) | 6.5e-12 | 7.1e-11 | 1.1e-14 | 8.9e-11 |
| **Negative control: reference with Wᵇ × 1.5** | **1.1e-3** | **3.6e-3** | **9.6e-3** | **1.0e-2** |

With the repo's default tuning the first comparison gave ~1e-7; all of it came from the ε·I
the Rust adds before each Cholesky (relevant because Pᵇ for the magnetometer is 1e-12).
Adding the same ε to the reference closed it to ~1e-10.

Recovery branch: with the default tuning P₀ˣ − βPᵇβᵀ reaches a minimum eigenvalue of
−6.9e-8 during a run, so the issue in §3 is reachable. Forcing the prediction Cholesky to fail
at step 200 (two seeds), the filter kept positive-definite covariances and needed no further
resets in the following steps, so in practice it degrades gracefully.

Baseline: all 15 Rust tests pass (`adcs-ffi`'s layout test needs `CC=gcc` on this machine,
which has no `cc`).

The harness (Rust driver + `reference.py`) was run from a scratch directory and is not in the
repo.

## 5. C and Python

**Python** (`tsukf.py`) is the general form of the same algorithm: it takes Fᵃ, Fˢ and h
as inputs instead of hard-coding Fᵃ = 0 and h(x) = x, and computes M, N, R, H the same way
(`_transform` is `diff · L⁻¹ / (2ξ)` via a triangular solve). Replayed on the same dumps
through the harness of §4:

| Case | vs augmented UKF (x / b / Pˣ / Pᵇ) | vs Rust (x / b / Pˣ / Pᵇ) |
|---|---|---|
| Linear f, α = 1 | 8.5e-13 / 1.2e-12 / 3.7e-13 / 3.4e-13 | 1.6e-14 / 6.7e-14 / 1.0e-14 / 3.6e-15 |
| Linear f, α = 0.3 | 8.4e-13 / 1.3e-12 / 3.8e-13 / 3.4e-13 | 1.1e-13 / 5.0e-13 / 1.1e-14 / 1.0e-14 |
| Nonlinear f, α = 1 | 9.0e-13 / 2.8e-12 / 9.7e-13 / 9.1e-13 | 2.2e-14 / 8.7e-14 / 1.6e-14 / 1.5e-14 |
| Nonlinear f, α = 0.3 | 9.0e-13 / 2.7e-12 / 1.0e-12 / 9.4e-13 | 1.5e-13 / 5.5e-13 / 1.6e-13 / 1.6e-13 |
| Nonlinear f, 30 % dropped | 7.7e-13 / 2.9e-12 / 1.3e-12 / 1.2e-12 | 2.1e-14 / 7.0e-14 / 2.2e-14 / 2.2e-14 |
| Linear f, 30 % dropped | 7.3e-13 / 1.5e-12 / 3.6e-13 / 3.2e-13 | 1.6e-14 / 6.1e-14 / 7.8e-15 / 6.4e-15 |
| Repo default tuning (ε·I mirrored), two seeds | ≤ 6.5e-12 / 7.2e-11 / 1.2e-14 / 8.9e-11 | ≤ 1.6e-13 / 2.6e-13 / 1.7e-15 / 2.0e-14 |

Differences from Rust/C, none affecting the algorithm:

- The same partial reset in both Cholesky-failure branches (fixed with the others, §6).
- A singular Pᵇ prediction or innovation covariance raises `LinAlgError` (`np.linalg.inv` /
  `solve`) where Rust and C call `reset_covariance`. Not reached in any run here.
- The Fᵃ ≠ 0 and nonlinear-h paths exist only in Python and were not exercised (the repo
  always passes Fᵃ = 0 and h = identity).

**C** (`tsukf.c`) is the same code as `tsukf.rs` statement for statement: configure, reset,
`robust_cholesky`, `sigma_points`, `transform`, `ut_mean`, `ut_cov`, `step` and
`reset_covariance` match line by line, including both Cholesky-failure branches. Run: all 17 C tests pass, including the 8 thesis golden scenarios,
which compare the C ADCS (TSUKF included) with the Python reference at 1e-9 (1e-13 for the
magnetometer bias). Together with the Python result above this ties C to the paper as well.

Building the C tests on this Windows machine (Strawberry gcc 13.2, MinGW) needed a scratch
copy without `-Werror` (MinGW's `isfinite` trips `-Wfloat-conversion`), `__USE_MINGW_ANSI_STDIO`
for `%zu`, and an `aligned_alloc` shim. These are portability issues of the test build on
MinGW, not of the filter; the repo's sources were not changed.

## 6. Fix of the recovery branches

The same change in `tsukf.rs`, `tsukf.c` and `tsukf.py`:

- **Prediction factorisation fails:** restart Pˣ, Pᵇ and β together (`reset_covariance`),
  keeping the estimate, instead of resetting Pˣ alone.
- **Measurement factorisation fails:** reset the predicted Pˣ to P₀ˣ as before and also set
  the predicted β to zero, so the bias-free covariance is P₀ˣ (positive definite). The bias
  covariance is kept: the bias estimate is unaffected by the state factorisation failing.

Regression tests (two per implementation: `tsukf.rs` `mod tests`,
`adcs-c/tests/test_unit.c`, `tests/test_tsukf_recovery.py`) force each branch and compare the
step with an equivalent clean filter: a filter freshly reset to P₀ with the same estimate for
the prediction branch, and one with Pˣ = P₀ˣ, β = 0, the same Pᵇ and no process noise on x
for the measurement branch. With the old code all six tests fail; with the fix they pass.

Neither branch runs in normal operation, so nothing else changes: the thesis golden vectors
still pass (Rust and C), and the augmented-UKF comparison of §4–5 gives the same numbers.
Rust was checked in double and single precision. The new Python tests were run with numpy
outside pytest (pytest is not installed on this machine); the rest of the Python suite was
not run.

## Recommendations

1. ~~Make the two Cholesky-failure branches reset β and Pᵇ as well~~: done (§6).
2. Optionally, add the augmented-UKF comparison as a permanent test: it pins the algorithm to
   the paper exactly, independent of the golden vectors.
3. Note that the default `ukf_alpha = 1.0` lies just outside the paper's α ∈ [1e-4, 1); the
   filter is exact at α = 1, so this is cosmetic.
