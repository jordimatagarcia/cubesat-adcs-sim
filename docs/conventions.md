# Conventions

These conventions apply to the simulator, the C library and the Rust library. Any code that
breaks one of them is a bug.

## Units

SI units everywhere in code and logs: metres, seconds, radians, tesla, A·m², N·m, kg·m².
Degrees and nanotesla are used only for display and in scenario files, where the field name
says so (`rate_deg_s`, `bias_nT`).

## Time

- Simulation time `t` is seconds since the scenario epoch.
- Absolute time is UTC, converted to a Julian date for SGP4, GMST and the Sun model.

## Reference frames

| Name | Symbol | Definition |
|---|---|---|
| Inertial | `i` | TEME, the frame SGP4 outputs. Used as "ECI" everywhere, consistently. |
| Earth-fixed | `e` | ECEF, rotated from TEME by GMST about the z-axis. |
| Body | `b` | Fixed to the spacecraft structure: origin at the centre of mass, axes along the 3U principal axes, z along the long axis. |

Using TEME instead of J2000/GCRS makes no difference to attitude estimation as long as the
simulator and the ADCS use the same frame, which they do. The difference to GCRS is under
half a degree and is documented, not corrected.

## Attitude quaternions

- **Scalar-last:** `q = [q1, q2, q3, q4]`, with `q4` the scalar part, as in the thesis and in
  Markley & Crassidis (2014).
- `q` describes the rotation **from the inertial frame to the body frame**: a vector known in
  the inertial frame is mapped to body-frame coordinates with `v_b = A(q) v_i`.
- Quaternions are kept at unit norm. When the sign is free (`q` and `-q` are the same
  attitude), estimators keep it consistent with the previous estimate.
- Quaternion product `⊗` follows Markley & Crassidis, so that `A(q ⊗ p) = A(q) A(p)`.

## Vectors and matrices

- `[v×]` is the skew-symmetric cross-product matrix, `[v×] u = v × u`.
- Arrays of vectors are stored as rows: an `N×3` array holds `N` vectors.
- The C and Rust libraries store matrices in row-major order.
