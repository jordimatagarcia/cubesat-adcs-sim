"""Fast Linear Quaternion Attitude Estimator (FLAE), J. Wu et al. (2018), as described in the
thesis (§4.2.2, eqs. 99–113).

Solves Wahba's problem for vector pairs ``b_i = A r_i``: build the attitude profile matrix
``B = Σ a_i b_i r_iᵀ`` and the 4×4 matrix ``W`` from its columns (eq. 105), find the largest
root λ of ``det(W − λI) = λ⁴ + τ1 λ² + τ2 λ + τ3`` by Newton's method starting from
``λ = Σ a_i`` (eqs. 106–109), and solve ``(W − λI) q = 0`` by elimination (eqs. 110–113).

``W``'s eigenvector is the attitude quaternion in scalar-first order; the result is returned
scalar-last, the project convention (checked in the tests against known attitudes).

Two implementation choices beyond the thesis text:

- The elimination uses full pivoting and fixes the free component of the eigenvector to
  whichever unknown is left last. The thesis (and Wu's paper) fix ``q4 = −1``, which divides
  by zero when that component of the attitude is near zero.
- The weights are normalised to sum to 1, and Newton starts from Σ aᵢ|bᵢ||rᵢ| (1 for unit
  vectors), an upper bound of the largest eigenvalue.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

MIN_SIN_ANGLE = 1e-3
"""Vector pairs closer to parallel than ~0.06° carry no attitude information."""


@dataclass(frozen=True)
class FlaeResult:
    q: FloatArray
    """Attitude quaternion, scalar-last, inertial → body. NaN if ``ok`` is False."""
    ok: bool
    iterations: int
    lambda_max: float


def _w_matrix(b: FloatArray) -> FloatArray:
    """Eq. 105, with H_x, H_y, H_z the columns of the attitude profile matrix ``b``."""
    hx, hy, hz = b[:, 0], b[:, 1], b[:, 2]
    return np.array(
        [
            [hx[0] + hy[1] + hz[2], -hy[2] + hz[1], -hz[0] + hx[2], -hx[1] + hy[0]],
            [-hy[2] + hz[1], hx[0] - hy[1] - hz[2], hx[1] + hy[0], hx[2] + hz[0]],
            [-hz[0] + hx[2], hx[1] + hy[0], hy[1] - hx[0] - hz[2], hy[2] + hz[1]],
            [-hx[1] + hy[0], hx[2] + hz[0], hy[2] + hz[1], hz[2] - hy[1] - hx[0]],
        ]
    )


def characteristic_coefficients(b: FloatArray, w: FloatArray) -> tuple[float, float, float]:
    """τ1, τ2, τ3 of eq. 107: ``det(W − λI) = λ⁴ + τ1 λ² + τ2 λ + τ3``."""
    hx, hy, hz = b[:, 0], b[:, 1], b[:, 2]
    tau1 = -2.0 * float(np.sum(b * b))
    tau2 = 8.0 * float(
        hx[2] * hy[1] * hz[0]
        - hx[1] * hy[2] * hz[0]
        - hx[2] * hy[0] * hz[1]
        + hx[0] * hy[2] * hz[1]
        + hx[1] * hy[0] * hz[2]
        - hx[0] * hy[1] * hz[2]
    )
    tau3 = float(np.linalg.det(w))
    return tau1, tau2, tau3


def null_vector(n: FloatArray) -> FloatArray | None:
    """Non-zero solution of ``n x = 0`` for a 4×4 matrix of rank 3, by Gaussian elimination
    with full pivoting. ``None`` if the rank is lower (degenerate geometry)."""
    a = n.astype(np.float64, copy=True)
    cols = np.arange(4)
    scale = float(np.max(np.abs(a))) or 1.0
    for k in range(3):
        sub = np.abs(a[k:, k:])
        flat = int(np.argmax(sub))
        i = flat // sub.shape[1] + k
        j = flat % sub.shape[1] + k
        if sub[i - k, j - k] < 1e-9 * scale:
            return None
        a[[k, i]] = a[[i, k]]
        a[:, [k, j]] = a[:, [j, k]]
        cols[[k, j]] = cols[[j, k]]
        a[k + 1 :] -= np.outer(a[k + 1 :, k] / a[k, k], a[k])
    # Upper-triangular in the first three rows; the last (pivoted) unknown is free.
    x = np.zeros(4)
    x[3] = 1.0
    for k in (2, 1, 0):
        x[k] = -(a[k, k + 1 :] @ x[k + 1 :]) / a[k, k]
    out = np.empty(4)
    out[cols] = x
    return out


def flae(
    v_body: ArrayLike,
    v_ref: ArrayLike,
    weights: ArrayLike,
    q_prev: ArrayLike | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 20,
) -> FlaeResult:
    """Attitude from ``N >= 2`` vector observations.

    ``v_body`` and ``v_ref`` are ``(N, 3)`` unit vectors in the body and inertial frames.
    The sign of the result is chosen to be closest to ``q_prev`` (thesis §5.4.1), so the
    estimate doesn't flip between ``q`` and ``−q``.
    """
    bv = np.asarray(v_body, dtype=np.float64)
    rv = np.asarray(v_ref, dtype=np.float64)
    a = np.asarray(weights, dtype=np.float64)
    fail = FlaeResult(np.full(4, np.nan), False, 0, float("nan"))
    if bv.shape != rv.shape or bv.ndim != 2 or bv.shape[0] < 2 or a.shape != (bv.shape[0],):
        return fail
    if a.sum() <= 0.0:
        return fail
    a = a / a.sum()
    # Need at least two non-parallel directions.
    if np.max(np.linalg.norm(np.cross(bv[0], bv[1:]), axis=-1)) < MIN_SIN_ANGLE:
        return fail

    b = (a[:, None, None] * bv[:, :, None] * rv[:, None, :]).sum(axis=0)  # Σ a b rᵀ
    w = _w_matrix(b)
    tau1, tau2, tau3 = characteristic_coefficients(b, w)

    # Newton from Σ a |b| |r|, an upper bound of λmax: from above it converges monotonically
    # to the largest root. (λ = 1 assumed unit vectors; with longer ones it could converge to
    # a smaller root and return a 180°-wrong attitude marked as success.)
    lam = float(np.sum(a * np.linalg.norm(bv, axis=1) * np.linalg.norm(rv, axis=1)))
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        f = lam**4 + tau1 * lam**2 + tau2 * lam + tau3
        df = 4.0 * lam**3 + 2.0 * tau1 * lam + tau2
        if df == 0.0:
            return fail
        step = f / df
        lam -= step
        if abs(step) < tolerance:
            break
    else:
        return fail

    v = null_vector(w - lam * np.eye(4))
    if v is None:
        return fail
    q = np.array([v[1], v[2], v[3], v[0]])  # scalar-first eigenvector → scalar-last
    q /= np.linalg.norm(q)
    if q_prev is not None:
        qp = np.asarray(q_prev, dtype=np.float64)
        if np.all(np.isfinite(qp)) and float(q @ qp) < 0.0:
            q = -q
    elif q[3] < 0.0:
        q = -q
    return FlaeResult(q, True, iterations, float(lam))
