"""Two-stage unscented Kalman filter (TSUKF), Chen et al., as derived in the thesis (§2.4.3,
eqs. 49–68) and implemented in its appendix A.1.

System (eq. 49)::

    x[k+1] = f(x[k]) + F_a b[k] + w_x        state
    b[k+1] = b[k] + w_b                       bias (random walk)
    y[k]   = h(x[k]) + F_s b[k] + v           measurement

The filter runs a bias-free UKF on ``x`` and a linear filter on ``b``, coupled through the
matrix β. The equation numbers in the comments are the thesis's.

Implementation notes (shared with the C and Rust versions):

- Sigma points use ``Xi = √3 α`` (κ = 3 − n_x, thesis appendix), so ``n_x + λ = 3α²``.
- Covariances are symmetrised after each update, and Cholesky failures are handled by
  adding increasing jitter; if that fails too, the covariance is reset to its initial value
  and the step is flagged.
- A measurement can be marked unavailable (e.g. the sun sensor in eclipse): its noise is
  inflated so the update ignores it, and its value is replaced by the prediction.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BatchFn = Callable[[FloatArray], FloatArray]
"""Function applied to a batch of row vectors, shape (K, n) → (K, m)."""

EPS = float(np.finfo(np.float64).eps)
UNAVAILABLE_VARIANCE = 1e12
"""Measurement variance used for an unavailable measurement (effectively infinite)."""
_JITTER = (0.0, 1e-12, 1e-10, 1e-8)


def _sym(p: FloatArray) -> FloatArray:
    return 0.5 * (p + p.T)


def robust_cholesky(p: FloatArray) -> FloatArray | None:
    """Lower Cholesky factor of ``p + εI`` (thesis), retrying with more jitter if needed."""
    n = p.shape[0]
    scale = max(float(np.max(np.abs(np.diag(p)))), 1e-300)
    for jitter in _JITTER:
        try:
            return np.linalg.cholesky(p + (EPS + jitter * scale) * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    return None


@dataclass(frozen=True)
class UtWeights:
    xi: float
    """Sigma-point spread √(n + λ)."""
    wm0: float
    wc0: float
    wi: float

    @classmethod
    def thesis(cls, alpha: float, n: int, beta: float = 2.0) -> "UtWeights":
        """Eq. 35 with κ = 3 − n, so n + λ = 3α²."""
        lam = 3.0 * alpha**2 - n
        wm0 = lam / (n + lam)
        return cls(
            xi=float(np.sqrt(n + lam)),
            wm0=wm0,
            wc0=wm0 + 1.0 - alpha**2 + beta,
            wi=1.0 / (2.0 * (n + lam)),
        )


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Quantities of the last measurement update, kept for analysis only."""

    innovation: FloatArray
    """y − ŷ (bias-filter prediction), with NaN for unavailable measurements."""
    innovation_cov: FloatArray
    """P_y, the predicted innovation covariance."""
    bias_gain: FloatArray
    """K_b, the bias filter's gain (n_b × n_y)."""
    available: NDArray[np.bool_]


@dataclass
class TsukfState:
    x: FloatArray
    px: FloatArray
    b: FloatArray
    pb: FloatArray
    beta: FloatArray


class TwoStageUkf:
    def __init__(
        self,
        fa: FloatArray,
        fs: FloatArray,
        w_x: FloatArray,
        w_b: FloatArray,
        v: FloatArray,
        alpha: float,
    ) -> None:
        self.nx, self.nb = fa.shape
        self.ny = fs.shape[0]
        self.fa, self.fs = fa, fs
        self.w_x, self.w_b, self.v = np.diag(w_x), np.diag(w_b), np.diag(v)
        self.weights = UtWeights.thesis(alpha, self.nx)
        self.state: TsukfState | None = None
        self._initial: tuple[FloatArray, FloatArray] | None = None
        self.last_update: UpdateDiagnostics | None = None

    def reset(self, x0: FloatArray, px0: FloatArray, b0: FloatArray, pb0: FloatArray) -> None:
        self.state = TsukfState(
            x=x0.astype(np.float64).copy(),
            px=np.diag(px0).astype(np.float64),
            b=b0.astype(np.float64).copy(),
            pb=np.diag(pb0).astype(np.float64),
            beta=np.zeros((self.nx, self.nb)),
        )
        self._initial = (np.diag(px0).astype(np.float64), np.diag(pb0).astype(np.float64))

    def _sigma_points(self, x: FloatArray, lower: FloatArray) -> FloatArray:
        spread = self.weights.xi * lower.T  # row i = ξ · column i of L
        return np.vstack([x, x + spread, x - spread])

    def _transform(self, out: FloatArray, lower: FloatArray) -> FloatArray:
        """Eqs. 50/57: ``Σ_i [g(χ_i) − g(χ_{i+n})] e_iᵀ / (2ξ) · L⁻¹``."""
        n = self.nx
        diff = (out[1 : n + 1] - out[n + 1 :]).T  # column i: g(χ_i) − g(χ_{i+n})
        return np.linalg.solve(lower.T, diff.T).T / (2.0 * self.weights.xi)

    def _mean(self, out: FloatArray) -> FloatArray:
        w = self.weights
        return np.asarray(w.wm0 * out[0] + w.wi * out[1:].sum(axis=0), dtype=np.float64)

    def _cov(self, dev: FloatArray) -> FloatArray:
        w = self.weights
        return w.wc0 * np.outer(dev[0], dev[0]) + w.wi * dev[1:].T @ dev[1:]

    def step(
        self,
        f: BatchFn,
        y: FloatArray,
        available: NDArray[np.bool_],
        h: BatchFn | None = None,
        normalize_pred: Callable[[FloatArray], FloatArray] | None = None,
        normalize_out: Callable[[FloatArray], FloatArray] | None = None,
    ) -> bool:
        """One predict + update. Returns False if a covariance had to be reset.

        ``normalize_pred`` is applied to the predicted mean and ``normalize_out`` to the
        result (the thesis renormalises q after the prediction, and q and the sun vector at
        the output)."""
        if self.state is None or self._initial is None:
            raise RuntimeError("filter not initialised")
        s = self.state
        ok = True
        h_fn: BatchFn = h if h is not None else (lambda xs: xs)

        # ---- prediction -------------------------------------------------------------
        lower = robust_cholesky(s.px)
        if lower is None:
            s.px, ok = self._initial[0].copy(), False
            lower = np.linalg.cholesky(s.px)
        chi = self._sigma_points(s.x, lower)
        fx = f(chi)
        m_mat = self._transform(fx, lower)  # eq. 50
        r_mat = m_mat @ s.beta + self.fa  # eq. 51
        mk = self._mean(fx)  # eq. 52
        x_pred = mk + self.fa @ s.b  # eq. 53
        if normalize_pred is not None:
            x_pred = normalize_pred(x_pred)
        px_pred = (
            self._cov(fx - mk)
            - m_mat @ s.beta @ s.pb @ s.beta.T @ m_mat.T
            + r_mat @ s.pb @ r_mat.T
            + self.w_x
        )  # eq. 54
        pb_pred = s.pb + self.w_b  # eq. 56
        beta_pred = r_mat @ s.pb @ np.linalg.inv(pb_pred)  # eq. 55
        b_pred = s.b

        # ---- measurement sigma points -----------------------------------------------
        px_pred = _sym(px_pred)
        lower2 = robust_cholesky(px_pred)
        if lower2 is None:
            px_pred, ok = self._initial[0].copy(), False
            lower2 = np.linalg.cholesky(px_pred)
        chi2 = self._sigma_points(x_pred, lower2)
        hx = h_fn(chi2)
        n_mat = self._transform(hx, lower2)  # eq. 57
        h_mat = n_mat @ beta_pred + self.fs  # eq. 58
        nk = self._mean(hx)  # eq. 59
        gamma = hx - nk

        v = self.v.copy()
        y_eff = np.where(available, y, nk + self.fs @ b_pred)
        v[~available, ~available] = UNAVAILABLE_VARIANCE

        # ---- bias-free filter (eq. 60) ------------------------------------------------
        x_t = x_pred - beta_pred @ b_pred
        px_t = px_pred - beta_pred @ pb_pred @ beta_pred.T
        y_t = nk - n_mat @ beta_pred @ b_pred
        py_t = self._cov(gamma) - n_mat @ beta_pred @ pb_pred @ beta_pred.T @ n_mat.T + v
        k_t = np.linalg.solve(py_t.T, (px_t @ n_mat.T).T).T
        beta_upd = beta_pred - k_t @ h_mat  # eq. 61
        x_t_upd = x_t + k_t @ (y_eff - y_t)  # eq. 62
        px_t_upd = _sym(px_t - k_t @ py_t @ k_t.T)  # eq. 63

        # ---- bias filter (eqs. 64–66) ------------------------------------------------
        y_b = nk + self.fs @ b_pred
        p_by = pb_pred @ h_mat.T
        p_y = py_t + h_mat @ pb_pred @ h_mat.T
        k_b = np.linalg.solve(p_y.T, p_by.T).T
        b_upd = b_pred + k_b @ (y_eff - y_b)
        self.last_update = UpdateDiagnostics(
            innovation=np.where(available, y_eff - y_b, np.nan),
            innovation_cov=p_y,
            bias_gain=k_b,
            available=available.copy(),
        )
        pb_upd = _sym(pb_pred - k_b @ p_y @ k_b.T)

        # ---- coupled result (eqs. 67–68) ---------------------------------------------
        x_upd = x_t_upd + beta_upd @ b_upd
        px_upd = _sym(px_t_upd + beta_upd @ pb_upd @ beta_upd.T)
        if normalize_out is not None:
            x_upd = normalize_out(x_upd)

        if not (np.all(np.isfinite(x_upd)) and np.all(np.isfinite(px_upd))):
            s.px, s.pb = self._initial[0].copy(), self._initial[1].copy()
            s.beta = np.zeros_like(s.beta)
            return False
        s.x, s.px, s.b, s.pb, s.beta = x_upd, px_upd, b_upd, pb_upd, beta_upd
        return ok
