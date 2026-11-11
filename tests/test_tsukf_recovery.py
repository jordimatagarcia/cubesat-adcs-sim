"""TSUKF recovery after a failed Cholesky factorisation (docs/results/tsukf_verification.md).

Resetting only Px while keeping the old β and Pb can leave the bias-free covariance
Px − β Pb βᵀ indefinite. The fix restarts Px, Pb and β together when the prediction
factorisation fails, and drops the predicted β when the measurement factorisation fails.
Each test compares the recovering filter with an equivalent clean one. The same tests are in
tsukf.rs and adcs-c/tests/test_unit.c."""

import copy

import numpy as np

from adcssim.adcs_ref.adcs import fault_matrices
from adcssim.adcs_ref.tsukf import TwoStageUkf

NX, NB = 13, 6
ALPHA = 1.0
W_B = np.full(NB, 1e-6)
V = np.full(NX, 1e-3)
PX0 = np.full(NX, 1e-2)
PB0 = np.full(NB, 1e-2)
ALL = np.ones(NX, dtype=bool)


def identity(xs: np.ndarray) -> np.ndarray:
    return xs.copy()


def y() -> np.ndarray:
    out = np.zeros(NX)
    out[3] = 1.0
    out[6] = 1.0
    out[7:10] = [0.3, -0.2, 0.1]
    out[10:13] = [0.02, -0.01, 0.015]
    return out


def make(w_x: np.ndarray) -> TwoStageUkf:
    fa, fs = fault_matrices()
    return TwoStageUkf(fa, fs, w_x, W_B, V, ALPHA)


def warmed() -> TwoStageUkf:
    """A filter after a few steps, so that β and Pb are no longer their initial values."""
    f = make(np.full(NX, 1e-6))
    x0 = np.zeros(NX)
    x0[3] = 1.0
    x0[6] = 1.0
    f.reset(x0, PX0, np.zeros(NB), PB0)
    for _ in range(5):
        assert f.step(identity, y(), ALL)
    assert f.state is not None and np.any(f.state.beta != 0.0)
    return f


def assert_same(a: TwoStageUkf, b: TwoStageUkf) -> None:
    assert a.state is not None and b.state is not None
    for name in ("x", "px", "b", "pb"):
        np.testing.assert_allclose(
            getattr(a.state, name), getattr(b.state, name), rtol=0, atol=1e-12, err_msg=name
        )


def test_failed_prediction_cholesky_restarts_all_covariances() -> None:
    a = warmed()
    assert a.state is not None
    b = copy.deepcopy(a)
    b.reset(a.state.x, PX0, a.state.b, PB0)
    a.state.px[0, 0] = -1.0
    assert not a.step(identity, y(), ALL)
    assert b.step(identity, y(), ALL)
    assert_same(a, b)


def test_failed_measurement_cholesky_drops_the_coupling() -> None:
    """The step equals one from a filter with Px = P0x, β = 0, the same Pb and no process
    noise on x (an identity process then predicts exactly that)."""
    a = warmed()
    assert a.state is not None
    b = copy.deepcopy(a)
    assert b.state is not None
    b.state.px = np.diag(PX0)
    b.state.beta = np.zeros_like(b.state.beta)
    b.w_x = np.zeros((NX, NX))
    w_x = np.full(NX, 1e-6)
    w_x[0] = -1.0  # makes the predicted Px indefinite
    a.w_x = np.diag(w_x)
    assert not a.step(identity, y(), ALL)
    assert b.step(identity, y(), ALL)
    assert_same(a, b)
