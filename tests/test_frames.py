import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from adcssim.frames import (
    IDENTITY_QUAT,
    WGS84_A,
    WGS84_F,
    dcm_to_quat,
    ecef_to_geodetic,
    ecef_to_spherical,
    ecef_to_teme,
    geodetic_to_ecef,
    quat_angle,
    quat_conj,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_dcm,
    skew,
    spherical_to_ecef_vector,
    teme_to_ecef,
    teme_to_ecef_velocity,
)
from adcssim.timebase import gmst, julian_date

finite = st.floats(-1.0, 1.0, allow_nan=False)
vectors = arrays(np.float64, 3, elements=st.floats(-1e3, 1e3, allow_nan=False))


@st.composite
def quaternions(draw: st.DrawFn) -> np.ndarray:
    q = draw(arrays(np.float64, 4, elements=finite))
    if np.linalg.norm(q) < 1e-3:
        q = np.array([0.1, -0.2, 0.3, 0.9])
    return quat_normalize(q)


@given(quaternions(), quaternions())
def test_product_composes_attitude_matrices(q: np.ndarray, p: np.ndarray) -> None:
    np.testing.assert_allclose(
        quat_to_dcm(quat_mul(q, p)), quat_to_dcm(q) @ quat_to_dcm(p), atol=1e-12
    )


@given(quaternions())
def test_attitude_matrix_is_a_rotation(q: np.ndarray) -> None:
    a = quat_to_dcm(q)
    np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(a) == pytest.approx(1.0, abs=1e-12)


@given(quaternions())
def test_dcm_to_quat_inverts_quat_to_dcm(q: np.ndarray) -> None:
    back = dcm_to_quat(quat_to_dcm(q))
    assert back[3] >= 0.0
    # q and −q are the same attitude (and both have q4 = 0 for a 180° rotation).
    assert min(np.abs(back - q).max(), np.abs(back + q).max()) < 1e-10


@given(quaternions())
def test_conjugate_is_inverse(q: np.ndarray) -> None:
    np.testing.assert_allclose(quat_mul(q, quat_conj(q)), IDENTITY_QUAT, atol=1e-12)


@given(quaternions(), vectors)
def test_rotate_matches_matrix(q: np.ndarray, v: np.ndarray) -> None:
    np.testing.assert_allclose(quat_rotate(q, v), quat_to_dcm(q) @ v, atol=1e-9)


@given(vectors, vectors)
def test_skew_is_cross_product(v: np.ndarray, u: np.ndarray) -> None:
    np.testing.assert_allclose(skew(v) @ u, np.cross(v, u), atol=1e-8)


def test_axis_angle_is_a_frame_rotation() -> None:
    # Rotating the body frame +90° about z: the inertial x axis appears as body −y.
    q = quat_from_axis_angle([0, 0, 1], math.pi / 2)
    np.testing.assert_allclose(quat_rotate(q, [1, 0, 0]), [0, -1, 0], atol=1e-15)


@given(arrays(np.float64, 3, elements=finite), st.floats(-math.pi, math.pi))
def test_axis_angle_matches_rodrigues_formula(axis: np.ndarray, angle: float) -> None:
    if np.linalg.norm(axis) < 1e-3:
        axis = np.array([1.0, 0.0, 0.0])
    e = axis / np.linalg.norm(axis)
    expected = (
        math.cos(angle) * np.eye(3)
        + (1 - math.cos(angle)) * np.outer(e, e)
        - math.sin(angle) * skew(e)
    )
    np.testing.assert_allclose(quat_to_dcm(quat_from_axis_angle(e, angle)), expected, atol=1e-12)


def test_quat_angle() -> None:
    q = quat_from_axis_angle([1, 2, 3], 0.3)
    assert float(quat_angle(q, IDENTITY_QUAT)) == pytest.approx(0.3)
    assert float(quat_angle(q, -q)) == pytest.approx(0.0, abs=1e-7)


def test_functions_broadcast_over_stacks() -> None:
    qs = np.stack([quat_from_axis_angle([0, 0, 1], a) for a in (0.1, 0.2, 0.3)])
    assert quat_to_dcm(qs).shape == (3, 3, 3)
    assert quat_mul(qs, qs).shape == (3, 4)
    assert quat_angle(qs, IDENTITY_QUAT) == pytest.approx([0.1, 0.2, 0.3])


def test_teme_ecef_round_trip() -> None:
    r = np.array([[6878e3, 1200e3, -300e3], [-1e6, 7e6, 2e5]])
    theta = np.array([0.3, 4.0])
    np.testing.assert_allclose(ecef_to_teme(teme_to_ecef(r, theta), theta), r, atol=1e-6)


def test_teme_to_ecef_matches_astropy(astropy_ref: list[dict[str, Any]]) -> None:
    # astropy also applies polar motion (~10 m), which this model neglects on purpose.
    for point in astropy_ref:
        t = datetime.fromisoformat(point["utc"]).replace(tzinfo=UTC)
        r_e = teme_to_ecef(point["r_teme_m"], gmst(*julian_date(t)))
        assert np.linalg.norm(r_e - np.array(point["r_itrs_m"])) < 20.0, point["utc"]


def test_ecef_velocity_removes_earth_rotation() -> None:
    # A point fixed on the rotating Earth has zero Earth-relative velocity.
    r = np.array([WGS84_A, 0.0, 0.0])
    v_inertial = np.array([0.0, 7.292_115_146_706_979e-5 * WGS84_A, 0.0])
    np.testing.assert_allclose(teme_to_ecef_velocity(r, v_inertial, 0.0), 0.0, atol=1e-9)


def test_geodetic_special_points() -> None:
    np.testing.assert_allclose(geodetic_to_ecef(0.0, 0.0, 0.0), [WGS84_A, 0, 0], atol=1e-9)
    b = WGS84_A * (1 - WGS84_F)
    np.testing.assert_allclose(geodetic_to_ecef(math.pi / 2, 0.0, 0.0), [0, 0, b], atol=1e-6)


@given(
    st.floats(-math.pi / 2, math.pi / 2),
    st.floats(-math.pi, math.pi),
    st.floats(-1e3, 2e6),
)
def test_geodetic_round_trip(lat: float, lon: float, alt: float) -> None:
    lat2, lon2, alt2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, alt))
    assert float(lat2) == pytest.approx(lat, abs=1e-10)
    if abs(lat) < math.pi / 2 - 1e-9:
        assert math.remainder(float(lon2) - lon, 2 * math.pi) == pytest.approx(0.0, abs=1e-10)
    assert float(alt2) == pytest.approx(alt, abs=1e-4)


def test_spherical_basis() -> None:
    r_e = np.array([3e6, -4e6, 5e6])
    _, colat, lon = ecef_to_spherical(r_e)
    radial = spherical_to_ecef_vector(1.0, 0.0, 0.0, colat, lon)
    np.testing.assert_allclose(radial, r_e / np.linalg.norm(r_e), atol=1e-12)
    # θ̂ points south, φ̂ points east; together with r̂ they form a right-handed basis.
    theta = spherical_to_ecef_vector(0.0, 1.0, 0.0, colat, lon)
    phi = spherical_to_ecef_vector(0.0, 0.0, 1.0, colat, lon)
    np.testing.assert_allclose(np.cross(radial, theta), phi, atol=1e-12)
    assert theta[2] < 0
