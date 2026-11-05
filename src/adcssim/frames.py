"""Reference frames, rotations and quaternion algebra.

Conventions (see ``docs/conventions.md``):

- Quaternions are scalar-last, ``q = [q1, q2, q3, q4]``.
- ``q`` maps inertial-frame coordinates to body-frame coordinates: ``v_b = A(q) @ v_i``.
- The product ``⊗`` follows Markley & Crassidis (2014, eq. 2.82a), so that
  ``A(q ⊗ p) = A(q) @ A(p)``.

Functions accept a single vector/quaternion or a stack of them (leading dimensions are
broadcast).
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

EARTH_ROTATION_RATE = 7.292_115_146_706_979e-5
"""Earth rotation rate in rad/s (Vallado, used with GMST for TEME → Earth-fixed)."""

WGS84_A = 6_378_137.0
"""WGS-84 equatorial radius in metres."""
WGS84_F = 1.0 / 298.257_223_563
"""WGS-84 flattening."""
_WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _f64(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------------------


def skew(v: ArrayLike) -> FloatArray:
    """Cross-product matrix ``[v×]`` such that ``skew(v) @ u == cross(v, u)``."""
    x, y, z = _f64(v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def cross3(a: FloatArray, b: FloatArray) -> FloatArray:
    """Cross product of two single 3-vectors.

    Same result as ``np.cross`` for shape ``(3,)``, but about 20 times faster, which matters in
    the dynamics loop where it runs several times per RK4 stage.
    """
    a1, a2, a3 = float(a[0]), float(a[1]), float(a[2])
    b1, b2, b3 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])


def unit(v: ArrayLike) -> FloatArray:
    """Vector(s) scaled to unit length along the last axis."""
    a = _f64(v)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# --------------------------------------------------------------------------------------
# Quaternions (scalar-last)
# --------------------------------------------------------------------------------------

IDENTITY_QUAT: FloatArray = np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize(q: ArrayLike) -> FloatArray:
    return unit(q)


def quat_conj(q: ArrayLike) -> FloatArray:
    """Conjugate, equal to the inverse for unit quaternions."""
    a = _f64(q)
    return np.concatenate([-a[..., :3], a[..., 3:]], axis=-1)


def quat_mul(q: ArrayLike, p: ArrayLike) -> FloatArray:
    """Quaternion product ``q ⊗ p`` with ``A(q ⊗ p) = A(q) @ A(p)``."""
    qa, pa = _f64(q), _f64(p)
    qv, q4 = qa[..., :3], qa[..., 3:]
    pv, p4 = pa[..., :3], pa[..., 3:]
    v = q4 * pv + p4 * qv - np.cross(qv, pv)
    s = q4 * p4 - np.sum(qv * pv, axis=-1, keepdims=True)
    return np.concatenate([v, s], axis=-1)


def quat_to_dcm(q: ArrayLike) -> FloatArray:
    """Attitude matrix ``A(q)`` (inertial → body), Markley & Crassidis eq. 2.125."""
    a = _f64(q)
    q1, q2, q3, q4 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    m = np.empty((*a.shape[:-1], 3, 3))
    m[..., 0, 0] = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4
    m[..., 0, 1] = 2.0 * (q1 * q2 + q3 * q4)
    m[..., 0, 2] = 2.0 * (q1 * q3 - q2 * q4)
    m[..., 1, 0] = 2.0 * (q1 * q2 - q3 * q4)
    m[..., 1, 1] = -q1 * q1 + q2 * q2 - q3 * q3 + q4 * q4
    m[..., 1, 2] = 2.0 * (q2 * q3 + q1 * q4)
    m[..., 2, 0] = 2.0 * (q1 * q3 + q2 * q4)
    m[..., 2, 1] = 2.0 * (q2 * q3 - q1 * q4)
    m[..., 2, 2] = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4
    return m


def dcm_to_quat(m: ArrayLike) -> FloatArray:
    """Quaternion of an attitude matrix (Shepperd's method), with ``q4 >= 0``."""
    a = _f64(m)
    if a.shape != (3, 3):
        raise ValueError("dcm_to_quat expects a single 3x3 matrix")
    tr = a[0, 0] + a[1, 1] + a[2, 2]
    # Pick the largest of 4*qi^2 to avoid dividing by a small number.
    candidates = [a[0, 0], a[1, 1], a[2, 2], tr]
    i = int(np.argmax(candidates))
    if i == 3:
        q4 = 0.5 * math.sqrt(1.0 + tr)
        f = 0.25 / q4
        q = [(a[1, 2] - a[2, 1]) * f, (a[2, 0] - a[0, 2]) * f, (a[0, 1] - a[1, 0]) * f, q4]
    elif i == 0:
        q1 = 0.5 * math.sqrt(1.0 + 2.0 * a[0, 0] - tr)
        f = 0.25 / q1
        q = [q1, (a[0, 1] + a[1, 0]) * f, (a[0, 2] + a[2, 0]) * f, (a[1, 2] - a[2, 1]) * f]
    elif i == 1:
        q2 = 0.5 * math.sqrt(1.0 + 2.0 * a[1, 1] - tr)
        f = 0.25 / q2
        q = [(a[0, 1] + a[1, 0]) * f, q2, (a[1, 2] + a[2, 1]) * f, (a[2, 0] - a[0, 2]) * f]
    else:
        q3 = 0.5 * math.sqrt(1.0 + 2.0 * a[2, 2] - tr)
        f = 0.25 / q3
        q = [(a[0, 2] + a[2, 0]) * f, (a[1, 2] + a[2, 1]) * f, q3, (a[0, 1] - a[1, 0]) * f]
    return quat_positive(np.array(q))


def quat_positive(q: ArrayLike) -> FloatArray:
    """The same attitude with the scalar part ``q4 >= 0``."""
    a = _f64(q)
    return np.where(a[..., 3:] < 0.0, -a, a)


def quat_from_axis_angle(axis: ArrayLike, angle: float) -> FloatArray:
    """Quaternion of a frame rotation by ``angle`` (rad) about ``axis``.

    ``A`` of the result is ``cos(θ) I + (1 - cos θ) e eᵀ - sin(θ) [e×]``.
    """
    e = unit(axis)
    return np.concatenate([e * math.sin(angle / 2.0), [math.cos(angle / 2.0)]])


def quat_rotate(q: ArrayLike, v: ArrayLike) -> FloatArray:
    """``A(q) @ v``: inertial-frame vector(s) expressed in the body frame."""
    return _f64(np.einsum("...ij,...j->...i", quat_to_dcm(q), _f64(v)))


def quat_xi(q: ArrayLike) -> FloatArray:
    """4×3 matrix ``Ξ(q)`` of the kinematics ``q̇ = ½ Ξ(q) ω`` (Markley & Crassidis eq. 3.21)."""
    q1, q2, q3, q4 = _f64(q)
    return np.array(
        [
            [q4, -q3, q2],
            [q3, q4, -q1],
            [-q2, q1, q4],
            [-q1, -q2, -q3],
        ]
    )


def quat_angle(q_a: ArrayLike, q_b: ArrayLike) -> FloatArray:
    """Rotation angle (rad, in [0, π]) between two attitudes."""
    dq = quat_mul(q_a, quat_conj(q_b))
    s = np.clip(np.abs(dq[..., 3]), 0.0, 1.0)
    return np.asarray(2.0 * np.arccos(s), dtype=np.float64)


# --------------------------------------------------------------------------------------
# TEME (inertial) ↔ Earth-fixed
# --------------------------------------------------------------------------------------


def _r3(theta: ArrayLike) -> FloatArray:
    """Frame rotation about z by ``theta``: ``R3(θ) = [[c, s, 0], [-s, c, 0], [0, 0, 1]]``."""
    th = _f64(theta)
    c, s = np.cos(th), np.sin(th)
    m = np.zeros((*th.shape, 3, 3))
    m[..., 0, 0] = c
    m[..., 0, 1] = s
    m[..., 1, 0] = -s
    m[..., 1, 1] = c
    m[..., 2, 2] = 1.0
    return m


def teme_to_ecef(r_teme: ArrayLike, gmst: ArrayLike) -> FloatArray:
    """Rotate TEME position(s) into the Earth-fixed frame. Polar motion is neglected."""
    return _f64(np.einsum("...ij,...j->...i", _r3(gmst), _f64(r_teme)))


def ecef_to_teme(r_ecef: ArrayLike, gmst: ArrayLike) -> FloatArray:
    """Rotate Earth-fixed vector(s) into TEME (inverse of :func:`teme_to_ecef`)."""
    return _f64(np.einsum("...ji,...j->...i", _r3(gmst), _f64(r_ecef)))


def teme_to_ecef_velocity(r_teme: ArrayLike, v_teme: ArrayLike, gmst: ArrayLike) -> FloatArray:
    """Velocity relative to the rotating Earth, in Earth-fixed coordinates."""
    r_e = teme_to_ecef(r_teme, gmst)
    v_e = teme_to_ecef(v_teme, gmst)
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    return np.asarray(v_e - np.cross(omega, r_e), dtype=np.float64)


# --------------------------------------------------------------------------------------
# Earth-fixed ↔ geodetic / geocentric spherical
# --------------------------------------------------------------------------------------


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike) -> FloatArray:
    """WGS-84 geodetic latitude/longitude (rad) and altitude (m) → Earth-fixed position (m)."""
    la, lo, h = _f64(lat), _f64(lon), _f64(alt)
    n = WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(la) ** 2)
    x = (n + h) * np.cos(la) * np.cos(lo)
    y = (n + h) * np.cos(la) * np.sin(lo)
    z = (n * (1.0 - _WGS84_E2) + h) * np.sin(la)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(r_ecef: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Earth-fixed position (m) → WGS-84 geodetic latitude, longitude (rad), altitude (m).

    Iterative solution; converges to well below a millimetre in a few iterations for any
    point outside the Earth's core.
    """
    r = _f64(r_ecef)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + _WGS84_E2 * n * np.sin(lat), p)
    # h = p cos φ + z sin φ − a √(1 − e² sin² φ): well conditioned at every latitude.
    alt = p * np.cos(lat) + z * np.sin(lat) - WGS84_A * np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    return lat, lon, np.asarray(alt, dtype=np.float64)


def ecef_to_spherical(r_ecef: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Earth-fixed position → geocentric radius (m), colatitude and longitude (rad)."""
    r = _f64(r_ecef)
    rad = np.linalg.norm(r, axis=-1)
    colat = np.arccos(np.clip(r[..., 2] / rad, -1.0, 1.0))
    lon = np.arctan2(r[..., 1], r[..., 0])
    return rad, colat, lon


def spherical_to_ecef_vector(
    v_r: ArrayLike, v_theta: ArrayLike, v_phi: ArrayLike, colat: ArrayLike, lon: ArrayLike
) -> FloatArray:
    """Vector given by its local spherical components (r, θ, φ) → Earth-fixed components."""
    vr, vt, vp = _f64(v_r), _f64(v_theta), _f64(v_phi)
    st, ct = np.sin(colat), np.cos(colat)
    sp, cp = np.sin(lon), np.cos(lon)
    x = vr * st * cp + vt * ct * cp - vp * sp
    y = vr * st * sp + vt * ct * sp + vp * cp
    z = vr * ct - vt * st
    return np.stack([x, y, z], axis=-1)
