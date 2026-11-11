//! The scalar type and the maths functions, which come from `libm` because `core` has none.

#[cfg(not(feature = "f32"))]
mod imp {
    /// `adcs_real` of the C header.
    pub type Real = f64;
    pub(crate) const TINY: Real = 1e-300;
    pub(crate) fn sqrt(x: Real) -> Real {
        libm::sqrt(x)
    }
    pub(crate) fn sin(x: Real) -> Real {
        libm::sin(x)
    }
    pub(crate) fn cos(x: Real) -> Real {
        libm::cos(x)
    }
    /// Round half to even, like C `rint` and Python `round`.
    pub(crate) fn rint(x: Real) -> Real {
        libm::rint(x)
    }
    pub(crate) fn abs(x: Real) -> Real {
        libm::fabs(x)
    }
}

#[cfg(feature = "f32")]
mod imp {
    /// `adcs_real` of the C header, built with `ADCS_USE_FLOAT`.
    pub type Real = f32;
    pub(crate) const TINY: Real = 1e-30;
    pub(crate) fn sqrt(x: Real) -> Real {
        libm::sqrtf(x)
    }
    pub(crate) fn sin(x: Real) -> Real {
        libm::sinf(x)
    }
    pub(crate) fn cos(x: Real) -> Real {
        libm::cosf(x)
    }
    pub(crate) fn rint(x: Real) -> Real {
        libm::rintf(x)
    }
    pub(crate) fn abs(x: Real) -> Real {
        libm::fabsf(x)
    }
}

pub use imp::Real;
pub(crate) use imp::{TINY, abs, cos, rint, sin, sqrt};

/// Tolerances for the unit tests, which run in both precisions.
#[cfg(all(test, not(feature = "f32")))]
pub(crate) const TEST_TOL: Real = 1e-12;
#[cfg(all(test, feature = "f32"))]
pub(crate) const TEST_TOL: Real = 2e-4;
#[cfg(all(test, not(feature = "f32")))]
pub(crate) const TEST_NEWTON_TOL: Real = 1e-10;
#[cfg(all(test, feature = "f32"))]
pub(crate) const TEST_NEWTON_TOL: Real = 1e-5;
pub(crate) const NAN: Real = Real::NAN;
