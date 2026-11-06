//! ADCS algorithms for a CubeSat, written for `no_std` targets: a port of the C library in
//! `adcs-c/`, with the same interface (`adcs.h`, mirrored in [`abi`]) and the same
//! algorithms, operation order included, so that both reproduce the Python reference's
//! golden vectors.
//!
//! The C-compatible functions live in the `adcs-ffi` crate; this crate has no `unsafe` code.
//!
//! ```
//! use adcs_core::{Adcs, abi::Input};
//! let mut adcs = Adcs::new();
//! assert!(adcs.step(&Input::default()).is_err()); // not initialised yet
//! ```
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
// Index loops and `0.5 * (a + b)` mirror the C code line by line, which keeps the two easy
// to compare and rounding identically. `!(x > 0.0)` is deliberate: it is true for NaN.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::manual_midpoint,
    clippy::neg_cmp_op_on_partial_ord
)]

pub mod abi;
mod adcs;
pub mod control;
pub mod flae;
pub mod linalg;
pub mod mekf;
pub mod quat;
pub mod real;
pub mod tsukf;

pub use adcs::{Adcs, InvalidConfig, NotInitialised};
pub use real::Real;

/// Version of the C interface defined in `adcs-c/include/adcs.h` (`ADCS_ABI_VERSION`).
pub const ABI_VERSION: u32 = 3;

#[cfg(test)]
mod tests {
    use super::ABI_VERSION;

    #[test]
    fn abi_version_matches_c_header() {
        let header = include_str!("../../../adcs-c/include/adcs.h");
        let expected = format!("#define ADCS_ABI_VERSION {ABI_VERSION}u");
        assert!(
            header.contains(&expected),
            "adcs.h does not define {expected}"
        );
    }
}
