//! ADCS algorithms for a CubeSat, written for `no_std` targets.
//!
//! The C-compatible interface lives in the `adcs-ffi` crate; this crate holds the algorithms
//! and has no `unsafe` code.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

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
