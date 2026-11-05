//! C ABI of the Rust ADCS library. Exports the same symbols as the C library
//! (`adcs-c/include/adcs.h`), so the simulator can load either one.
//!
//! The algorithms live in `adcs-core`, which is `no_std`. This shim uses `std` for host
//! builds; an embedded build with its own panic handler comes with the Cortex-M target.

/// Returns the interface version compiled into the library (`ADCS_ABI_VERSION`).
#[unsafe(no_mangle)]
pub extern "C" fn adcs_abi_version() -> u32 {
    adcs_core::ABI_VERSION
}

#[cfg(test)]
mod tests {
    #[test]
    fn exports_core_abi_version() {
        assert_eq!(super::adcs_abi_version(), adcs_core::ABI_VERSION);
    }
}
