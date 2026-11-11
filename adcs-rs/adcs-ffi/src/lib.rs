//! C ABI of the Rust ADCS library. Exports the same four functions as the C library
//! (`adcs-c/include/adcs.h`), so the simulator, or any C program, can link either one.
//!
//! This is the only crate with `unsafe` code: turning the caller's pointers into Rust
//! references. Every `unsafe` block states the conditions it relies on; they are the
//! preconditions documented in `adcs.h`.
//!
//! Built without the default `std` feature it is `no_std`, for linking into firmware
//! (`thumbv7em-none-eabihf`); a panic then halts in a loop. Nothing in the library is
//! expected to panic: there is no allocation, and indices are fixed by the array sizes.
#![cfg_attr(not(feature = "std"), no_std)]

use core::ffi::{c_int, c_void};
use core::mem::{align_of, size_of};

use adcs_core::Adcs;
use adcs_core::abi::{Config, Input, Output};

/// Returns the interface version compiled into the library (`ADCS_ABI_VERSION`).
#[unsafe(no_mangle)]
pub extern "C" fn adcs_abi_version() -> u32 {
    adcs_core::ABI_VERSION
}

/// `sizeof(adcs_real)`: 8, or 4 with the `f32` feature. A caller checks it against its own
/// header, since the ABI version does not include the precision.
#[unsafe(no_mangle)]
pub extern "C" fn adcs_real_size() -> usize {
    size_of::<adcs_core::Real>()
}

/// Size in bytes of the state the caller must provide.
#[unsafe(no_mangle)]
pub extern "C" fn adcs_state_size() -> usize {
    size_of::<Adcs>()
}

fn aligned(state: *mut c_void) -> bool {
    (state as usize).is_multiple_of(align_of::<Adcs>())
}

/// Initialises the state from a configuration. Returns 0 on success, -1 for a null pointer,
/// -2 for misaligned state memory and -3 for an invalid configuration.
///
/// # Safety
/// `state` must point to `adcs_state_size()` writable bytes, aligned for any type (as
/// `malloc` returns), that nothing else accesses during the call. `config` must point to a
/// valid `adcs_config`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adcs_init(state: *mut c_void, config: *const Config) -> c_int {
    if state.is_null() {
        return -1;
    }
    if !aligned(state) {
        return -2;
    }
    let state = state.cast::<Adcs>();
    // SAFETY: `state` is non-null, aligned (checked above) and points to enough writable
    // memory (caller contract). `write` does not read or drop the previous contents, which
    // may be uninitialised. Writing before the config check means a later adcs_step on a
    // failed init reads an initialised, unmarked state and fails cleanly.
    unsafe { state.write(Adcs::new()) };
    if config.is_null() {
        return -1;
    }
    // SAFETY: the memory now holds a valid `Adcs`, and the caller guarantees exclusive
    // access for the duration of the call. `config` is non-null and points to a valid
    // `adcs_config` (caller contract); `Config` has the same layout and every bit pattern
    // of its fields is valid (no `bool`: the flag is a `u8`).
    let (adcs, config) = unsafe { (&mut *state, &*config) };
    match adcs.init(config) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Processes one control period. Returns 0 on success, -1 on a programming error (null
/// pointer, misaligned or not initialised state), in which case `out` (if not null) reports
/// `ADCS_STATUS_NOT_INITIALISED`.
///
/// # Safety
/// `state` must be memory previously passed to `adcs_init` (or at least
/// `adcs_state_size()` initialised bytes, e.g. zeroed, in which case the call fails
/// cleanly), not accessed by anything else during the call. `input` must point to a valid
/// `adcs_input` and `out` to writable memory for an `adcs_output`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn adcs_step(
    state: *mut c_void,
    input: *const Input,
    out: *mut Output,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    // SAFETY (for the read): non-null and aligned (checked first); the magic number is the
    // first field of the `repr(C)` `Adcs`, and the caller provides at least
    // `adcs_state_size()` initialised bytes (memory given to `adcs_init`, or e.g. zeroed). It
    // is read as a plain `u32` before any reference to an `Adcs` exists.
    let result = if state.is_null()
        || input.is_null()
        || !aligned(state)
        || unsafe { state.cast::<u32>().read() } != adcs_core::MAGIC
    {
        Err(())
    } else {
        // SAFETY: the magic number is only ever written by `adcs_init` after it has written a
        // complete, valid `Adcs` (caller contract: nothing else writes this memory), and the
        // caller guarantees exclusive access. `input` is non-null and points to a valid
        // `adcs_input`, whose fields accept every bit pattern.
        let (adcs, input) = unsafe { (&mut *state.cast::<Adcs>(), &*input) };
        adcs.step(input).map_err(|_| ())
    };
    let (output, rc) = match result {
        Ok(o) => (o, 0),
        Err(()) => (Adcs::not_initialised_output(), -1),
    };
    // SAFETY: `out` is non-null and writable for an `adcs_output` (caller contract).
    unsafe { out.write(output) };
    rc
}

/// No panic is expected (no allocation, indices fixed by array sizes). If one happened on a
/// microcontroller, trap: the fault handler or watchdog resets the board, instead of a
/// silent loop that leaves the coils on their last command.
#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    #[cfg(target_arch = "arm")]
    // SAFETY: `udf` raises an undefined-instruction exception; it never returns.
    unsafe {
        core::arch::asm!("udf #0", options(noreturn));
    }
    #[cfg(not(target_arch = "arm"))]
    loop {
        core::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests;
