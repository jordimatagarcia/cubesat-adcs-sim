#!/usr/bin/env bash
# Build the Rust library as a no_std static library for the host, link the C example
# adcs-c/examples/bare_metal.c against it unchanged, and run it: the exit code says whether the
# state fits the example's buffer and the ADCS initialises and runs (the Cortex-M builds in
# tools/cross_build.sh can only be linked, not run). Both precisions.
#
#   adcs-rs/tools/host_example.sh
set -euo pipefail
# (`cargo rustc --crate-type staticlib` builds only the static library: with the crate's other
# types, rlib included, Cargo skips LTO, which leaves an unresolved unwinding symbol in a
# no_std static library.)

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cc="${CC:-cc}"

for precision in double float; do
    features=""
    flags=""
    if [[ $precision == float ]]; then
        features="--features f32"
        flags="-DADCS_USE_FLOAT"
    fi
    # shellcheck disable=SC2086
    cargo rustc --quiet --lib --crate-type staticlib --release --manifest-path "$here/Cargo.toml" -p adcs-ffi \
        --no-default-features $features --target-dir "$work/target-$precision" \
        2> >(grep -v "dropping unsupported crate type" >&2 || true)
    # shellcheck disable=SC2086
    "$cc" -std=c11 -O2 -Wall -Wextra -Werror $flags -I"$repo/adcs-c/include" \
        "$repo/adcs-c/examples/bare_metal.c" "$work/target-$precision/release/libadcs_rs.a" \
        -lm -o "$work/bare_metal_$precision"
    if "$work/bare_metal_$precision"; then
        echo "bare_metal ($precision, Rust): ok"
    else
        echo "bare_metal ($precision, Rust): FAILED (state too large or ADCS did not run)" >&2
        exit 1
    fi
done
