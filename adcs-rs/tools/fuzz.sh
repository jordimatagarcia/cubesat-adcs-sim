#!/usr/bin/env bash
# Coverage-guided fuzzing of Adcs::step with libFuzzer on the stable toolchain.
#
#   adcs-rs/tools/fuzz.sh [seconds]      (default 120)
#
# cargo-fuzz needs nightly only for its sanitizer flags; the core crate forbids unsafe code,
# so the coverage instrumentation (stable -C flags) plus debug assertions and overflow
# checks is what matters. The target is fuzz/fuzz_targets/step.rs; crashes are written to
# the working directory as crash-*.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
seconds="${1:-120}"
host="$(rustc -vV | sed -n 's/^host: //p')"
export RUSTFLAGS="-C passes=sancov-module \
    -C llvm-args=-sanitizer-coverage-level=4 \
    -C llvm-args=-sanitizer-coverage-inline-8bit-counters \
    -C llvm-args=-sanitizer-coverage-pc-table \
    -C llvm-args=-sanitizer-coverage-trace-compares \
    -C debug-assertions -C overflow-checks"
# An explicit --target keeps the instrumentation out of build scripts.
cargo build --quiet --release --manifest-path "$here/fuzz/Cargo.toml" --bin step --target "$host"
mkdir -p corpus
"$here/fuzz/target/$host/release/step" -max_total_time="$seconds" -max_len=900 corpus
