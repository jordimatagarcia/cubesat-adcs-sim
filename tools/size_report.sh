#!/usr/bin/env bash
# Code size of the bare-metal example (adcs-c/examples/bare_metal.c) linked against the C
# and the Rust library, for Cortex-M4F (float) and Cortex-M7 (double), at matching
# optimisation levels. Writes docs/results/data/sizes.json (.text bytes) and prints the
# number of bounds-check panic sites in each Rust image.
#
#   tools/size_report.sh
#
# Needs arm-none-eabi-gcc with newlib and the thumbv7em-none-eabihf Rust target;
# EXTRA_CFLAGS / EXTRA_LDFLAGS are passed through to the C compiler.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
targets=(cortex-m4f-float cortex-m7-double)

text() { arm-none-eabi-size "$1" | awk 'NR == 2 { print $1 }'; }

declare -A size
for opt in "-O2" "-Os" "-O2 -flto" "-Os -flto"; do
    dir="$work/c${opt// /}"
    OPT="$opt" STACK_LIMIT=16384 "$repo/adcs-c/tools/cross_build.sh" "$dir" >/dev/null 2>&1
    for t in "${targets[@]}"; do
        size["C $opt|$t"]=$(text "$dir/$t/bare_metal.elf")
    done
done
for level in 2 s z; do
    dir="$work/rust$level"
    CARGO_PROFILE_FIRMWARE_OPT_LEVEL="$level" "$repo/adcs-rs/tools/cross_build.sh" "$dir" >/dev/null 2>&1
    for t in "${targets[@]}"; do
        size["Rust opt-level $level|$t"]=$(text "$dir/$t.elf")
        checks=$(arm-none-eabi-objdump -d "$dir/$t.elf" | grep -c "bl.*panic_bounds_check" || true)
        echo "Rust opt-level $level, $t: $checks calls to panic_bounds_check" >&2
    done
done

{
    echo "{"
    first=1
    for build in "C -O2" "C -Os" "C -O2 -flto" "C -Os -flto" "Rust opt-level 2" "Rust opt-level s" "Rust opt-level z"; do
        [[ $first == 1 ]] || echo ","
        first=0
        printf '  "%s": {"cortex-m4f-float": %s, "cortex-m7-double": %s}' "$build" \
            "${size["$build|cortex-m4f-float"]}" "${size["$build|cortex-m7-double"]}"
    done
    echo
    echo "}"
} >"$repo/docs/results/data/sizes.json"
cat "$repo/docs/results/data/sizes.json"
