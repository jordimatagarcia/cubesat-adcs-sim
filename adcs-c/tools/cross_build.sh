#!/usr/bin/env bash
# Cross-compile the library and examples/bare_metal.c for two ARM Cortex-M targets and
# report code size, stack usage per function and whether malloc was linked in.
#
#   adcs-c/tools/cross_build.sh [out_dir]
#
# Needs arm-none-eabi-gcc with newlib (Ubuntu: gcc-arm-none-eabi libnewlib-arm-none-eabi).
# OPT sets the optimisation flags (default -O2), STACK_LIMIT the per-function stack limit
# in bytes (default 2048; an LTO build inlines everything into main). EXTRA_CFLAGS / EXTRA_LDFLAGS are passed
# through (e.g. paths for a relocated toolchain).
# The linker warns that newlib's nosys stubs (_close, _read, ...) always fail: the library
# never calls them, they are only referenced from parts of libc it does not use.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$here/build-arm}"
mkdir -p "$out"

targets=(
    "cortex-m4f-float|-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DADCS_USE_FLOAT"
    "cortex-m7-double|-mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16"
)
status=0
for entry in "${targets[@]}"; do
    name="${entry%%|*}"
    flags="${entry#*|}"
    dir="$out/$name"
    rm -rf "$dir" && mkdir -p "$dir"
    # shellcheck disable=SC2086
    (cd "$dir" && arm-none-eabi-gcc -std=c11 $flags ${OPT:--O2} -ffunction-sections -fdata-sections \
        -fstack-usage -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion \
        -Wdouble-promotion -Wshadow -Wvla -Wstack-usage=${STACK_LIMIT:-2048} ${EXTRA_CFLAGS:-} \
        -I"$here/include" "$here"/src/*.c "$here/examples/bare_metal.c" \
        --specs=nosys.specs -Wl,--gc-sections ${EXTRA_LDFLAGS:-} -lm -o bare_metal.elf)
    echo "== $name"
    arm-none-eabi-size "$dir/bare_metal.elf"
    echo "largest stack frames:"
    sort -t$'\t' -k2 -n "$dir"/*.su | tail -4 | sed 's/^/  /'
    # (nm to a file first: with pipefail, `nm | grep -q` can report nm's SIGPIPE as "no match")
    arm-none-eabi-nm "$dir/bare_metal.elf" >"$dir/bare_metal.elf.nm"
    if grep -qE ' (malloc|_malloc_r)$' "$dir/bare_metal.elf.nm"; then
        echo "error: malloc is linked in" >&2
        status=1
    fi
done
exit $status
