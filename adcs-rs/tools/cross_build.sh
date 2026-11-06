#!/usr/bin/env bash
# Build the Rust library as a no_std static library for two ARM Cortex-M targets and link
# the C example adcs-c/examples/bare_metal.c against it: the Rust library is a drop-in
# replacement for the C one in firmware too. Reports code size and whether malloc was
# linked in. Built with the size-optimised `firmware` profile (opt-level "s").
#
#   adcs-rs/tools/cross_build.sh [out_dir]
#
# Needs the thumbv7em-none-eabihf Rust target and arm-none-eabi-gcc with newlib.
# EXTRA_CFLAGS / EXTRA_LDFLAGS are passed to the C compiler (e.g. for a relocated toolchain).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
out="${1:-$here/target/bare-metal}"
mkdir -p "$out"

targets=(
    "cortex-m4f-float|cortex-m4|--features f32|-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DADCS_USE_FLOAT"
    "cortex-m7-double|cortex-m7||-mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16"
)
status=0
for entry in "${targets[@]}"; do
    IFS='|' read -r name cpu features cflags <<<"$entry"
    target_dir="$out/cargo-$name"
    # shellcheck disable=SC2086
    RUSTFLAGS="-C target-cpu=$cpu" cargo build --quiet --manifest-path "$here/Cargo.toml" \
        -p adcs-ffi --profile firmware --target thumbv7em-none-eabihf --no-default-features $features \
        --target-dir "$target_dir"
    lib="$target_dir/thumbv7em-none-eabihf/firmware/libadcs_rs.a"
    elf="$out/$name.elf"
    # shellcheck disable=SC2086
    arm-none-eabi-gcc -std=c11 $cflags -O2 -ffunction-sections -fdata-sections \
        -Wall -Wextra -Werror ${EXTRA_CFLAGS:-} -I"$repo/adcs-c/include" \
        "$repo/adcs-c/examples/bare_metal.c" "$lib" \
        --specs=nosys.specs -Wl,--gc-sections ${EXTRA_LDFLAGS:-} -lm -o "$elf"
    echo "== $name (Rust)"
    arm-none-eabi-size "$elf"
    if arm-none-eabi-nm "$elf" | grep -qE ' (malloc|_malloc_r)$'; then
        echo "error: malloc is linked in" >&2
        status=1
    fi
done
exit $status
