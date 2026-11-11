#!/usr/bin/env bash
# Build every library configuration and run them all through the same benchmark driver
# (adcs-c/bench/bench.c). Writes one JSON line per configuration, variant and round to
# docs/results/data/bench_host.jsonl; tools/plot_bench.py turns it into the report tables
# and figure.
#
#   tools/bench.sh [reps] [rounds] [cpu]      (defaults 200, 3, 2)
#
# Fairness: the driver, its compiler (gcc -O2) and the inputs (all golden vectors) are the
# same for every library; libraries target the same baseline x86-64 CPU (no -march=native);
# the run is pinned to one performance core; configurations are interleaved over rounds so
# that thermal drift affects all of them alike.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
reps="${1:-200}"
rounds="${2:-3}"
cpu="${3:-2}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
out="$repo/docs/results/data/bench_host.jsonl"
mkdir -p "$(dirname "$out")"

c_lib() { # name compiler flags...   (an -flto build is archived with the matching ar)
    local name="$1" cc="$2" ar=ar
    shift 2
    [[ " $* " == *" -flto "* ]] && { [[ "$cc" == gcc ]] && ar=gcc-ar || ar=llvm-ar; }
    mkdir -p "$work/$name"
    for f in "$repo"/adcs-c/src/*.c; do
        "$cc" -std=c11 "$@" -I"$repo/adcs-c/include" -c "$f" -o "$work/$name/$(basename "$f" .c).o"
    done
    "$ar" rcs "$work/$name/lib.a" "$work/$name"/*.o
    echo "$cc" >"$work/$name/linker"
}

rust_lib() { # name opt-level [features]
    local name="$1" opt="$2" features="${3:-}"
    # shellcheck disable=SC2086
    CARGO_PROFILE_RELEASE_OPT_LEVEL="$opt" cargo rustc --quiet --lib --crate-type staticlib --release \
        --manifest-path "$repo/adcs-rs/Cargo.toml" -p adcs-ffi --no-default-features $features \
        --target-dir "$work/target-$name" 2> >(grep -v "dropping unsupported crate type" >&2)
    mkdir -p "$work/$name"
    cp "$work/target-$name/release/libadcs_rs.a" "$work/$name/lib.a"
    echo gcc >"$work/$name/linker"
}

echo "building libraries..." >&2
c_lib c-gcc-O2 gcc -O2
c_lib c-gcc-O3 gcc -O3
c_lib c-gcc-O3-lto gcc -O3 -flto
c_lib c-clang-O2 clang -O2
c_lib c-clang-O3 clang -O3
c_lib c-clang-O2-lto clang -O2 -flto
c_lib c-clang-O3-lto clang -O3 -flto
c_lib c-gcc-O3-lto-f32 gcc -O3 -flto -DADCS_USE_FLOAT
c_lib c-clang-O3-lto-f32 clang -O3 -flto -DADCS_USE_FLOAT
rust_lib rust-O2 2
rust_lib rust-O3 3
rust_lib rust-O3-f32 3 "--features f32"

# The driver is compiled once per precision, by gcc and without LTO, so it is the same code
# for every library and nothing can be inlined across it; LTO builds only optimise across
# their own modules (like the Rust library, which is built with lto = true).
for prec in double float; do
    flags=""
    [[ $prec == float ]] && flags="-DADCS_USE_FLOAT"
    # shellcheck disable=SC2086
    gcc -std=c11 -O2 $flags -I"$repo/adcs-c/include" -I"$repo/adcs-c/tests" \
        -c "$repo/adcs-c/bench/bench.c" -o "$work/bench-$prec.o"
    # shellcheck disable=SC2086
    gcc -std=c11 -O2 $flags -I"$repo/adcs-c/include" -I"$repo/adcs-c/tests" \
        -c "$repo/adcs-c/tests/golden_io.c" -o "$work/golden_io-$prec.o"
done

configs=(c-gcc-O2 c-gcc-O3 c-gcc-O3-lto c-clang-O2 c-clang-O3 c-clang-O2-lto c-clang-O3-lto
    rust-O2 rust-O3 c-gcc-O3-lto-f32 c-clang-O3-lto-f32 rust-O3-f32)
for name in "${configs[@]}"; do
    prec=double
    [[ "$name" == *-f32 ]] && prec=float
    lto=""
    # link an LTO build at its own level (LTO optimises again at link time)
    if [[ "$name" == *-lto* ]]; then
        lto="-flto -O3"
        [[ "$name" == *-O2-* ]] && lto="-flto -O2"
    fi
    # shellcheck disable=SC2086
    "$(cat "$work/$name/linker")" $lto "$work/bench-$prec.o" "$work/golden_io-$prec.o" \
        "$work/$name/lib.a" -lm -lpthread -o "$work/$name/bench"
done

{
    echo "{\"meta\": true, \"cpu\": \"$(lscpu | sed -n 's/^Model name: *//p')\", \"pinned_cpu\": $cpu, \"governor\": \"$(cat /sys/devices/system/cpu/cpu"$cpu"/cpufreq/scaling_governor 2>/dev/null || echo unknown)\", \"gcc\": \"$(gcc -dumpfullversion)\", \"clang\": \"$(clang -dumpversion)\", \"rustc\": \"$(rustc -V | cut -d' ' -f2)\", \"reps\": $reps, \"rounds\": $rounds}"
    for round in $(seq 1 "$rounds"); do
        for name in "${configs[@]}"; do
            echo "round $round: $name" >&2
            taskset -c "$cpu" "$work/$name/bench" --label "$name" --reps "$reps" \
                "$repo"/testdata/golden/*/*.json | sed "s/^{/{\"round\": $round, /"
        done
    done
} >"$out"
echo "wrote $out" >&2
