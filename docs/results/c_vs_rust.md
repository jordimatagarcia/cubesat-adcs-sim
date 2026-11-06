# C vs Rust: the same ADCS in both languages

Both libraries implement `adcs.h` with the same algorithms, the same operation order and the
same hand-written linear algebra (docs/adcs.md), so the differences measured here come from
the languages, compilers and build settings, not from the code's design.

## How it was measured

- **One driver for every library.** `adcs-c/bench/bench.c` is compiled once (gcc -O2, no
  LTO) and linked against each library through `adcs.h`, so the timing code, the inputs and
  the call boundary are identical. Inputs: all 16 golden vectors (8 per variant, 400 steps
  each), replayed 200 times after 20 warm-up passes, in 3 rounds with the configurations
  interleaved. Reported: median over rounds of the mean and percentiles, maximum of the
  maxima.
- **Machine:** Intel Core i7-13700H, run pinned to one performance core (`taskset -c 2`),
  Linux `powersave` governor (changing it needs root), gcc 13.3, clang 18.1, rustc 1.99.1.
  All libraries target the baseline x86-64 CPU (no `-march=native`).
- **Builds:** C with gcc and clang at `-O2` and `-O3`, each with and without link-time
  optimisation (LTO); Rust at `opt-level` 2 and 3, always with `lto = true` and
  `codegen-units = 1` (the workspace's release profile). The Rust library is built
  `no_std`, as for firmware.
- **Code size:** the bare-metal C example linked against each library for a Cortex-M4F
  (single precision) and a Cortex-M7 (double precision), `.text` bytes from
  `arm-none-eabi-size`.

Reproduce: `tools/bench.sh`, `tools/size_report.sh`, then `uv run python tools/plot_bench.py`
(raw data in `docs/results/data/`).

## Speed

![Time per step by build](../images/c_vs_rust_timing.png)

Time per `adcs_step` in µs:

| Build | Thesis mean | p99 | max | MEKF mean | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| C gcc -O2 | 20.81 | 27.64 | 138.7 | 2.71 | 3.30 | 36.1 |
| C gcc -O3 | 17.61 | 22.20 | 190.4 | 2.25 | 3.10 | 38.0 |
| C gcc -O3 LTO | **11.32** | **13.60** | 60.7 | **1.19** | **1.33** | 57.2 |
| C clang -O2 | 16.74 | 21.70 | 75.2 | 1.89 | 2.26 | 52.1 |
| C clang -O3 | 16.32 | 21.42 | 76.9 | 1.90 | 2.24 | 28.2 |
| C clang -O2 LTO | 14.35 | 18.24 | 113.5 | 1.37 | 1.52 | 119.4 |
| C clang -O3 LTO | 14.29 | 18.25 | 70.4 | 1.37 | 1.52 | 29.8 |
| Rust opt-level 2 | 18.53 | 24.03 | 85.4 | 1.41 | 1.63 | 63.6 |
| Rust opt-level 3 | 13.76 | 18.56 | 99.4 | 1.21 | 1.38 | 24.6 |

- **Build settings matter more than the language.** C compiled file by file is 1.1–2.3×
  slower than the same C with LTO, because the small linear-algebra routines cannot be
  inlined across files. A first run without C LTO showed Rust ahead by up to 2× on the
  MEKF; the gap was LTO, which the Rust release profile enables and a plain C build
  does not.
- **Same backend, same speed.** Rust and clang both compile through LLVM. With LTO on both,
  Rust is 4% faster on the thesis variant and 12% faster on the MEKF: the bounds checks
  Rust adds cost nothing measurable here.
- **gcc with LTO is fastest** on the thesis variant (18% faster than Rust) and ties with
  Rust on the MEKF.
- **The maxima are the operating system, not the code**: occasional interrupts and context
  switches of 25–190 µs, in every build. p99 is the meaningful tail on a desktop;
  worst-case execution time would need the code on a microcontroller.
- For scale, at 10 Hz the slowest build uses 0.02% of the control period on this machine.

## Accuracy and bit-exactness

All 9 double-precision builds (both languages, both compilers, every optimisation level)
produce **bit-identical outputs** on all 16 golden vectors (same FNV-1a hash of every output
byte), and differ from the Python reference by at most 6·10⁻¹² ° (thesis) and 1·10⁻⁹ °
(MEKF). The 3 single-precision builds are bit-identical to each other too. Same operation
order and no floating-point contraction (C11 in strict mode, Rust by default) make the
result independent of the language and the optimiser on this target.

Over full closed-loop scenarios (`tools/crosscheck.py`) Rust and C stay bit-identical in 8 of
10 thesis-variant runs and within 2·10⁻¹¹ elsewhere, the residue coming from the maths
library (`libm` crate vs glibc) in sine and cosine.

### Single precision

Largest attitude difference from the double-precision reference, float builds (the same
for C and Rust):

| Vector | Thesis (°) | MEKF (°) |
|---|---:|---:|
| nominal | 0.023 | 0.0013 |
| bad input | 0.023 | 0.0013 |
| gyro bias | 0.50 | 0.40 |
| magnetometer bias | **42** | 0.0049 |
| eclipse entry | 0.64 | 0.0002 |
| start in eclipse | 6.3 | 0.0027 |
| high noise | 3.8 | 0.0002 |
| control | 0.39 | 0.0001 |

**Correction (after a later code review):** the thesis-filter numbers above come from a
porting bug, not from single precision itself. Before every Cholesky factorisation the
filter adds machine epsilon to the covariance diagonal; in float that is 1.2·10⁻⁷, larger
than the process noise on ω (10⁻⁹) and q (10⁻⁸), so it acts as extra process noise. With
the double-precision value (2.2·10⁻¹⁶) the float thesis filter stays within **0.003°** of
the double reference on every vector. The bug is in both the C and the Rust port and will
be fixed with the other review findings; this section will then be re-measured.

The MEKF stays within 0.005° except briefly after a declared bias fault (0.4°).
This matters on a Cortex-M4F, whose FPU is single precision only (double is emulated in
software): there, single precision is usable for both filters once the bug is fixed. On this x86 CPU float is only 3–20%
faster.

## Memory

| Build | State | Stack, `adcs_init` | Stack, step (thesis) | Stack, step (MEKF) |
|---|---:|---:|---:|---:|
| C (gcc -O3 LTO) | 36.0 KB | 1.0 KB | 2.5 KB | 2.1 KB |
| C (clang -O3 LTO) | 36.0 KB | 1.0 KB | 3.1 KB | 2.8 KB |
| Rust opt-level 3 | 41.8 KB | 8.2 KB | 5.9 KB | 4.1 KB |
| C float (clang -O3 LTO) | 18.0 KB | 1.2 KB | 2.0 KB | 0.8 KB |
| Rust float | 20.9 KB | 1.2 KB | 3.3 KB | 3.0 KB |

Stack is measured by running on a pattern-filled thread stack (it includes about 0.3 KB of
the driver's own frame). Rust uses more memory:

- **State, +16%:** the C state keeps the two filters in a `union`; the Rust `Adcs` keeps
  both (an `enum` would match C, at the cost of reconstructing it on every `init`).
- **Stack, about 2× per step:** Rust inlines more aggressively, which grows the step's
  frames. The double-precision `adcs_init` uses 8 KB: because `Mekf::ZERO` holds a
  non-zero quaternion, LLVM copies that 8 KB constant through a stack temporary instead of
  writing it in place (found in review; an all-zero constant brings it to 168 B).

## Code size on Cortex-M

`.text` of the bare-metal example (library + newlib + startup), in bytes:

| Build | Cortex-M4F, float | Cortex-M7, double |
|---|---:|---:|
| C -O2 | 22 852 | 22 780 |
| C -Os | 20 164 | 20 676 |
| C -O2, LTO | 24 768 | 25 092 |
| C -Os, LTO | **19 016** | **19 508** |
| Rust opt-level 2 | 89 988 | 95 352 |
| Rust opt-level s | 35 792 | 38 364 |
| Rust opt-level z | 35 344 | 37 720 |

Rust needs about 1.8–1.9× the code of C when both optimise for size, and 4× at `opt-level` 2,
where LLVM fully unrolls the 13×13 loops of the UKF. In the size-optimised M4F images the
ADCS functions themselves take 20.5 KB in Rust and 13.7 KB in C; Rust adds its compiler
runtime (2.3 KB, software double arithmetic used inside `libm`'s float functions) and
5.6 KB of read-only data against 1.0 KB (panic location records and `libm` tables).
Bounds checks are not the main cause: only 23–35 call sites remain after optimisation.

Both fit easily in the 512 KB–2 MB of flash of typical CubeSat on-board computers; for a
small part (64–128 KB) the difference would matter.

## Safety and verification

| | C | Rust |
|---|---|---|
| Memory-safety tooling | AddressSanitizer + UndefinedBehaviorSanitizer over all tests | Safe by construction outside 4 `unsafe` blocks in the FFI crate; Miri over the unit and FFI tests |
| Static analysis | cppcheck (two false positives when both precisions were analysed together) | clippy pedantic, no findings beyond style |
| Fuzzing | libFuzzer + sanitizers: **found an input-overflow bug** (a finite but absurd gyro rate gave an all-zero quaternion), fixed in the reference and both libraries | libFuzzer: nothing new (ported after the fix) |
| Interface | Layout checked against the ctypes mirror | Layout checked against the C compiler, field by field |
| Foreign-input hazards | C `bool` read as is | C `bool` fields read as bytes, so a foreign caller cannot cause undefined behaviour |

The one real bug was algorithmic and language-independent, found by fuzzing the first port
(C). The borrow checker prevented nothing here because the code has no dynamic memory or
sharing; its one visible effect was pushing the first Rust draft toward copying a 36 KB
filter onto the stack to satisfy it, which was then replaced by borrowing disjoint fields.

## Effort

| | Lines (no blanks or comments) |
|---|---:|
| C library (`src/`, `adcs.h`) | 1 958 |
| Rust library (`adcs-core`, `adcs-ffi`, no tests) | 2 125 |

The Rust port took less time than the C one, but it came second: the algorithms, the
operation order and the test vectors were already settled, and the C version was the
template. Its main costs were the `no_std` maths (a crate instead of the system library),
the FFI boundary, and test tolerances that work in both precisions.

## Conclusions

1. **Speed is a draw; build settings decide it.** With LTO on both sides, Rust matches
   clang (same LLVM backend) and gcc is fastest on the larger filter. Without LTO, C loses
   to Rust's default release profile.
2. **Results are bit-identical** across languages, compilers and optimisation levels.
3. **Rust costs memory:** about 1.8–1.9× the code size when optimised for size, 2× the stack per
   step and 16% more state, all small in absolute terms for a CubeSat computer.
4. **Single precision behaves the same in both languages.** An apparent failure of the
   thesis filter in float turned out to be a shared porting bug (see "Single precision").
5. **Not measured:** timing on a real Cortex-M (no board), per-stage timing, and a third
   implementation using `nalgebra`.
