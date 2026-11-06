# C style rules (adcs-c)

The C library is written the way flight software usually is. These rules are enforced by
the build where a tool can check them.

| Rule | Why | Enforced by |
|---|---|---|
| C11, no compiler extensions | Portable to any embedded toolchain | `-std=c11`, `CMAKE_C_EXTENSIONS OFF` |
| No heap: no `malloc`/`free` in the library | Memory use fixed and known before launch | The Cortex-M build checks that `malloc` is not linked |
| The caller allocates the state; every temporary lives in it | Small, bounded stack | `-Wstack-usage=2048` per function (GCC) |
| No recursion, no variable-length arrays | Bounded stack | `-Wvla`; code review (no recursion) |
| No I/O and no global mutable state | Reentrant, testable, nothing hidden | Code review; only the four `adcs.h` functions are exported |
| Fixed-size arrays with dimensions known at compile time | No out-of-bounds indexing from data | Sanitizers and fuzzing |
| Errors as return codes and status flags, never `abort` | The ADCS must keep running on bad input | Fuzzing: every step must return normally |
| Every input checked: NaN, Inf, implausible values | One bad packet must not corrupt the state | `BAD_INPUT` path, fuzzing, golden vector `bad_input` |
| Strict warnings, all errors | Conversions and shadowing are where bugs hide | `-Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion -Wdouble-promotion -Wshadow -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes` |
| `adcs_real` for every floating-point value | Single and double precision from one source | Float build with `-Wdouble-promotion` in CI |
| Same operation order as the reference | Results reproducible to the last bit | Golden vectors (tolerance 10⁻⁹, agreement ~10⁻¹⁴) |
| Comments explain why, with equation numbers from the thesis or the papers | The maths must be traceable | Code review |

Tooling in CI: GCC and Clang, each with and without AddressSanitizer and
UndefinedBehaviorSanitizer; cppcheck for both precisions; libFuzzer; cross builds for
Cortex-M4F and Cortex-M7.

Not adopted: MISRA C. Many of its rules are followed in spirit (no dynamic memory, no
recursion, explicit conversions), but the library has not been checked against it.
