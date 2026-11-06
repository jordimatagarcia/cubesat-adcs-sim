"""Tables and figure for docs/results/c_vs_rust.md from the benchmark data.

    tools/bench.sh                      # host timing, stack, accuracy → bench_host.jsonl
    tools/size_report.sh                # Cortex-M code size → sizes.json
    uv run python tools/plot_bench.py   # prints Markdown tables, writes the figure

Over rounds: the median of the means and percentiles, the maximum of the maxima.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "results" / "data"
FIGURE = REPO / "docs" / "images" / "c_vs_rust_timing.png"

ORDER = [
    "c-gcc-O2",
    "c-gcc-O3",
    "c-gcc-O3-lto",
    "c-clang-O2",
    "c-clang-O3",
    "c-clang-O2-lto",
    "c-clang-O3-lto",
    "rust-O2",
    "rust-O3",
    "c-gcc-O3-lto-f32",
    "c-clang-O3-lto-f32",
    "rust-O3-f32",
]


def load() -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    meta: dict[str, object] = {}
    runs: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for line in (DATA / "bench_host.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("meta"):
            meta = d
        else:
            runs[(d["label"], d["variant"])].append(d)
    agg: dict[tuple[str, str], dict[str, object]] = {}
    for key, rs in runs.items():
        a = dict(rs[0])
        for f in ("mean_us", "p50_us", "p99_us", "p999_us"):
            a[f] = statistics.median(float(r[f]) for r in rs)  # type: ignore[arg-type]
        a["max_us"] = max(float(r["max_us"]) for r in rs)  # type: ignore[arg-type]
        a["rounds"] = len(rs)
        agg[key] = a
    return meta, agg


def timing_table(agg: dict[tuple[str, str], dict[str, object]]) -> str:
    rows = [
        "| Build | Thesis mean | p99 | max | MEKF mean | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ORDER:
        t, m = agg.get((label, "thesis")), agg.get((label, "mekf"))
        if t is None or m is None:
            continue
        rows.append(
            f"| `{label}` | {t['mean_us']:.2f} | {t['p99_us']:.2f} | {t['max_us']:.1f} "
            f"| {m['mean_us']:.2f} | {m['p99_us']:.2f} | {m['max_us']:.1f} |"
        )
    return "\n".join(rows)


def memory_table(agg: dict[tuple[str, str], dict[str, object]]) -> str:
    rows = [
        "| Build | State (bytes) | Stack: `adcs_init` | Stack: step, thesis | Stack: step, MEKF |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ORDER:
        t, m = agg.get((label, "thesis")), agg.get((label, "mekf"))
        if t is None or m is None:
            continue
        rows.append(
            f"| `{label}` | {t['state_bytes']} | {t['stack_init_bytes']} "
            f"| {t['stack_step_bytes']} | {m['stack_step_bytes']} |"
        )
    return "\n".join(rows)


def accuracy_table(agg: dict[tuple[str, str], dict[str, object]]) -> str:
    rows = [
        "| Build | Variant | Max attitude diff. (°) | RMS (°) | Output hash |",
        "|---|---|---:|---:|---|",
    ]
    for label in ORDER:
        for variant in ("thesis", "mekf"):
            a = agg.get((label, variant))
            if a is None:
                continue
            rows.append(
                f"| `{label}` | {variant} | {float(a['max_att_deg']):.1e} "  # type: ignore[arg-type]
                f"| {float(a['rms_att_deg']):.1e} | `{a['hash']}` |"  # type: ignore[arg-type]
            )
    return "\n".join(rows)


def per_vector_table(agg: dict[tuple[str, str], dict[str, object]], label: str) -> str:
    rows = ["| Vector | Thesis (°) | MEKF (°) |", "|---|---:|---:|"]
    t = agg[(label, "thesis")]["max_att_deg_per_vector"]
    m = agg[(label, "mekf")]["max_att_deg_per_vector"]
    assert isinstance(t, dict) and isinstance(m, dict)
    for name in sorted(t):
        rows.append(f"| {name.removesuffix('.json')} | {t[name]:.1e} | {m[name]:.1e} |")
    return "\n".join(rows)


def size_table() -> str:
    path = DATA / "sizes.json"
    if not path.exists():
        return "(no sizes.json)"
    sizes = json.loads(path.read_text())
    rows = ["| Build | Cortex-M4F, float (.text) | Cortex-M7, double (.text) |", "|---|---:|---:|"]
    for build, targets in sizes.items():
        rows.append(
            f"| {build} | {targets.get('cortex-m4f-float', '–')} "
            f"| {targets.get('cortex-m7-double', '–')} |"
        )
    return "\n".join(rows)


def figure(agg: dict[tuple[str, str], dict[str, object]]) -> None:
    labels = [lb for lb in ORDER if (lb, "thesis") in agg and not lb.endswith("-f32")]
    colors = [
        "#4285f4"
        if lb.startswith("c-gcc")
        else "#3cba54"
        if lb.startswith("c-clang")
        else "#e8453c"
        for lb in labels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, variant, title in zip(
        axes, ("thesis", "mekf"), ("Thesis (FLAE + two-stage UKF)", "MEKF"), strict=True
    ):
        mean = [float(agg[(lb, variant)]["mean_us"]) for lb in labels]  # type: ignore[arg-type]
        p99 = [float(agg[(lb, variant)]["p99_us"]) for lb in labels]  # type: ignore[arg-type]
        y = range(len(labels))
        ax.barh(y, mean, color=colors)
        ax.scatter(p99, y, color="black", marker="|", s=120, zorder=3, label="p99")
        ax.set_yticks(list(y), labels)
        ax.invert_yaxis()
        ax.set_xlabel("time per adcs_step (µs)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[1].set_yticklabels([])
    fig.suptitle(
        "Same driver, same inputs: C (gcc blue, clang green) and Rust (red), double precision",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=100)


def main() -> None:
    meta, agg = load()
    print("Machine:", {k: v for k, v in meta.items() if k != "meta"}, "\n")
    print("## Time per step (µs)\n")
    print(timing_table(agg), "\n")
    print("## Memory\n")
    print(memory_table(agg), "\n")
    print("## Accuracy against the double-precision reference\n")
    print(accuracy_table(agg), "\n")
    print("## Single precision per vector (rust-O3-f32)\n")
    print(per_vector_table(agg, "rust-O3-f32"), "\n")
    print("## Code size\n")
    print(size_table())
    figure(agg)
    print(f"\nwrote {FIGURE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
