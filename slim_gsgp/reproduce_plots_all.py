# Single size-vs-RMSE pareto for the ALL variant, comparing the three SLIM
# versions (ABS, N1, N2) WITH and WITHOUT early stopping -> 6 lines.
#
# Like the "full_pareto" block of reproduce_plots.py: each line is expressed as
# % change vs a BASE reference (BASE.best_fitness, per dataset, then averaged
# over datasets). Lower-left is better; the base sits at (0, 0) by construction.
# Negative = improvement (smaller size / lower RMSE than base).
#
# A SINGLE shared base is used for all 6 lines: BASE_CSV below.
#
#   colour   -> SLIM version (ABS / N1 / N2)
#   dashed   -> with early stopping      (legend: ABS+ES, N1+ES, N2+ES)
#   solid    -> without early stopping   (legend: ABS, N1, N2)
#
# Data layout (see reproduce_results.py):
#   reproduced_results_3/<version>/ALL.csv        (early stopping ON)
#   reproduced_results_3_noES/<version>/ALL.csv   (early stopping OFF)
# Any (version, mode) lacking ALL.csv is skipped with a warning.
#
# Paths are anchored to this file, so the script works from any cwd.

import os

import pandas as pd
import matplotlib.pyplot as plt

VARIANT_NAME = "ALL"

# One colour + marker per SLIM version.
VERSIONS = ["SLIM+ABS", "SLIM+N1", "SLIM+N2"]
VERSION_STYLE = {
    "SLIM+ABS": {"tag": "ABS", "color": "#3377FF", "marker": "o"},
    "SLIM+N1":  {"tag": "N1",  "color": "#2ca02c", "marker": "^"},
    "SLIM+N2":  {"tag": "N2",  "color": "#d62728", "marker": "s"},
}

# Early-stopping on/off: results root, legend suffix and line look.
ES_MODES = [
    {"root": "reproduced_results_3_noES", "suffix": "",    "linestyle": "-",  "fill": True},   # no ES
    {"root": "reproduced_results_3",      "suffix": "+ES", "linestyle": "--", "fill": False},  # ES
]

IND_ORDER = ["best_fitness", "optimal_compromise", "best_size"]

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../slim_gsgp
_REPO_ROOT = os.path.dirname(_HERE)                  # repo root

# Single shared baseline for every line (% change is measured against this).
BASE_CSV = os.path.join(_REPO_ROOT, "reproduced_results_3_noES", "SLIM+ABS", "BASE.csv")


def baseline_ref(base_csv: str):
    """BASE.best_fitness median per dataset (fitness, size) -> the (0,0) ref."""
    b = pd.read_csv(base_csv, index_col=[0, 1, 2]).reset_index()
    b["variant"] = "BASE"
    b = b.set_index(["variant", "dataset", "individual", "run"]).sort_index()
    medians = b.groupby(level=["variant", "dataset", "individual"]).median()
    return medians.xs(("BASE", "best_fitness"), level=["variant", "individual"])


def pareto_pct(all_csv: str, baseline):
    """% change of ALL's (BF, OC, BS) vs the shared baseline, per dataset then
    averaged over datasets. Mirrors reproduce_plots.py. Returns (sizes, rmses)."""
    a = pd.read_csv(all_csv, index_col=[0, 1, 2]).reset_index()
    a["variant"] = VARIANT_NAME
    a = a.set_index(["variant", "dataset", "individual", "run"]).sort_index()

    medians = a.groupby(level=["variant", "dataset", "individual"]).median()
    pcs = medians.sub(baseline, level="dataset").div(baseline, level="dataset") * 100
    mean_pcs = pcs.groupby(level="individual").mean()

    xs = [mean_pcs.loc[ind]["size"] for ind in IND_ORDER]
    ys = [mean_pcs.loc[ind]["fitness"] for ind in IND_ORDER]
    return xs, ys


if __name__ == "__main__":

    base_plots_path = os.path.join(_HERE, "reproduced_plots", "all")
    os.makedirs(base_plots_path, exist_ok=True)

    if not os.path.isfile(BASE_CSV):
        raise SystemExit(f"Baseline not found: {BASE_CSV}")
    print(f"Baseline : {BASE_CSV}")
    baseline = baseline_ref(BASE_CSV)

    fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
    ax_p.axhline(0.0, color="black", linewidth=0.8)
    ax_p.axvline(0.0, color="black", linewidth=0.8)

    plotted = []   # (version_idx, mode_idx, line) -> reordered for the legend
    for mode_idx, mode in enumerate(ES_MODES):       # no-ES (0) then ES (1)
        for version_idx, version in enumerate(VERSIONS):
            style = VERSION_STYLE[version]
            _version_dir = version.replace("*", "x")
            all_csv = os.path.join(_REPO_ROOT, mode["root"], _version_dir, f"{VARIANT_NAME}.csv")

            label = f"{style['tag']}{mode['suffix']}"
            if not os.path.isfile(all_csv):
                print(f"[skip] {label}: {all_csv} not found")
                continue

            xs, ys = pareto_pct(all_csv, baseline)
            color = style["color"]
            line, = ax_p.plot(
                xs, ys,
                linewidth=2, markersize=6,
                marker=style["marker"],
                markeredgecolor=color,
                markerfacecolor=color if mode["fill"] else "none",
                linestyle=mode["linestyle"],
                color=color,
                label=label, alpha=0.75,
            )
            plotted.append((version_idx, mode_idx, line))
            pts = ", ".join(f"({x:.1f},{y:.1f})" for x, y in zip(xs, ys))
            print(f"[plot] {label}: {pts}")

    if not plotted:
        raise SystemExit(
            f"No {VARIANT_NAME}.csv found under reproduced_results_3[/_noES]/<version>/. "
            f"Run reproduce_results.py for the ALL variant, with early stopping on and off.")

    ax_p.set_xlabel("Mean Change in Size (%)", fontsize=11)
    ax_p.set_ylabel("Mean Change in RMSE (%)", fontsize=11)
    ax_p.xaxis.set_label_position("top")
    ax_p.grid()
    # matplotlib fills legend columns top-to-bottom, so order version-major
    # (no-ES then ES) to get one version per column: top = plain, bottom = +ES.
    legend_lines = [line for _, _, line in sorted(plotted, key=lambda t: (t[0], t[1]))]
    fig_p.legend(handles=legend_lines, loc="upper center",
                 bbox_to_anchor=(0.5, 0.04), ncol=len(VERSIONS), frameon=False, fontsize=8)

    out_path = os.path.join(base_plots_path, f"pareto_{VARIANT_NAME}_ES_vs_noES.pdf")
    fig_p.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
