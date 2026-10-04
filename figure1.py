"""Figure 1: HATMD group distributions using the Table 1 testing family."""

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import paths, load_data

SOURCE, ROOT = paths("Figure 1: direct HATMD group comparisons")
OUT = ROOT / "figures/Figure1"
OUT.mkdir(parents=True, exist_ok=True)
d = load_data(SOURCE)
reference = json.loads((ROOT / "tables/Table1_results.json").read_text())
stats = {
    x["key"]: {"p": x["p"], "adjusted_p": x["p_adjusted"]} for x in reference["rows"]
}
assert len(d) == reference["n"] == 3672
assert int(d.HATMD.sum()) == reference["n_with"] == 861
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    }
)
colors = ["#234361", "#9B4E5A"]
fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.1))
fig.subplots_adjust(
    left=0.065, right=0.985, top=0.94, bottom=0.14, wspace=0.35, hspace=0.42
)
items = [
    ("PSQI_global", "A", "Sleep quality", "PSQI global score", (0, 26)),
    ("VAS", "B", "Pain intensity", "VAS score", (0, 12.7)),
    ("Sleep_duration_h", "C", "Sleep duration", "Hours per night", (0, 16)),
    ("Stress", "D", "Self-reported stress", "Patients reporting stress (%)", (0, 80)),
    (
        "Clenching",
        "E",
        "Self-reported clenching",
        "Patients reporting clenching (%)",
        (0, 80),
    ),
    (
        "Bruxism",
        "F",
        "Self-reported bruxism",
        "Patients reporting bruxism (%)",
        (0, 80),
    ),
]
rng = np.random.default_rng(25033)


def ptext(p):
    return "< 0.001" if p < 0.001 else "= " + f"{p:.3f}"


def wilson(k, n):
    z = 1.959963984540054
    ph = k / n
    center = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (center - half) * 100, (center + half) * 100


for ax, (col, letter, title, ylabel, ylim) in zip(axes.flat, items):
    s = stats[col]
    groups = [
        pd.to_numeric(d.loc[d.HATMD == k, col]).dropna().to_numpy() for k in (0, 1)
    ]
    ax.set_title(
        f"{letter}  {title}", loc="left", fontsize=11, fontweight="bold", pad=12
    )
    ax.text(
        0.5,
        0.975,
        f'P {ptext(s["p"])}\nFDR P {ptext(s["adjusted_p"])}',
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.7,
        linespacing=1.3,
    )
    ax.set_xlim(0.35, 2.65)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E7EBEE", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", labelsize=9, width=0.7)
    ax.set_xticks(
        [1, 2],
        [f"No HATMD\nn = {len(groups[0]):,}", f"With HATMD\nn = {len(groups[1]):,}"],
        fontsize=9.2,
    )
    if col in ("PSQI_global", "VAS", "Sleep_duration_h"):
        for pos, x, color in zip([1, 2], groups, colors):
            violin = ax.violinplot(
                [x],
                positions=[pos],
                widths=0.70,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.23,
                points=150,
            )
            body = violin["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.20)
            body.set_linewidth(1.2)
            ax.scatter(
                pos + rng.uniform(-0.235, 0.235, len(x)),
                x,
                s=3,
                c=color,
                alpha=0.085,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            lo = np.min(x[x >= q1 - 1.5 * (q3 - q1)])
            hi = np.max(x[x <= q3 + 1.5 * (q3 - q1)])
            ax.bxp(
                [dict(q1=q1, med=med, q3=q3, whislo=lo, whishi=hi, fliers=[])],
                positions=[pos],
                widths=0.14,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(
                    facecolor="white", edgecolor=color, linewidth=1.25, zorder=4
                ),
                medianprops=dict(color="black", linewidth=1.35, zorder=5),
                whiskerprops=dict(color=color, linewidth=1.0, zorder=4),
                capprops=dict(color=color, linewidth=1.0, zorder=4),
                manage_ticks=False,
            )
            ax.scatter(
                [pos],
                [x.mean()],
                marker="D",
                s=27,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=6,
            )
        if col == "PSQI_global":
            ax.set_yticks([0, 5, 10, 15, 20])
        elif col == "VAS":
            ax.set_yticks([0, 2, 4, 6, 8, 10])
        else:
            ax.set_yticks([0, 3, 6, 9, 12])
    else:
        for pos, x, color in zip([1, 2], groups, colors):
            assert set(x).issubset({0, 1})
            k, n = int(x.sum()), len(x)
            pct = k / n * 100
            low, high = wilson(k, n)
            ax.bar(
                pos,
                pct,
                width=0.52,
                color=color,
                edgecolor=color,
                linewidth=0.7,
                zorder=3,
            )
            ax.errorbar(
                pos,
                pct,
                yerr=np.array([[pct - low], [high - pct]]),
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=3,
                capthick=0.9,
                zorder=4,
            )
            ax.text(
                pos,
                high + 1.4,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )
        ax.set_yticks([0, 20, 40, 60])

legend = [
    Line2D(
        [0],
        [0],
        marker="D",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="none",
        label="Mean (A–C)",
    ),
    Patch(
        facecolor="white",
        edgecolor="black",
        label="Box: median and interquartile range (A–C)",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=0.9,
        marker="|",
        markersize=6,
        label="Whiskers: 95% CI (D–F)",
    ),
]
fig.legend(
    handles=legend,
    loc="lower center",
    bbox_to_anchor=(0.52, 0.04),
    ncol=3,
    frameon=False,
    fontsize=9,
    handlelength=1.4,
    columnspacing=1.7,
)

base = OUT / "Figure1_HATMD_Group_Comparison"
fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
fig.savefig(
    base.with_suffix(".tiff"),
    dpi=600,
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(base.with_suffix(".svg"), dpi=300, facecolor="white")
fig.savefig(base.with_suffix(".pdf"), facecolor="white")
plt.close(fig)
print(
    json.dumps(
        {
            "outputs": [
                str(base.with_suffix(ext)) for ext in [".png", ".tiff", ".svg"]
            ],
            "figure_size_in": [11.5, 8.1],
            "raster_dpi": 600,
            "p_adjustment": "Benjamini–Hochberg across original 12 Table 1 comparisons",
        },
        indent=2,
    )
)
