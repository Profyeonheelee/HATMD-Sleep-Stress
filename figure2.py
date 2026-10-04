"""Figure 2: restricted cubic PSQI spline and observed score distribution."""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "1"
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import paths
from association_models import build_models

SOURCE, ROOT = paths("Figure 2: PSQI spline")
OUT = ROOT / "figures/Figure2"
OUT.mkdir(parents=True, exist_ok=True)
context = build_models(SOURCE)
df, cc, sha, y = context.df, context.cc, context.sha, context.y
m, joint_model, tests = context.m, context.joint_model, context.tests
knots, rcs, quarter, adjust, p = (
    context.knots,
    context.rcs,
    context.quarter,
    context.adjust,
    context.p,
)


def marginal(score):
    a = m["raw"].copy()
    a[:, 0] = score
    a[:, 1] = rcs(score)
    x = np.column_stack([np.ones(len(a)), (a - m["mu"]) / m["sd"]])
    pred = expit(x @ m["beta"])
    probability = float(pred.mean())
    gradient = (x * (pred * (1 - pred))[:, None]).mean(0)
    se = np.sqrt(gradient @ m["cov"] @ gradient)
    bounds = expit(
        logit(probability)
        + np.array([-1, 1]) * norm.ppf(0.975) * se / (probability * (1 - probability))
    )
    # Check the analytic gradient independently with central finite differences.
    if score in [0.0, 5.0, 10.0]:
        eps = 1e-5
        numerical = np.array(
            [
                (
                    expit(x @ (m["beta"] + np.eye(len(gradient))[j] * eps)).mean()
                    - expit(x @ (m["beta"] - np.eye(len(gradient))[j] * eps)).mean()
                )
                / (2 * eps)
                for j in range(len(gradient))
            ]
        )
        assert np.max(abs(gradient - numerical)) < 1e-7
    return probability, float(bounds[0]), float(bounds[1])


maximum = int(cc.PSQI_global.max())
minimum = int(cc.PSQI_global.min())
grid = np.linspace(minimum, maximum, 241)
curve = np.array([marginal(v) for v in grid])
anchors = {
    str(v): dict(zip(["probability", "lower", "upper"], marginal(float(v))))
    for v in [0, 5, 10, 15, maximum]
    if minimum <= v <= maximum
}
counts = (
    cc.PSQI_global.astype(int)
    .value_counts()
    .reindex(range(22), fill_value=0)
    .sort_index()
)
joint_counts = (
    cc.assign(Poor_sleep=p.astype(int))
    .groupby(["Poor_sleep", "Stress", "Clenching"], observed=True)
    .HATMD.agg(["size", "sum"])
    .reset_index()
)
result = {
    "source_sha256": sha,
    "n": len(cc),
    "events": int(y.sum()),
    "excluded_n": len(df) - len(cc),
    "knot_percentiles": [0.1, 0.5, 0.9],
    "knots": knots.tolist(),
    "psqi_observed_min": minimum,
    "psqi_observed_max": maximum,
    "adjustment_variables": adjust + quarter.columns.tolist(),
    "covariance": "HC0 robust sandwich",
    "ci": "pointwise 95%, delta method on logit of marginal mean; covariate distribution held fixed",
    "tests": tests,
    "fdr_family_size": 3,
    "model_verification": {
        "score_max_abs": m["score"],
        "independent_solver_max_difference": m["solver_diff"],
        "iterations": m["iterations"],
    },
    "anchor_probabilities": anchors,
    "score_distribution": {str(k): int(v) for k, v in counts.items()},
    "joint_group_counts": joint_counts.to_dict("records"),
    "coefficient_names": m["names"],
    "standardized_coefficients": m["beta"].tolist(),
    "standardized_covariance": m["cov"].tolist(),
}
(OUT / "Figure2_analysis_results.json").write_text(json.dumps(result, indent=2))
pd.DataFrame(
    {
        "PSQI_global": grid,
        "adjusted_probability": curve[:, 0],
        "ci_lower": curve[:, 1],
        "ci_upper": curve[:, 2],
    }
).to_csv(OUT / "Figure2_curve_values.csv", index=False)


def display(v):
    return "<0.001" if v < 0.001 else f"= {v:.3f}"


plt.rcParams.update(
    {
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig = plt.figure(figsize=(7.05, 5.6), facecolor="white")
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[3.7, 1],
    hspace=0.13,
    left=0.15,
    right=0.97,
    bottom=0.13,
    top=0.96,
)
ax = fig.add_subplot(gs[0])
hist = fig.add_subplot(gs[1], sharex=ax)
color = "#B45B5B"
ax.fill_between(
    grid, curve[:, 1] * 100, curve[:, 2] * 100, color="#28678C", alpha=0.17, linewidth=0
)
ax.plot(grid, curve[:, 0] * 100, color=color, lw=2.1)
ax.set_ylabel("Adjusted probability of HATMD (%)", labelpad=8)
ceiling = min(100, int(np.ceil(curve[:, 2].max() * 100 / 10) * 10 + 10))
ax.set_ylim(0, ceiling)
ax.set_xlim(-0.5, maximum + 0.5)
ax.yaxis.set_major_locator(MultipleLocator(10 if ceiling <= 60 else 20))
ax.grid(axis="y", color="#E7EBEF", lw=0.6)
ax.tick_params(axis="x", labelbottom=False)
overall = tests["PSQI_overall"]
nonlinear = tests["PSQI_nonlinearity"]
annotation = (
    f"Overall association: P {display(overall['p'])}; FDR P {display(overall['p_adjusted'])}\n"
    f"Nonlinearity: P {display(nonlinear['p'])}; FDR P {display(nonlinear['p_adjusted'])}\n"
    f"n = {len(cc):,}; HATMD cases = {int(y.sum()):,}"
)
ax.text(
    0.035,
    0.965,
    annotation,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.1,
    linespacing=1.55,
)
hist.bar(
    counts.index,
    counts.values,
    color="#243F66",
    width=0.86,
    edgecolor="white",
    linewidth=0.3,
)
hist.set_ylabel("Patients (n)", labelpad=8, fontsize=9)
hist.set_xlabel("PSQI global score", labelpad=7)
hist.set_xticks(list(range(0, maximum + 1, 2)))
hist.yaxis.set_major_locator(MultipleLocator(200))
hist.set_ylim(0, max(counts) * 1.16)
hist.text(
    0.98,
    0.90,
    "Observed PSQI distribution",
    transform=hist.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    color="#000000",
)
fig.savefig(OUT / "Figure2_PSQI_HATMD.png", dpi=300)
fig.savefig(OUT / "Figure2_PSQI_HATMD.pdf")
fig.savefig(
    OUT / "Figure2_PSQI_HATMD.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"}
)
plt.close(fig)


print(json.dumps(result, indent=2))
