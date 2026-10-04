"""Figure 3: marginal sleep, stress and clenching joint-profile probabilities."""

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
from scipy.stats import chi2, norm, false_discovery_control
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import paths
from association_models import build_models

SOURCE, ROOT = paths("Figure 3: sleep, stress and clenching profiles")
OUT = ROOT / "figures/Figure3"
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


def joint_probability(poor, stress, clenching):
    a = joint_model["raw"].copy()
    a[:, :7] = [
        poor,
        stress,
        clenching,
        poor * stress,
        poor * clenching,
        stress * clenching,
        poor * stress * clenching,
    ]
    x = np.column_stack([np.ones(len(a)), (a - joint_model["mu"]) / joint_model["sd"]])
    prob = expit(x @ joint_model["beta"])
    average = float(prob.mean())
    gradient = (x * (prob * (1 - prob))[:, None]).mean(0)
    variance = float(gradient @ joint_model["cov"] @ gradient)
    assert variance > 0
    lower, upper = expit(
        logit(average)
        + np.array([-1, 1])
        * norm.ppf(0.975)
        * np.sqrt(variance)
        / (average * (1 - average))
    )
    # Independent numerical gradient verification for each joint profile.
    eps = 1e-5
    ident = np.eye(len(gradient))
    numerical = np.array(
        [
            (
                expit(x @ (joint_model["beta"] + ident[j] * eps)).mean()
                - expit(x @ (joint_model["beta"] - ident[j] * eps)).mean()
            )
            / (2 * eps)
            for j in range(len(gradient))
        ]
    )
    assert abs(gradient - numerical).max() < 1e-7
    return average, float(lower), float(upper)


rows = []
for clenching in [0, 1]:
    for poor, stress in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        subset = cc[(p == poor) & (cc.Stress == stress) & (cc.Clenching == clenching)]
        probability, lower, upper = joint_probability(poor, stress, clenching)
        rows.append(
            dict(
                poor_sleep=poor,
                stress=stress,
                clenching=clenching,
                n=len(subset),
                events=int(subset.HATMD.sum()),
                adjusted_probability=probability,
                ci_lower=lower,
                ci_upper=upper,
            )
        )
assert sum(v["n"] for v in rows) == 3319
assert sum(v["events"] for v in rows) == 780
assert min(v["n"] for v in rows) > 0
result = {
    "source_sha256": sha,
    "n": len(cc),
    "events": int(y.sum()),
    "excluded_n": len(df) - len(cc),
    "poor_sleep_definition": "PSQI_global > 5",
    "adjustment_variables": [
        "Bruxism",
        "Age",
        "Female",
        "log_duration",
        "VAS",
        "Bilateral_pain",
    ]
    + quarter.columns.tolist(),
    "model": "Logistic regression with sleep, stress, clenching, all pairwise interactions and their three-way interaction",
    "covariance": "HC0 robust sandwich",
    "standardization": "All joint profiles averaged over the same 3319 participants retaining observed adjustment variables",
    "ci": "Pointwise nominal 95%, delta method on logit of marginal mean conditional on observed covariate distribution",
    "figure_tests": tests,
    "fdr_family_size": 3,
    "profiles": rows,
    "model_verification": {
        "score_max_abs": joint_model["score"],
        "independent_solver_max_difference": joint_model["solver_diff"],
        "iterations": joint_model["iterations"],
    },
    "coefficient_names": joint_model["names"],
    "standardized_coefficients": joint_model["beta"].tolist(),
    "standardized_covariance": joint_model["cov"].tolist(),
}
(OUT / "Figure3_analysis_results.json").write_text(json.dumps(result, indent=2))
pd.DataFrame(rows).to_csv(OUT / "Figure3_profile_values.csv", index=False)

# Secondary panel-wise omnibus tests: equality of the four joint profiles within each clenching level.
panel_tests = []
for level in [0, 1]:
    contrasts = []
    for poor, stress in [(1, 0), (0, 1), (1, 1)]:
        vector = np.zeros(len(joint_model["beta"]))
        raw_weights = [
            poor,
            stress,
            0,
            poor * stress,
            poor * level,
            stress * level,
            poor * stress * level,
        ]
        for index, weight in enumerate(raw_weights):
            vector[index + 1] = weight / joint_model["sd"][index]
        contrasts.append(vector)
    L = np.array(contrasts)
    assert np.linalg.matrix_rank(L) == 3
    b = L @ joint_model["beta"]
    v = L @ joint_model["cov"] @ L.T
    statistic = float(b @ np.linalg.solve(v, b))
    panel_tests.append(
        {
            "clenching": level,
            "statistic": statistic,
            "df": 3,
            "p": float(chi2.sf(statistic, 3)),
        }
    )
for test, q in zip(
    panel_tests, false_discovery_control([v["p"] for v in panel_tests], method="bh")
):
    test["p_adjusted"] = float(q)
result["panel_tests"] = panel_tests
result["panel_test_fdr_family_size"] = 2
result["panel_test_status"] = (
    "Secondary exploratory omnibus tests; two-panel FDR family distinct from the shared three-figure-test family"
)
(OUT / "Figure3_analysis_results.json").write_text(json.dumps(result, indent=2))


def pdisplay(value):
    return "<0.001" if value < 0.001 else f"= {value:.3f}"


plt.rcParams.update(
    {
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(7.05, 4.25), sharey=True)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.26, top=0.90, wspace=0.16)
colors = ["#17365D", "#9E3D43", "#17365D", "#9E3D43"]
labels = [
    "Good sleep\nNo stress",
    "Poor sleep\nNo stress",
    "Good sleep\nStress",
    "Poor sleep\nStress",
]
ceiling = min(100, int(np.ceil(max(v["ci_upper"] for v in rows) * 100 / 10) * 10 + 10))
for panel, clenching in zip(axes, [0, 1]):
    vals = rows[4 * clenching : 4 * clenching + 4]
    panel.set_xlim(-0.45, 3.45)
    panel.set_ylim(0, ceiling)
    panel.yaxis.set_major_locator(MultipleLocator(10))
    panel.grid(axis="y", color="#E7EBEF", lw=0.6, zorder=0)
    panel.spines["left"].set_color("black")
    panel.spines["bottom"].set_color("black")
    for i, (v, color) in enumerate(zip(vals, colors)):
        center = v["adjusted_probability"] * 100
        lo = v["ci_lower"] * 100
        hi = v["ci_upper"] * 100
        panel.errorbar(
            i,
            center,
            yerr=[[center - lo], [hi - center]],
            fmt="o",
            ms=8.1,
            mfc="white",
            mec=color,
            mew=1.8,
            color=color,
            ecolor=color,
            elinewidth=1.9,
            capsize=4.5,
            capthick=1.6,
            zorder=3,
        )
        panel.text(
            i,
            hi + 1.4,
            f"{center:.1f}%",
            ha="center",
            va="bottom",
            color="black",
            fontsize=8.3,
        )
    panel.set_xticks(
        range(4), [f"{label}\nn = {v['n']:,}" for label, v in zip(labels, vals)]
    )
    panel.tick_params(axis="x", length=0, pad=9)
    test = panel_tests[clenching]
    panel.text(
        0.02,
        0.97,
        f"Overall group comparison: P {pdisplay(test['p'])}\nFDR P {pdisplay(test['p_adjusted'])}",
        transform=panel.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        color="black",
        linespacing=1.5,
    )
    panel.set_title(
        "A  No clenching" if clenching == 0 else "B  Clenching",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color="black",
    )
axes[0].set_ylabel("Adjusted probability of HATMD (%)", labelpad=9)
axes[1].tick_params(axis="y", left=False, labelleft=False)
axes[1].spines["left"].set_visible(False)
t = tests["Figure3_omnibus_interaction"]
fig.text(
    0.55,
    0.085,
    "Sleep quality and self-reported stress",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
)
fig.savefig(OUT / "Figure3_Sleep_Stress_Clenching.png", dpi=300)
fig.savefig(OUT / "Figure3_Sleep_Stress_Clenching.pdf")
fig.savefig(
    OUT / "Figure3_Sleep_Stress_Clenching.tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.close(fig)
