"""Figure S1: held-out ROC curves, bootstrap bands and calibration."""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "1"
import json, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, chi2, false_discovery_control
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import paths

SOURCE, ROOT = paths("Figure S1: ROC curves and calibration")
OUT = ROOT / "figures/FigureS1"
OUT.mkdir(parents=True, exist_ok=True)

predictions_path = ROOT / "private/out_of_fold_predictions.csv"
results_path = ROOT / "tables/Table3_results.json"
reference = json.loads(results_path.read_text())
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == reference["source_sha256"]
df = pd.read_csv(predictions_path)
y = df.HATMD.to_numpy(int)
assert len(df) == df.Study_ID.nunique() == reference["n"] == 3672
assert int(y.sum()) == reference["events"] == 861
assert df.outer_fold.isin([1, 2, 3, 4, 5]).all()
names = ["Logistic regression", "HistGradientBoosting", "MLP"]
short = ["LR", "HGB", "MLP"]
colors = ["#17365D", "#9E3D43", "#326F74"]
calibration = {}
roc_rows = []
bin_rows = []


def cal_fit(probs):
    x = np.column_stack([np.ones(len(probs)), logit(np.clip(probs, 1e-7, 1 - 1e-7))])
    b = np.array([0.0, 1.0])
    for iteration in range(60):
        fitted = expit(x @ b)
        score = x.T @ (y - fitted)
        if abs(score).max() < 1e-9:
            break
        info = x.T @ (x * (fitted * (1 - fitted))[:, None])
        b += np.linalg.solve(info, score)
    else:
        raise RuntimeError("Calibration regression did not converge")
    fitted = expit(x @ b)
    info = x.T @ (x * (fitted * (1 - fitted))[:, None])
    bread = np.linalg.inv(info)
    scores = x * (y - fitted)[:, None]
    cov = bread @ (scores.T @ scores) @ bread
    cov = (cov + cov.T) / 2
    assert np.linalg.eigvalsh(cov).min() > 0
    independent = LogisticRegression(
        penalty=None, fit_intercept=False, solver="lbfgs", tol=1e-11, max_iter=2000
    ).fit(x, y)
    assert abs(b - independent.coef_[0]).max() < 2e-5
    delta = b - np.array([0.0, 1.0])
    stat = float(delta @ np.linalg.solve(cov, delta))
    se = np.sqrt(np.diag(cov))
    z = norm.ppf(0.975)
    return dict(
        intercept=float(b[0]),
        slope=float(b[1]),
        intercept_ci=(b[0] + np.array([-1, 1]) * z * se[0]).tolist(),
        slope_ci=(b[1] + np.array([-1, 1]) * z * se[1]).tolist(),
        wald_statistic=stat,
        df=2,
        p=float(chi2.sf(stat, 2)),
        score_max_abs=float(abs(score).max()),
        independent_solver_max_difference=float(abs(b - independent.coef_[0]).max()),
        robust_covariance=cov.tolist(),
    )


for name in names:
    for feature in ["Baseline", "Expanded"]:
        probs = df[f"{name}|{feature}"].to_numpy(float)
        assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
        expected = reference["models"][name][feature]
        assert abs(roc_auc_score(y, probs) - expected["auc"]) < 1e-12
        assert (
            abs(average_precision_score(y, probs) - expected["auprc_average_precision"])
            < 1e-12
        )
        assert abs(brier_score_loss(y, probs) - expected["brier"]) < 1e-12
    probs = df[f"{name}|Expanded"].to_numpy(float)
    calibration[name] = cal_fit(probs)
    for metric in ["intercept", "slope"]:
        assert (
            abs(
                calibration[name][metric]
                - reference["models"][name]["Expanded"][metric]
            )
            < 1e-8
        )
    edges = np.percentile(probs, np.linspace(0, 100, 11))
    ids = np.searchsorted(edges[1:-1], probs)
    for group in sorted(np.unique(ids)):
        mask = ids == group
        n = int(mask.sum())
        events = int(y[mask].sum())
        obs = events / n
        prediction = float(probs[mask].mean())
        z = norm.ppf(0.975)
        denom = 1 + z * z / n
        center = (obs + z * z / (2 * n)) / denom
        half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / denom
        bin_rows.append(
            dict(
                model=name,
                bin=int(group + 1),
                n=n,
                events=events,
                predicted_mean=prediction,
                observed_proportion=obs,
                observed_ci_lower=center - half,
                observed_ci_upper=center + half,
            )
        )
    values = [v for v in bin_rows if v["model"] == name]
    observed, predicted = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    assert np.allclose(observed, [v["observed_proportion"] for v in values], atol=1e-12)
    assert np.allclose(predicted, [v["predicted_mean"] for v in values], atol=1e-12)
    assert (
        sum(v["n"] for v in values) == 3672 and sum(v["events"] for v in values) == 861
    )
for name, q in zip(
    names,
    false_discovery_control([calibration[name]["p"] for name in names], method="bh"),
):
    calibration[name]["p_adjusted"] = float(q)


def pformat(value):
    return "<0.001" if value < 0.001 else f"{value:.3f}"


# Pointwise ROC confidence bands, conditional on the same held-out predictions.
display = [
    ("Logistic regression", "Baseline", "LR baseline", "#727272", "-"),
    ("Logistic regression", "Expanded", "LR expanded", colors[0], "-"),
    ("HistGradientBoosting", "Expanded", "HGB expanded", colors[1], "-"),
    ("MLP", "Expanded", "MLP expanded", colors[2], "-"),
]
fpr_grid = np.linspace(0, 1, 201)
bootstrap = np.empty((2000, len(display), len(fpr_grid)))
positive = np.flatnonzero(y == 1)
negative = np.flatnonzero(y == 0)
rng = np.random.default_rng(reference["seed"] + 100)
scores = [df[f"{name}|{feature}"].to_numpy(float) for name, feature, _, _, _ in display]
print("Calculating 2,000 paired outcome-stratified bootstrap ROC bands.", flush=True)
for b in range(2000):
    indices = np.r_[
        rng.choice(positive, len(positive), replace=True),
        rng.choice(negative, len(negative), replace=True),
    ]
    for j, probs in enumerate(scores):
        false_positive, true_positive, _ = roc_curve(
            y[indices], probs[indices], drop_intermediate=False
        )
        bootstrap[b, j] = np.interp(fpr_grid, false_positive, true_positive)
        bootstrap[b, j, 0] = 0
        bootstrap[b, j, -1] = 1
lower_bands, upper_bands = np.quantile(bootstrap, [0.025, 0.975], axis=0)
assert np.all(lower_bands <= upper_bands)
band_rows = []
for j, (name, feature, _, _, _) in enumerate(display):
    band_rows.extend(
        dict(
            model=name,
            predictor_set=feature,
            false_positive_rate=float(f),
            tpr_ci_lower=float(lo),
            tpr_ci_upper=float(hi),
        )
        for f, lo, hi in zip(fpr_grid, lower_bands[j], upper_bands[j])
    )

plt.rcParams.update(
    {
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.05, 4.15))
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.90, wspace=0.36)
for a, title in [(ax, "A  Discrimination"), (bx, "B  Calibration")]:
    a.set_title(
        title, loc="left", fontsize=11, fontweight="bold", pad=12, color="black"
    )
    a.grid(color="#E7EBEF", lw=0.6)
    a.set_aspect("equal", adjustable="box")
for j, (_, _, _, color, _) in enumerate(display):
    ax.fill_between(
        fpr_grid,
        lower_bands[j],
        upper_bands[j],
        color=color,
        alpha=0.20,
        linewidth=0,
        zorder=1,
    )
for name, feature, label, color, style in display:
    fpr, tpr, thresholds = roc_curve(y, df[f"{name}|{feature}"], drop_intermediate=True)
    info = reference["models"][name][feature]
    lo, hi = info["auc_ci"]
    ax.plot(
        fpr,
        tpr,
        color=color,
        ls=style,
        lw=1.65,
        label=f"{label}: {info['auc']:.3f} ({lo:.3f}–{hi:.3f})",
        zorder=3,
    )
    roc_rows.extend(
        dict(
            model=name,
            predictor_set=feature,
            false_positive_rate=float(f),
            true_positive_rate=float(t),
        )
        for f, t in zip(fpr, tpr)
    )
ax.plot([0, 1], [0, 1], color="#969696", lw=0.8, ls="-", zorder=0)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate")
ax.legend(
    loc="lower right",
    fontsize=6.8,
    title="AUROC (95% CI)",
    title_fontsize=7.4,
    frameon=False,
    handlelength=2,
    handletextpad=0.5,
    borderaxespad=0.5,
    labelcolor="black",
)
assert all(
    reference["models"][name]["p"] < 0.001
    and reference["models"][name]["p_adjusted"] < 0.001
    for name in names
)
ax.text(
    0.04,
    0.965,
    "AUROC gain (LR, HGB, MLP):\nAll P <0.001; FDR P <0.001",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.1,
    color="black",
    linespacing=1.5,
)

markers = ["o", "s", "^"]
styles = ["-", "--", "-."]
for name, label, color, marker, style in zip(names, short, colors, markers, styles):
    bins = [v for v in bin_rows if v["model"] == name]
    px = np.array([v["predicted_mean"] for v in bins])
    py = np.array([v["observed_proportion"] for v in bins])
    low = np.array([v["observed_ci_lower"] for v in bins])
    high = np.array([v["observed_ci_upper"] for v in bins])
    bx.errorbar(
        px,
        py,
        yerr=np.vstack([py - low, high - py]),
        fmt="none",
        ecolor=color,
        alpha=0.40,
        elinewidth=0.65,
        capsize=1.5,
        zorder=1,
    )
    bx.plot(
        px,
        py,
        color=color,
        ls=style,
        lw=1.4,
        marker=marker,
        ms=4.5,
        mfc="white",
        mec=color,
        mew=1,
        label=label,
        zorder=3,
    )
limit = float(
    np.ceil(
        (
            max(max(v["predicted_mean"], v["observed_ci_upper"]) for v in bin_rows)
            + 0.025
        )
        * 10
    )
    / 10
)
bx.plot(
    [0, limit],
    [0, limit],
    color="#727272",
    lw=0.9,
    ls=":",
    label="Ideal calibration",
    zorder=0,
)
bx.set_xlim(0, limit)
bx.set_ylim(0, limit)
bx.xaxis.set_major_locator(MultipleLocator(0.1))
bx.yaxis.set_major_locator(MultipleLocator(0.1))
bx.set_xlabel("Mean predicted probability")
bx.set_ylabel("Observed HATMD proportion")
bx.legend(
    loc="lower right", fontsize=7.2, frameon=False, handlelength=2, labelcolor="black"
)
annotation = "Calibration P / FDR P\n" + "\n".join(
    f"{label}: {pformat(calibration[name]['p'])} / {pformat(calibration[name]['p_adjusted'])}"
    for name, label in zip(names, short)
)
bx.text(
    0.04,
    0.965,
    annotation,
    transform=bx.transAxes,
    ha="left",
    va="top",
    fontsize=7.0,
    color="black",
    linespacing=1.45,
)
fig.savefig(OUT / "FigureS1_ROC_Calibration.png", dpi=300)
fig.savefig(OUT / "FigureS1_ROC_Calibration.pdf")
fig.savefig(
    OUT / "FigureS1_ROC_Calibration.tiff",
    dpi=600,
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.close(fig)

summary = {
    "n": len(y),
    "events": int(y.sum()),
    "oof_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    "table3_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    "source_sha256": reference["source_sha256"],
    "outer_folds": 5,
    "inner_folds": 3,
    "model_metrics": reference["models"],
    "calibration_tests": calibration,
    "calibration_p_family_size": 3,
    "calibration_bins": bin_rows,
    "axis_limit_calibration": limit,
    "interpretation": "Pooled out-of-fold internal validation; inferential tests condition on fitted models",
}
summary["roc_confidence_bands"] = {
    "resamples": 2000,
    "sampling": "Paired, outcome-stratified patient bootstrap",
    "fpr_grid_points": 201,
    "coverage": "Pointwise percentile 95%, conditional on fitted models",
    "endpoints": "TPR anchored at 0 and 1 at FPR endpoints",
    "seed": reference["seed"] + 100,
    "alpha": 0.20,
    "transparency": 0.80,
}
(OUT / "FigureS1_analysis_results.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame(bin_rows).to_csv(OUT / "FigureS1_calibration_bins.csv", index=False)
pd.DataFrame(roc_rows).to_csv(OUT / "FigureS1_ROC_curve_values.csv", index=False)
pd.DataFrame(band_rows).to_csv(OUT / "FigureS1_ROC_confidence_bands.csv", index=False)
