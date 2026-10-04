"""Table S2: component-level FCS imputation and adult-only sensitivity."""

import os

for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
import json, hashlib, time
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.stats import norm, t, false_discovery_control
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import LogisticRegression
from common import paths, load_data

SOURCE, ROOT = paths("Table S2: multiple imputation and adult-only sensitivity")
df = load_data(SOURCE)
ref = json.load(open(ROOT / "tables/Table2_results.json"))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == ref["source_sha256"]
exposures = ["PSQI_global", "Stress", "Clenching", "Bruxism"]
required = (
    ["HATMD"]
    + exposures
    + ["Age", "Female", "Symptom_duration_mo", "VAS", "Bilateral_pain", "Entry_quarter"]
)
cc = df.dropna(subset=required).copy()
adult = cc[cc.Age >= 18].copy()
components = [f"PSQI_C{i}" for i in range(1, 8)]
assert np.allclose(
    df.loc[df.PSQI_global.notna(), components].sum(axis=1), df.PSQI_global.dropna()
)
quarter = pd.get_dummies(
    df.Entry_quarter.astype(str), prefix="quarter", drop_first=True, dtype=float
)


# Unpenalized logistic regression on standardized columns, returning coefficients
# and HC0 covariance on the original scale.
def logistic(x, y, covariance=True):
    b = np.zeros(x.shape[1])
    for it in range(60):
        p = expit(x @ b)
        score = x.T @ (y - p)
        info = x.T @ (x * (p * (1 - p))[:, None])
        if np.max(np.abs(score)) < 1e-8:
            break
        step = np.linalg.solve(info, score)
        loss = np.logaddexp(0, x @ b).sum() - y @ (x @ b)
        fac = 1.0
        while fac > 1e-9:
            candidate = b + fac * step
            if (
                np.logaddexp(0, x @ candidate).sum() - y @ (x @ candidate)
                <= loss + 1e-9
            ):
                b = candidate
                break
            fac *= 0.5
        else:
            raise RuntimeError("Logistic line search failed")
    else:
        raise RuntimeError("Logistic fit failed to converge")
    p = expit(x @ b)
    score = x.T @ (y - p)
    assert np.max(np.abs(score)) < 1e-6
    info = x.T @ (x * (p * (1 - p))[:, None])
    inverse = np.linalg.inv(info)
    if covariance:
        residual_scores = x * (y - p)[:, None]
        cov = inverse @ (residual_scores.T @ residual_scores) @ inverse
    else:
        cov = inverse
    return b, (cov + cov.T) / 2, float(np.max(np.abs(score)))


def analysis(z, verify=False):
    raw = pd.concat(
        [
            z[exposures + ["Age", "Female"]].astype(float),
            np.log1p(z.Symptom_duration_mo).rename("log_duration"),
            z[["VAS", "Bilateral_pain"]].astype(float),
            quarter.loc[z.index],
        ],
        axis=1,
    )
    names = ["Intercept"] + raw.columns.tolist()
    a = raw.to_numpy()
    mean = a.mean(axis=0)
    scale = a.std(axis=0)
    assert np.all(scale > 0)
    x = np.column_stack([np.ones(len(z)), (a - mean) / scale])
    y = z.HATMD.to_numpy(float)
    b, cov, score = logistic(x, y)
    if verify:
        check = LogisticRegression(
            penalty=None, fit_intercept=False, tol=1e-11, max_iter=2000
        ).fit(x, y)
        assert np.max(np.abs(check.coef_[0] - b)) < 2e-5
    tr = np.eye(len(b))
    tr[0, 1:] = -mean / scale
    tr[1:, 1:] = np.diag(1 / scale)
    return tr @ b, tr @ cov @ tr.T, names, score


def estimates(b, cov, names, dist=norm, dfree=None):
    out = []
    for name in exposures:
        i = names.index(name)
        se = np.sqrt(cov[i, i])
        quantile = dist.ppf(0.975) if dfree is None else dist.ppf(0.975, dfree[i])
        p = 2 * (
            dist.sf(abs(b[i] / se))
            if dfree is None
            else dist.sf(abs(b[i] / se), dfree[i])
        )
        ci = np.exp([b[i] - quantile * se, b[i] + quantile * se])
        out.append(
            {
                "variable": name,
                "beta": float(b[i]),
                "se": float(se),
                "or": float(np.exp(b[i])),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "p": float(p),
                "df": None if dfree is None else float(dfree[i]),
            }
        )
    return out


b, v, names, _ = analysis(cc, True)
for a, r in zip(estimates(b, v, names), ref["model2"]["estimates"]):
    assert abs(a["or"] - r["or"]) < 1e-8 and abs(a["p"] - r["p"]) < 1e-8
br, vr, nr, _ = analysis(adult, True)
adult_model = {
    "n": len(adult),
    "events": int(adult.HATMD.sum()),
    "estimates": estimates(br, vr, nr),
}

targets = components + ["log_duration", "VAS", "Bilateral_pain"]
original = np.column_stack(
    [
        df[components].to_numpy(float),
        np.log1p(df.Symptom_duration_mo),
        df.VAS,
        df.Bilateral_pain,
    ]
)
missing = np.isnan(original)
fixed = np.column_stack(
    [
        df[["HATMD", "Stress", "Clenching", "Bruxism", "Age", "Female"]].to_numpy(
            float
        ),
        quarter.to_numpy(float),
    ]
)
start = time.time()
M = 50
cycles = 70
burn = 50
trace = []
fits = []
covs = []
imputed = []
max_scores = []


def design(z, j):
    a = np.column_stack([fixed, np.delete(z, j, axis=1)])
    sd = a.std(axis=0)
    assert np.all(sd > 0)
    return np.column_stack([np.ones(len(a)), (a - a.mean(axis=0)) / sd])


for chain in range(M):
    rng = np.random.default_rng(np.random.SeedSequence([20261003, 2, chain]))
    z = original.copy()
    for j in range(len(targets)):
        z[missing[:, j], j] = rng.choice(
            z[~missing[:, j], j], missing[:, j].sum(), replace=True
        )
    chain_trace = []
    for iteration in range(cycles):
        # Preserve the study's two deterministic random-number streams.
        if iteration == 35:
            rng = np.random.default_rng(
                np.random.SeedSequence([20261003, 2, chain, 35])
            )
        for j, target in enumerate(targets):
            mi = missing[:, j]
            ob = ~mi
            x = design(z, j)
            xo = x[ob]
            ym = original[ob, j]
            if target == "Bilateral_pain":
                coefficient, vc, _ = logistic(xo, ym, False)
                draw = coefficient + np.linalg.cholesky(vc) @ rng.standard_normal(
                    len(coefficient)
                )
                z[mi, j] = rng.binomial(1, expit(x[mi] @ draw))
            else:
                gram = xo.T @ xo
                ridge = 1e-8 * np.diag(gram)
                ridge[0] = 0
                inv = cho_solve(
                    cho_factor(gram + np.diag(ridge)), np.eye(gram.shape[0])
                )
                coefficient = inv @ (xo.T @ ym)
                residual = ym - xo @ coefficient
                sigma2 = (residual @ residual) / rng.chisquare(len(ym) - xo.shape[1])
                draw = coefficient + np.sqrt(sigma2) * np.linalg.cholesky(
                    inv
                ) @ rng.standard_normal(len(coefficient))
                # Type 1 predictive mean matching, five nearest observed donors.
                donor_predictions = xo @ coefficient
                recipient_predictions = x[mi] @ draw
                distance = np.abs(
                    recipient_predictions[:, None] - donor_predictions[None, :]
                )
                nearest = np.argpartition(distance, 4, axis=1)[:, :5]
                chosen = nearest[
                    np.arange(len(nearest)), rng.integers(0, 5, len(nearest))
                ]
                z[mi, j] = ym[chosen]
        chain_trace.append(
            [float(z[missing[:, j], j].mean()) for j in range(len(targets))]
        )
    assert np.array_equal(z[~missing], original[~missing])
    assert (
        np.isfinite(z).all()
        and np.isin(z[:, :7], [0, 1, 2, 3]).all()
        and np.isin(z[:, -1], [0, 1]).all()
    )
    complete = df.copy()
    complete[components] = z[:, :7]
    complete["PSQI_global"] = z[:, :7].sum(axis=1)
    complete["Symptom_duration_mo"] = np.expm1(z[:, 7])
    complete["VAS"] = z[:, 8]
    complete["Bilateral_pain"] = z[:, 9]
    assert np.array_equal(
        complete.loc[df.PSQI_global.notna(), "PSQI_global"], df.PSQI_global.dropna()
    )
    bm, vm, nm, score = analysis(complete, chain == 0)
    assert nm == names
    fits.append(bm)
    covs.append(vm)
    imputed.append(z)
    trace.append(chain_trace)
    max_scores.append(score)
    if (chain + 1) % 10 == 0:
        print(
            "Completed",
            chain + 1,
            "of",
            M,
            "imputations; elapsed",
            round(time.time() - start, 1),
            "seconds",
            flush=True,
        )

q = np.array(fits)
u = np.array(covs)
qbar = q.mean(axis=0)
ubar = u.mean(axis=0)
between = np.cov(q, rowvar=False, ddof=1)
total = ubar + (1 + 1 / M) * between
lam = (1 + 1 / M) * np.diag(between) / np.diag(total)
dfcom = len(df) - len(names)
df_old = (M - 1) / np.maximum(lam, 1e-8) ** 2
df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
df_pooled = 1 / (1 / df_old + 1 / df_obs)
mi_model = {
    "n": len(df),
    "events": int(df.HATMD.sum()),
    "imputations": M,
    "estimates": estimates(qbar, total, names, t, df_pooled),
}
for a in mi_model["estimates"]:
    i = names.index(a["variable"])
    a["fraction_between_variance"] = float(lam[i])
    a["mcse_beta"] = float(np.sqrt(between[i, i] / M))
    a["mcse_fraction_se"] = float(np.sqrt(between[i, i] / M) / np.sqrt(total[i, i]))

# Split-chain R-hat on the final twenty iterations of imputed-value means.
last = np.asarray(trace)[:, -20:, :]
split = np.concatenate([last[:, :10, :], last[:, 10:, :]], axis=0)
within = split.var(axis=1, ddof=1).mean(axis=0)
btrace = 10 * split.mean(axis=1).var(axis=0, ddof=1)
rhat = np.sqrt(((9 / 10) * within + btrace / 10) / within)
sens = mi_model["estimates"] + adult_model["estimates"]
qs = false_discovery_control([a["p"] for a in sens], method="bh")
for a, padj in zip(sens, qs):
    a["p_adjusted"] = float(padj)


def display(a):
    a["or_ci_display"] = f"{a['or']:.2f} ({a['ci_lower']:.2f}–{a['ci_upper']:.2f})"
    a["p_display"] = "<0.001" if a["p"] < 0.001 else f"{a['p']:.3f}"
    a["q_display"] = "<0.001" if a["p_adjusted"] < 0.001 else f"{a['p_adjusted']:.3f}"


reference = {
    "n": len(cc),
    "events": int(cc.HATMD.sum()),
    "estimates": ref["model2"]["estimates"],
}
for model in [reference, mi_model, adult_model]:
    for a in model["estimates"]:
        display(a)
result = {
    "source_sha256": ref["source_sha256"],
    "reference": reference,
    "multiple_imputation": mi_model,
    "adult_complete_case": adult_model,
    "sensitivity_fdr_family_size": 8,
    "reference_fdr_family_size": 8,
    "imputation_method": "Custom fully conditional specification; posterior parameter draws; type 1 predictive mean matching (5 observed donors) for PSQI components, log1p symptom duration and VAS; logistic-normal parameter draws and Bernoulli sampling for bilateral pain; PSQI global passively recalculated as the component sum; observed cells held fixed",
    "imputation_assumption": "Missing at random conditional on the imputation predictors",
    "imputation_predictors": [
        "HATMD",
        "Stress",
        "Clenching",
        "Bruxism",
        "Age",
        "Female",
        "Entry_quarter",
    ]
    + targets,
    "chains": M,
    "cycles_per_chain": cycles,
    "burn_in": burn,
    "seed_sequence": [20261003, 2, "chain index"],
    "continuation_seed_sequence": [20261003, 2, "chain index", 35],
    "diagnostics": {
        "split_rhat_imputed_means": dict(zip(targets, map(float, rhat))),
        "max_fit_score": float(max(max_scores)),
        "observed_values_preserved": True,
        "PSQI_component_ranges_preserved": True,
        "PSQI_global_identity_preserved": True,
        "MCSE_fraction_SE": {
            a["variable"]: a["mcse_fraction_se"] for a in mi_model["estimates"]
        },
    },
    "pooled_df": "Barnard–Rubin, complete-data residual df n-p",
    "within_imputation_covariance": "HC0 robust sandwich",
    "software": {"numpy": np.__version__, "pandas": pd.__version__},
    "elapsed_seconds": time.time() - start,
}
np.savez_compressed(
    ROOT / "private/imputation_checkpoint.npz",
    imputations=np.array(imputed),
    coefficients=q,
    covariances=u,
    trace=np.array(trace),
)
(ROOT / "tables/TableS2_results.json").write_text(json.dumps(result, indent=2))
print("DIAGNOSTICS", json.dumps(result["diagnostics"]), flush=True)
for label, model in [
    ("Reference", reference),
    ("MI", mi_model),
    ("Adult complete cases", adult_model),
]:
    print(label, model["n"], model["events"], flush=True)
    for a in model["estimates"]:
        print(
            a["variable"],
            a["or_ci_display"],
            "P",
            a["p_display"],
            "FDR P",
            a["q_display"],
            flush=True,
        )
