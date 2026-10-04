"""Table 2: unpenalized logistic associations with HC0 sandwich inference."""

import json
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, false_discovery_control
from sklearn.linear_model import LogisticRegression

from common import paths, load_data

SOURCE, ROOT = paths("Table 2: adjusted logistic associations")
OUT = ROOT / "tables/Table2_results.json"
sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
t1 = json.loads((ROOT / "tables/Table1_results.json").read_text())
assert sha == t1["source_sha256"]
df = load_data(SOURCE)
assert len(df) == t1["n"]
exposures = ["PSQI_global", "Stress", "Clenching", "Bruxism"]
numeric = (
    ["HATMD"]
    + exposures
    + ["Age", "Female", "Symptom_duration_mo", "VAS", "Bilateral_pain"]
)
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="raise")
required = numeric + ["Entry_quarter"]
cc = df.dropna(subset=required).copy()
assert cc.HATMD.isin([0, 1]).all()
assert cc.Symptom_duration_mo.ge(0).all()
cc["log_duration"] = np.log1p(cc.Symptom_duration_mo)
quarter = pd.get_dummies(
    cc.Entry_quarter.astype(str), prefix="quarter", drop_first=True, dtype=float
)
y = cc.HATMD.to_numpy(dtype=float)
base = exposures + ["Age", "Female", "log_duration"]


def fit(columns):
    raw = pd.concat([cc[columns].astype(float), quarter], axis=1)
    names = ["Intercept"] + raw.columns.tolist()
    a = raw.to_numpy()
    mean = a.mean(axis=0)
    scale = a.std(axis=0)
    assert np.all(scale > 0)
    x = np.column_stack([np.ones(len(a)), (a - mean) / scale])
    assert np.linalg.matrix_rank(x) == x.shape[1]
    beta = np.zeros(x.shape[1])

    def objective(b):
        eta = x @ b
        return float(np.logaddexp(0, eta).sum() - y @ eta)

    for iteration in range(100):
        prob = expit(x @ beta)
        score = x.T @ (y - prob)
        info = x.T @ (x * (prob * (1 - prob))[:, None])
        if np.max(np.abs(score)) < 1e-8:
            break
        step = np.linalg.solve(info, score)
        loss = objective(beta)
        factor = 1.0
        while factor > 1e-10:
            candidate = beta + factor * step
            if objective(candidate) <= loss + 1e-10:
                beta = candidate
                break
            factor *= 0.5
        else:
            raise RuntimeError("Line search failed")
    else:
        raise RuntimeError("Logistic regression did not converge")
    prob = expit(x @ beta)
    score = x.T @ (y - prob)
    assert np.max(np.abs(score)) < 1e-6
    info = x.T @ (x * (prob * (1 - prob))[:, None])
    inv = np.linalg.inv(info)
    scores = x * (y - prob)[:, None]
    cov = inv @ (scores.T @ scores) @ inv
    cov = (cov + cov.T) / 2
    assert np.linalg.eigvalsh(cov).min() > 0
    # Verify the maximum likelihood estimates with an independent solver.
    sk = LogisticRegression(
        penalty=None, fit_intercept=False, solver="lbfgs", tol=1e-11, max_iter=2000
    )
    sk.fit(x, y)
    solver_difference = float(np.max(np.abs(beta - sk.coef_[0])))
    assert solver_difference < 2e-5
    # Convert coefficients and sandwich covariance back to original units.
    transform = np.eye(len(beta))
    transform[0, 1:] = -mean / scale
    transform[1:, 1:] = np.diag(1 / scale)
    b = transform @ beta
    v = transform @ cov @ transform.T
    se = np.sqrt(np.diag(v))
    p = 2 * norm.sf(np.abs(b / se))
    estimates = []
    for key in exposures:
        i = names.index(key)
        ci = np.exp([b[i] - norm.ppf(0.975) * se[i], b[i] + norm.ppf(0.975) * se[i]])
        estimates.append(
            {
                "variable": key,
                "beta": float(b[i]),
                "se_robust": float(se[i]),
                "or": float(np.exp(b[i])),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "p": float(p[i]),
            }
        )
    return {
        "n": len(y),
        "events": int(y.sum()),
        "parameters": len(beta),
        "quarter_levels": sorted(cc.Entry_quarter.astype(str).unique().tolist()),
        "iterations": iteration + 1,
        "score_max_abs": float(np.max(np.abs(score))),
        "hessian_condition": float(np.linalg.cond(info)),
        "independent_solver_max_difference": solver_difference,
        "estimates": estimates,
    }


m1 = fit(base)
m2 = fit(base + ["VAS", "Bilateral_pain"])
ps = np.array([x["p"] for m in [m1, m2] for x in m["estimates"]])
qs = false_discovery_control(ps, method="bh")
assert np.all(qs >= ps - 1e-14)
for x, q in zip([x for m in [m1, m2] for x in m["estimates"]], qs):
    x["p_adjusted"] = float(q)
    x["or_ci_display"] = f"{x['or']:.2f} ({x['ci_lower']:.2f}–{x['ci_upper']:.2f})"
    x["p_display"] = "<0.001" if x["p"] < 0.001 else f"{x['p']:.3f}"
    x["p_adjusted_display"] = "<0.001" if q < 0.001 else f"{q:.3f}"
result = {
    "source_sha256": sha,
    "total_n": len(df),
    "complete_case_n": len(cc),
    "excluded_n": len(df) - len(cc),
    "model1": m1,
    "model2": m2,
    "fdr_method": "Benjamini–Hochberg",
    "fdr_family_size": 8,
    "missing_by_variable": {c: int(df[c].isna().sum()) for c in required},
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(
    "Complete cases:", len(cc), "HATMD:", int(y.sum()), "Excluded:", len(df) - len(cc)
)
for name, m in [("Model 1", m1), ("Model 2", m2)]:
    print(
        name,
        "parameters",
        m["parameters"],
        "iterations",
        m["iterations"],
        "score",
        m["score_max_abs"],
        "independent fit difference",
        m["independent_solver_max_difference"],
    )
    for x in m["estimates"]:
        print(
            x["variable"],
            x["or_ci_display"],
            "P",
            x["p_display"],
            "FDR P",
            x["p_adjusted_display"],
            "full P",
            x["p"],
        )
