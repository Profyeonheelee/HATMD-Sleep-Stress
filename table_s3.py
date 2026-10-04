"""Table S3: standardized probability contrasts and interaction inference."""

import json
import numpy as np
from scipy.special import expit
from scipy.stats import norm, false_discovery_control
from common import paths
from association_models import build_models

SOURCE, ROOT = paths("Table S3: profile contrasts and interactions")
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
ref = json.loads((OUT / "Figure3_analysis_results.json").read_text())
assert ref["source_sha256"] == sha
m = joint_model
cache = {}


def profile(tup):
    poor, stress, clench = tup
    a = m["raw"].copy()
    a[:, :7] = [
        poor,
        stress,
        clench,
        poor * stress,
        poor * clench,
        stress * clench,
        poor * stress * clench,
    ]
    x = np.column_stack([np.ones(len(a)), (a - m["mu"]) / m["sd"]])
    prob = expit(x @ m["beta"])
    pmean = float(prob.mean())
    grad = (x * (prob * (1 - prob))[:, None]).mean(axis=0)
    eps = 1e-5
    eye = np.eye(len(grad))
    ng = np.array(
        [
            (
                expit(x @ (m["beta"] + eps * e)).mean()
                - expit(x @ (m["beta"] - eps * e)).mean()
            )
            / (2 * eps)
            for e in eye
        ]
    )
    assert np.max(abs(ng - grad)) < 1e-7
    existing = next(
        r
        for r in ref["profiles"]
        if (r["poor_sleep"], r["stress"], r["clenching"]) == tup
    )
    assert abs(pmean - existing["adjusted_probability"]) < 1e-12
    return pmean, grad, x


for row in ref["profiles"]:
    tup = (row["poor_sleep"], row["stress"], row["clenching"])
    cache[tup] = profile(tup)
comparisons = [
    (
        "All three exposures vs none",
        (1, 1, 1),
        (0, 0, 0),
        "Poor sleep with stress and clenching vs good sleep without stress or clenching",
    ),
    (
        "Clenching within poor sleep and stress",
        (1, 1, 1),
        (1, 1, 0),
        "Clenching vs no clenching among patients with poor sleep and stress",
    ),
    (
        "Poor sleep within stress and clenching",
        (1, 1, 1),
        (0, 1, 1),
        "Poor vs good sleep among patients with stress and clenching",
    ),
]
contrasts = []
for key, first, second, label in comparisons:
    pa, ga, xa = cache[first]
    pb, gb, xb = cache[second]
    delta = pa - pb
    gradient = ga - gb
    var = float(gradient @ m["cov"] @ gradient)
    se = np.sqrt(var)
    ci = delta + np.array([-1, 1]) * norm.ppf(0.975) * se
    numerical = []
    eps = 1e-5
    for e in np.eye(len(gradient)):
        fn = lambda b: expit(xa @ b).mean() - expit(xb @ b).mean()
        numerical.append(
            (fn(m["beta"] + eps * e) - fn(m["beta"] - eps * e)) / (2 * eps)
        )
    assert np.max(np.abs(gradient - numerical)) < 1e-7
    contrasts.append(
        {
            "comparison": key,
            "label": label,
            "first_profile": first,
            "second_profile": second,
            "first_probability": pa,
            "second_probability": pb,
            "difference": delta,
            "se": float(se),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "p": float(2 * norm.sf(abs(delta / se))),
        }
    )
qs = false_discovery_control([r["p"] for r in contrasts], method="bh")
for r, q in zip(contrasts, qs):
    r["p_adjusted"] = float(q)
# Convert standardized coefficients and covariance to original 0/1 units.
transform = np.eye(len(m["beta"]))
transform[0, 1:] = -m["mu"] / m["sd"]
transform[1:, 1:] = np.diag(1 / m["sd"])
b = transform @ m["beta"]
v = transform @ m["cov"] @ transform.T
terms = [
    ("Sleep_x_stress", "Poor sleep × stress (no clenching)"),
    ("Sleep_x_clenching", "Poor sleep × clenching (no stress)"),
    ("Stress_x_clenching", "Stress × clenching (good sleep)"),
    ("Sleep_x_stress_x_clenching", "Poor sleep × stress × clenching"),
]
interactions = []
for term, label in terms:
    i = m["names"].index(term)
    se = np.sqrt(v[i, i])
    ci = np.exp(b[i] + np.array([-1, 1]) * norm.ppf(0.975) * se)
    interactions.append(
        {
            "term": term,
            "label": label,
            "beta": float(b[i]),
            "se": float(se),
            "or_ratio": float(np.exp(b[i])),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "p": float(2 * norm.sf(abs(b[i] / se))),
        }
    )
qs = false_discovery_control([r["p"] for r in interactions], method="bh")
for r, q in zip(interactions, qs):
    r["p_adjusted"] = float(q)
# Check interaction contrast identities on the log-odds scale at fixed covariates.
for term, _ in terms:
    i = m["names"].index(term)
    if term == "Sleep_x_stress":
        coeff = {(1, 1, 0): 1, (1, 0, 0): -1, (0, 1, 0): -1, (0, 0, 0): 1}
    elif term == "Sleep_x_clenching":
        coeff = {(1, 0, 1): 1, (1, 0, 0): -1, (0, 0, 1): -1, (0, 0, 0): 1}
    elif term == "Stress_x_clenching":
        coeff = {(0, 1, 1): 1, (0, 1, 0): -1, (0, 0, 1): -1, (0, 0, 0): 1}
    else:
        coeff = {tup: (-1) ** (3 - sum(tup)) for tup in cache}
    contrast_vector = sum(weight * cache[tup][2][0] for tup, weight in coeff.items())
    assert abs(contrast_vector @ m["beta"] - b[i]) < 1e-10
    assert abs(contrast_vector @ m["cov"] @ contrast_vector - v[i, i]) < 1e-10
omnibus = ref["figure_tests"]["Figure3_omnibus_interaction"]
assert abs(omnibus["p"] - tests["Figure3_omnibus_interaction"]["p"]) < 1e-12
result = {
    "source_sha256": sha,
    "n": len(cc),
    "events": int(y.sum()),
    "model": ref["model"],
    "adjustment_variables": ref["adjustment_variables"],
    "contrasts": contrasts,
    "contrast_fdr_family_size": 3,
    "interaction_terms": interactions,
    "individual_interaction_fdr_family_size": 4,
    "omnibus_interaction": omnibus,
    "omnibus_fdr_family": "Figure 2 overall/nonlinearity and Figure 3 omnibus tests (3 tests), retained unchanged",
    "contrast_ci": "Nominal 95% Wald interval for marginal standardized probability difference using joint delta-method covariance, conditional on the common observed covariate distribution",
    "covariance": "HC0 robust sandwich",
    "status": "Exploratory supplementary analyses selected after inspection of Figure 3; no prespecification claimed",
    "verification": {
        "same_coefficients_and_covariance_as_Figure3": True,
        "same_eight_profile_probabilities_as_Figure3": True,
        "analytic_gradients_verified_by_finite_differences": True,
        "interaction_coefficients_verified_by_logodds_contrast_identities": True,
        "score_max_abs": m["score"],
    },
}


def pdsp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


for r in contrasts:
    r["estimate_display"] = (
        f"{100*r['difference']:.1f} ({100*r['ci_lower']:.1f}–{100*r['ci_upper']:.1f})"
    )
for r in interactions:
    r["estimate_display"] = (
        f"{r['or_ratio']:.2f} ({r['ci_lower']:.2f}–{r['ci_upper']:.2f})"
    )
for r in contrasts + interactions + [omnibus]:
    r["p_display"] = pdsp(r["p"])
    r["q_display"] = pdsp(r["p_adjusted"])
(ROOT / "tables/TableS3_results.json").write_text(json.dumps(result, indent=2))
for family, rows in [
    ("Profile contrasts", contrasts),
    ("Interaction coefficients", interactions),
]:
    print(family)
    for r in rows:
        print(
            r["label"],
            r["estimate_display"],
            "P",
            r["p_display"],
            "FDR P",
            r["q_display"],
        )
print(
    "Omnibus interaction",
    omnibus["statistic"],
    omnibus["df"],
    omnibus["p_display"],
    omnibus["q_display"],
)
