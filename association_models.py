"""Shared HC0 logistic models for the spline, joint profiles and contrasts."""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "1"
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, false_discovery_control
from sklearn.linear_model import LogisticRegression
from types import SimpleNamespace
from common import load_data


def build_models(source):
    df = load_data(source)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    required = [
        "HATMD",
        "PSQI_global",
        "Stress",
        "Clenching",
        "Bruxism",
        "Age",
        "Female",
        "Symptom_duration_mo",
        "VAS",
        "Bilateral_pain",
    ]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="raise")
    cc = df.dropna(subset=required + ["Entry_quarter"]).copy()
    assert len(cc) == 3319 and int(cc.HATMD.sum()) == 780
    assert cc.PSQI_global.between(0, 21).all()
    assert np.equal(cc.PSQI_global, np.round(cc.PSQI_global)).all()
    y = cc.HATMD.to_numpy(float)
    cc["log_duration"] = np.log1p(cc.Symptom_duration_mo)
    quarter = pd.get_dummies(
        cc.Entry_quarter.astype(str), prefix="quarter", drop_first=True, dtype=float
    )
    adjust = [
        "Stress",
        "Clenching",
        "Bruxism",
        "Age",
        "Female",
        "log_duration",
        "VAS",
        "Bilateral_pain",
    ]
    knots = np.quantile(cc.PSQI_global, [0.10, 0.50, 0.90])
    assert np.diff(knots).min() > 0

    def rcs(x):
        x = np.asarray(x, dtype=float)
        a, b, c = knots
        positive = lambda v: np.maximum(v, 0) ** 3
        return (
            positive(x - a)
            - (c - a) / (c - b) * positive(x - b)
            + (b - a) / (c - b) * positive(x - c)
        ) / (c - a) ** 2

    def fit(raw):
        names = ["Intercept"] + raw.columns.tolist()
        a = raw.to_numpy(float)
        mu = a.mean(0)
        sd = a.std(0)
        assert (sd > 0).all()
        x = np.column_stack([np.ones(len(a)), (a - mu) / sd])
        assert np.linalg.matrix_rank(x) == x.shape[1]
        beta = np.zeros(x.shape[1])
        loss = lambda b: float(np.logaddexp(0, x @ b).sum() - y @ (x @ b))
        for iteration in range(100):
            p = expit(x @ beta)
            score = x.T @ (y - p)
            if abs(score).max() < 1e-8:
                break
            info = x.T @ (x * (p * (1 - p))[:, None])
            step = np.linalg.solve(info, score)
            before = loss(beta)
            factor = 1.0
            while factor > 1e-10:
                candidate = beta + factor * step
                if loss(candidate) <= before + 1e-10:
                    beta = candidate
                    break
                factor *= 0.5
            else:
                raise RuntimeError("Line search failure")
        else:
            raise RuntimeError("Nonconvergence")
        p = expit(x @ beta)
        score = x.T @ (y - p)
        assert abs(score).max() < 1e-6
        inv = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
        scores = x * (y - p)[:, None]
        cov = inv @ (scores.T @ scores) @ inv
        cov = (cov + cov.T) / 2
        assert np.linalg.eigvalsh(cov).min() > 0
        sk = LogisticRegression(
            penalty=None, fit_intercept=False, solver="lbfgs", tol=1e-11, max_iter=2000
        )
        sk.fit(x, y)
        solver_diff = float(abs(beta - sk.coef_[0]).max())
        assert solver_diff < 2e-5
        return dict(
            names=names,
            beta=beta,
            cov=cov,
            x=x,
            mu=mu,
            sd=sd,
            raw=a,
            score=float(abs(score).max()),
            solver_diff=solver_diff,
            iterations=iteration + 1,
        )

    def wald(model, terms):
        ind = [model["names"].index(t) for t in terms]
        b = model["beta"][ind]
        v = model["cov"][np.ix_(ind, ind)]
        statistic = float(b @ np.linalg.solve(v, b))
        return dict(
            statistic=statistic, df=len(ind), p=float(chi2.sf(statistic, len(ind)))
        )

    raw = pd.concat(
        [
            pd.DataFrame(
                {"PSQI_linear": cc.PSQI_global, "PSQI_nonlinear": rcs(cc.PSQI_global)},
                index=cc.index,
            ),
            cc[adjust],
            quarter,
        ],
        axis=1,
    )
    m = fit(raw)
    tests = {
        "PSQI_overall": wald(m, ["PSQI_linear", "PSQI_nonlinear"]),
        "PSQI_nonlinearity": wald(m, ["PSQI_nonlinear"]),
    }

    # Joint-profile omnibus interaction test completes the three-test FDR family.
    p = (cc.PSQI_global > 5).astype(float)
    s = cc.Stress
    c = cc.Clenching
    joint = pd.DataFrame(
        {
            "Poor_sleep": p,
            "Stress": s,
            "Clenching": c,
            "Sleep_x_stress": p * s,
            "Sleep_x_clenching": p * c,
            "Stress_x_clenching": s * c,
            "Sleep_x_stress_x_clenching": p * s * c,
        },
        index=cc.index,
    )
    joint_raw = pd.concat(
        [
            joint,
            cc[["Bruxism", "Age", "Female", "log_duration", "VAS", "Bilateral_pain"]],
            quarter,
        ],
        axis=1,
    )
    joint_model = fit(joint_raw)
    tests["Figure3_omnibus_interaction"] = wald(
        joint_model,
        [
            "Sleep_x_stress",
            "Sleep_x_clenching",
            "Stress_x_clenching",
            "Sleep_x_stress_x_clenching",
        ],
    )
    qs = false_discovery_control([v["p"] for v in tests.values()], method="bh")
    for test, q in zip(tests.values(), qs):
        test["p_adjusted"] = float(q)

    return SimpleNamespace(
        df=df,
        cc=cc,
        sha=sha,
        y=y,
        m=m,
        joint_model=joint_model,
        tests=tests,
        knots=knots,
        rcs=rcs,
        quarter=quarter,
        adjust=adjust,
        p=p,
    )
