"""Table 3: leakage-controlled nested CV and paired OOF comparisons."""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "1"
import json, hashlib, time, warnings
import numpy as np
import pandas as pd
import sklearn, scipy
from scipy.stats import rankdata, false_discovery_control
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.exceptions import ConvergenceWarning

from common import paths, load_data

SOURCE, ROOT = paths("Table 3: nested cross-validation classification")
OUT = ROOT / "tables/Table3_results.json"
OOF = ROOT / "private/out_of_fold_predictions.csv"
SEED = 20261003
start = time.time()
sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert (
    sha
    == json.loads((ROOT / "tables/Table2_results.json").read_text())["source_sha256"]
)
df = load_data(SOURCE).reset_index(drop=True)
assert len(df) == 3672
numeric = [
    "HATMD",
    "Age",
    "Female",
    "Symptom_duration_mo",
    "PSQI_global",
    "Stress",
    "Clenching",
    "Bruxism",
]
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="raise")
assert df.HATMD.notna().all() and df.HATMD.isin([0, 1]).all()
assert df.Symptom_duration_mo.dropna().ge(0).all()
df["log_duration"] = np.log1p(df.Symptom_duration_mo)
df["Entry_quarter"] = df.Entry_quarter.astype(str)
y = df.HATMD.to_numpy(dtype=int)
sets = {
    "Baseline": ["Age", "Female", "log_duration"],
    "Expanded": [
        "Age",
        "Female",
        "log_duration",
        "PSQI_global",
        "Stress",
        "Clenching",
        "Bruxism",
    ],
}
models = ["Logistic regression", "HistGradientBoosting", "MLP"]
outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y))
pred = {
    f"{name}|{feature}": np.full(len(y), np.nan) for name in models for feature in sets
}
fold_id = np.full(len(y), -1)
fitlogs = []
for fold, (train, test) in enumerate(outer, 1):
    fold_id[test] = fold
    inner = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold).split(
            df.iloc[train], y[train]
        )
    )
    assert not set(train).intersection(test)
    for name in models:
        for feature, cols in sets.items():
            numeric_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                ]
            )
            preprocess = ColumnTransformer(
                [
                    ("numeric", numeric_pipe, cols),
                    (
                        "quarter",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["Entry_quarter"],
                    ),
                ],
                sparse_threshold=0,
            )
            if name == "Logistic regression":
                estimator = LogisticRegression(
                    max_iter=2000, solver="lbfgs", random_state=SEED + fold
                )
                grid = {"model__C": [0.1, 1.0, 10.0]}
            elif name == "HistGradientBoosting":
                estimator = HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=SEED + fold,
                )
                grid = {"model__max_leaf_nodes": [7, 15, 31]}
            else:
                estimator = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    batch_size=128,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    tol=1e-4,
                    random_state=SEED + fold,
                )
                grid = {"model__alpha": [0.01, 0.1, 1.0]}
            pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(df.iloc[train], y[train])
            warnings_found = [
                str(w.message)
                for w in caught
                if issubclass(w.category, ConvergenceWarning)
            ]
            assert (
                not warnings_found
            ), f"Convergence warning: {name} {feature} {fold}: {warnings_found}"
            probs = search.predict_proba(df.iloc[test])[:, 1]
            assert np.isfinite(probs).all() and np.all((probs >= 0) & (probs <= 1))
            pred[f"{name}|{feature}"][test] = probs
            best = search.best_estimator_.named_steps["model"]
            fitlogs.append(
                {
                    "outer_fold": fold,
                    "model": name,
                    "predictor_set": feature,
                    "train_n": len(train),
                    "test_n": len(test),
                    "test_events": int(y[test].sum()),
                    "best_parameters": search.best_params_,
                    "best_inner_auc": float(search.best_score_),
                    "outer_auc": float(roc_auc_score(y[test], probs)),
                    "iterations": (
                        int(best.n_iter_) if hasattr(best, "n_iter_") else None
                    ),
                }
            )
            print(
                f"Fold {fold}/5 | {name} | {feature} | AUC {roc_auc_score(y[test],probs):.3f} | elapsed {time.time()-start:.0f}s",
                flush=True,
            )
    # Retain a checkpoint after every outer fold.
    frame = pd.DataFrame(
        {"Study_ID": df.Study_ID, "HATMD": y, "outer_fold": fold_id, **pred}
    )
    frame.to_csv(OOF, index=False)
assert (fold_id > 0).all() and all(np.isfinite(p).all() for p in pred.values())


def fast_auc(labels, scores):
    pos = labels == 1
    np_ = int(pos.sum())
    nn = len(labels) - np_
    ranks = rankdata(scores, method="average")
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


for p in pred.values():
    assert abs(fast_auc(y, p) - roc_auc_score(y, p)) < 1e-12


def calibration(p):
    xx = logit(np.clip(p, 1e-7, 1 - 1e-7))
    x = np.column_stack([np.ones(len(p)), xx])
    beta = np.array([0.0, 1.0])
    for _ in range(30):
        prob = expit(x @ beta)
        score = x.T @ (y - prob)
        info = x.T @ (x * (prob * (1 - prob))[:, None])
        change = np.linalg.solve(info, score)
        beta += change
        if np.max(np.abs(change)) < 1e-10:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


result = {
    "source_sha256": sha,
    "n": len(y),
    "events": int(y.sum()),
    "prevalence": float(y.mean()),
    "seed": SEED,
    "outer_folds": 5,
    "inner_folds": 3,
    "candidate_settings_per_model": 3,
    "bootstrap_resamples": 2000,
    "permutation_resamples": 10000,
    "fdr_family_size": 3,
    "software": {
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    },
    "missing_inputs": {
        k: int(df[k].isna().sum()) for k in ["Symptom_duration_mo", "PSQI_global"]
    },
    "models": {},
    "fit_logs": fitlogs,
}
for name in models:
    result["models"][name] = {}
    for feature in sets:
        p = pred[f"{name}|{feature}"]
        result["models"][name][feature] = {
            "auc": float(roc_auc_score(y, p)),
            "auprc_average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            **calibration(p),
        }
    result["models"][name]["delta_auc"] = (
        result["models"][name]["Expanded"]["auc"]
        - result["models"][name]["Baseline"]["auc"]
    )

print("Training complete. Calculating paired bootstrap intervals.", flush=True)
keys = list(pred)
boot = np.empty((2000, len(keys)))
pos = np.flatnonzero(y == 1)
neg = np.flatnonzero(y == 0)
rng = np.random.default_rng(SEED + 100)
for b in range(2000):
    ind = np.concatenate(
        [
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ]
    )
    yy = y[ind]
    for j, key in enumerate(keys):
        boot[b, j] = fast_auc(yy, pred[key][ind])
for i, name in enumerate(models):
    for j, feature in enumerate(sets):
        result["models"][name][feature]["auc_ci"] = np.quantile(
            boot[:, 2 * i + j], [0.025, 0.975]
        ).tolist()
    result["models"][name]["delta_auc_ci"] = np.quantile(
        boot[:, 2 * i + 1] - boot[:, 2 * i], [0.025, 0.975]
    ).tolist()

print("Calculating paired permutation P values.", flush=True)
ps = []
for i, name in enumerate(models):
    p0 = pred[f"{name}|Baseline"]
    p1 = pred[f"{name}|Expanded"]
    observed = abs(result["models"][name]["delta_auc"])
    rng = np.random.default_rng(SEED + 200 + i)
    extreme = 0
    for b in range(10000):
        swap = rng.random(len(y)) < 0.5
        a = np.where(swap, p1, p0)
        c = np.where(swap, p0, p1)
        null = abs(fast_auc(y, c) - fast_auc(y, a))
        extreme += int(null >= observed - 1e-14)
    pval = (extreme + 1) / 10001
    result["models"][name]["p"] = pval
    ps.append(pval)
    print(name, "delta", result["models"][name]["delta_auc"], "P", pval, flush=True)
for name, q in zip(models, false_discovery_control(ps, method="bh")):
    result["models"][name]["p_adjusted"] = float(q)
result["elapsed_seconds"] = time.time() - start
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
for name in models:
    print(name, json.dumps(result["models"][name], ensure_ascii=False), flush=True)
print("Completed in", round(result["elapsed_seconds"], 1), "seconds", flush=True)
