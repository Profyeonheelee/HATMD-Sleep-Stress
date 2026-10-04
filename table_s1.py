"""Table S1: missingness and complete-case inclusion comparisons."""

import hashlib, json
import numpy as np
from scipy import stats
from common import paths, load_data

source, ROOT = paths("Table S1: missingness and complete-case inclusion")
df = load_data(source)
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
    "Entry_quarter",
]
keep = df[required].notna().all(axis=1)
assert len(df) == 3672 and df.Study_ID.nunique() == 3672 and keep.sum() == 3319
assert df.HATMD.notna().all() and df.loc[keep, "HATMD"].sum() == 780
assert np.all(
    df.loc[df.PSQI_global.notna(), "Poor_sleeper"]
    == (df.loc[df.PSQI_global.notna(), "PSQI_global"] > 5)
)
vs = [
    ("HATMD", "HATMD, n (%)", "b", None),
    ("Age", "Age, years", "a", 1),
    ("Female", "Female sex, n (%)", "b", None),
    ("Symptom_duration_mo", "Symptom duration, months", "a", 1),
    ("VAS", "Pain intensity, VAS 0–10", "a", 1),
    ("Bilateral_pain", "Bilateral pain, n (%)", "b", None),
    ("MUO_mm", "Maximum unassisted opening, mm", "a", 1),
    ("Sleep_duration_h", "Sleep duration, hours", "a", 1),
    ("PSQI_global", "PSQI global score, 0–21", "a", 0),
    ("Poor_sleeper", "Poor sleep quality, n (%)", "b", None),
    ("Stress", "Self-reported stress, n (%)", "b", None),
    ("Clenching", "Self-reported clenching, n (%)", "b", None),
    ("Bruxism", "Self-reported bruxism, n (%)", "b", None),
]
rows = []
for var, label, fn, dp in vs:
    arrays = [df.loc[keep, var].dropna(), df.loc[~keep, var].dropna()]
    row = {
        "variable": var,
        "label": label,
        "footnote": fn,
        "available_n": [len(a) for a in arrays],
        "missing_by_group": [int(df.loc[m, var].isna().sum()) for m in [keep, ~keep]],
        "missing_n": int(df[var].isna().sum()),
    }
    if dp is not None:
        qs = [a.quantile([0.25, 0.5, 0.75]).tolist() for a in arrays]
        row["quartiles"] = qs
        row["values"] = [f"{q[1]:.{dp}f} ({q[0]:.{dp}f}–{q[2]:.{dp}f})" for q in qs]
        p = stats.mannwhitneyu(
            *arrays, alternative="two-sided", method="asymptotic", use_continuity=True
        ).pvalue
        row["test"] = "Two-sided Mann–Whitney U with tie and continuity corrections"
    else:
        assert all(set(a.unique()).issubset({0, 1}) for a in arrays)
        events = [int(a.sum()) for a in arrays]
        row["events"] = events
        row["values"] = [f"{n:,} ({100*n/len(a):.1f})" for n, a in zip(events, arrays)]
        contingency = np.array([[int(a.sum()), int(len(a) - a.sum())] for a in arrays])
        test = stats.chi2_contingency(contingency, correction=False)
        assert test.expected_freq.min() >= 5
        p = test.pvalue
        row["test"] = "Pearson chi-square without continuity correction"
        row["minimum_expected_count"] = float(test.expected_freq.min())
    row["p"] = float(p)
    row["missing_display"] = f"{row['missing_n']} ({100*row['missing_n']/len(df):.1f})"
    rows.append(row)
q = stats.false_discovery_control([r["p"] for r in rows], method="bh")


def disp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


for r, padj in zip(rows, q):
    r["p_adjusted"] = float(padj)
    r["p_display"] = disp(r["p"])
    r["q_display"] = disp(padj)
assert all(r["missing_by_group"][0] == 0 for r in rows)
assert sum(~keep) == 353 and int(df.loc[~keep, "HATMD"].sum()) == 81
result = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "total_n": len(df),
    "included_n": int(keep.sum()),
    "excluded_n": int((~keep).sum()),
    "inclusion_variables": required,
    "fdr_family_size": len(rows),
    "fdr_method": "Benjamini–Hochberg",
    "rows": rows,
    "missing_required": {v: int(df[v].isna().sum()) for v in required},
    "excluded_with_complete_psqi": int(df.loc[~keep, "PSQI_global"].notna().sum()),
    "interpretation": "Observed group comparisons do not establish a missing-data mechanism or absence of selection bias.",
}
(ROOT / "tables/TableS1_results.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False)
)
