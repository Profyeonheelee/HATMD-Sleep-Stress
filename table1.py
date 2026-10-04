"""Table 1: available-case HATMD group summaries and 12-test BH adjustment."""

import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

from common import paths, load_data

SOURCE, ROOT = paths("Table 1: baseline comparisons")
DEST = ROOT / "tables/Table1_results.json"
df = load_data(SOURCE)
variables = [
    ("Age", "Age, years", "continuous", 1),
    ("Female", "Female sex, n (%)", "binary", 1),
    ("Symptom_duration_mo", "Symptom duration, months", "continuous", 1),
    ("VAS", "Pain intensity, VAS 0–10", "continuous", 1),
    ("Bilateral_pain", "Bilateral pain, n (%)", "binary", 1),
    ("MUO_mm", "Maximum unassisted opening, mm", "continuous", 1),
    ("Sleep_duration_h", "Sleep duration, hours", "continuous", 1),
    ("PSQI_global", "PSQI global score, 0–21", "continuous", 0),
    ("Poor_sleeper", "Poor sleep quality, n (%)", "binary", 1),
    ("Stress", "Self-reported stress, n (%)", "binary", 1),
    ("Clenching", "Self-reported clenching, n (%)", "binary", 1),
    ("Bruxism", "Self-reported bruxism, n (%)", "binary", 1),
]
for key in ["HATMD"] + [v[0] for v in variables]:
    df[key] = pd.to_numeric(df[key], errors="raise")
assert df["HATMD"].notna().all()
assert set(df["HATMD"].unique()) == {0, 1}
assert df["VAS"].dropna().between(0, 10).all()
assert df["PSQI_global"].dropna().between(0, 21).all()
assert df["Symptom_duration_mo"].dropna().ge(0).all()
assert df["Sleep_duration_h"].dropna().between(0, 24).all()
mask = df["PSQI_global"].notna()
assert np.all(
    df.loc[mask, "Poor_sleeper"] == (df.loc[mask, "PSQI_global"] > 5).astype(int)
)
assert df.loc[~mask, "Poor_sleeper"].isna().all()
result = {
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "n": len(df),
    "n_without": int((df.HATMD == 0).sum()),
    "n_with": int((df.HATMD == 1).sum()),
    "rows": [],
}
for key, label, kind, decimals in variables:
    arrays = [
        df[key].dropna(),
        df.loc[df.HATMD == 0, key].dropna(),
        df.loc[df.HATMD == 1, key].dropna(),
    ]
    row = {
        "key": key,
        "label": label,
        "kind": kind,
        "n": [len(x) for x in arrays],
        "missing": [
            int(len(df) - len(arrays[0])),
            int(result["n_without"] - len(arrays[1])),
            int(result["n_with"] - len(arrays[2])),
        ],
    }
    if kind == "continuous":
        qq = [
            np.quantile(x.to_numpy(), [0.25, 0.5, 0.75], method="linear").tolist()
            for x in arrays
        ]
        row["quartiles"] = qq
        row["display"] = [
            f"{q[1]:.{decimals}f} ({q[0]:.{decimals}f}–{q[2]:.{decimals}f})" for q in qq
        ]
        tst = stats.mannwhitneyu(
            arrays[1],
            arrays[2],
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        )
        row["statistic"] = float(tst.statistic)
        row["p"] = float(tst.pvalue)
        row["test"] = "Mann–Whitney U"
        row["means"] = [float(x.mean()) for x in arrays]
        row["sd"] = [float(x.std(ddof=1)) for x in arrays]
    else:
        assert all(set(x.unique()).issubset({0, 1}) for x in arrays)
        events = [int(x.sum()) for x in arrays]
        assert events[0] == events[1] + events[2]
        row["events"] = events
        row["display"] = [f"{n:,} ({100*n/len(x):.1f})" for n, x in zip(events, arrays)]
        contingency = np.array(
            [
                [events[1], len(arrays[1]) - events[1]],
                [events[2], len(arrays[2]) - events[2]],
            ]
        )
        chi = stats.chi2_contingency(contingency, correction=False)
        row["min_expected"] = float(chi.expected_freq.min())
        if (chi.expected_freq < 5).any():
            tst = stats.fisher_exact(contingency, alternative="two-sided")
            row["p"] = float(tst.pvalue)
            row["test"] = "Fisher exact"
        else:
            row["p"] = float(chi.pvalue)
            row["test"] = "Pearson chi-square"
            row["statistic"] = float(chi.statistic)
    row["p_display"] = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
    result["rows"].append(row)
assert result["n_without"] + result["n_with"] == result["n"]
for row, q in zip(
    result["rows"],
    stats.false_discovery_control([r["p"] for r in result["rows"]], method="bh"),
):
    row["p_adjusted"] = float(q)
    row["p_adjusted_display"] = "<0.001" if q < 0.001 else f"{q:.3f}"
result["multiplicity"] = {"method": "Benjamini–Hochberg", "family_size": 12}
DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(
    json.dumps({k: result[k] for k in ["n", "n_without", "n_with"]}, ensure_ascii=False)
)
for r in result["rows"]:
    print(
        r["label"],
        r["display"],
        "P",
        r["p_display"],
        "missing",
        r["missing"],
        "test",
        r["test"],
    )
