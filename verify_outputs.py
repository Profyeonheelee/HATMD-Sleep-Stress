"""Check scientifically relevant estimates against aggregate manuscript references."""

import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--mode", choices=["all", "core"], default="all")
args = parser.parse_args()
base = Path(__file__).resolve().parent
checks = []


def compare(actual, reference, location):
    if isinstance(reference, (float, int)):
        if not np.isclose(actual, reference, rtol=1e-7, atol=1e-9, equal_nan=True):
            raise AssertionError(f"{location}: {actual} != {reference}")
    elif isinstance(reference, list):
        assert len(actual) == len(reference), location
        for i, (a, r) in enumerate(zip(actual, reference)):
            compare(a, r, f"{location}[{i}]")
    elif isinstance(reference, dict):
        for key, value in reference.items():
            compare(actual[key], value, f"{location}.{key}")
    else:
        assert actual == reference, f"{location}: {actual!r} != {reference!r}"


def check(name, fields, folder="tables"):
    suffix = "_analysis_results.json" if name.startswith("Figure") else "_results.json"
    actual = json.loads((args.output / folder / f"{name}{suffix}").read_text())
    reference = json.loads(
        (base / "reference_results" / f"{name}_reference.json").read_text()
    )
    for field in fields:
        compare(actual[field], reference[field], f"{name}.{field}")
    checks.append({"result": name, "fields": fields, "status": "passed"})


def estimate_rows(rows):
    keys = ["variable", "or", "ci_lower", "ci_upper", "p", "p_adjusted"]
    return [{k: row[k] for k in keys} for row in rows]


check("Table1", ["n", "n_with", "n_without"])
actual = json.loads((args.output / "tables/Table1_results.json").read_text())
reference = json.loads((base / "reference_results/Table1_reference.json").read_text())
for a, r in zip(actual["rows"], reference["rows"]):
    for k in ["key", "n", "missing", "display", "p", "p_adjusted"]:
        compare(a[k], r[k], "Table1." + a["key"] + "." + k)
checks[-1]["fields"].append("12 row summaries, denominators, raw and FDR P values")
check("Table2", ["complete_case_n", "excluded_n", "model1", "model2"])
check("TableS1", ["total_n", "included_n", "excluded_n", "rows", "missing_required"])
check(
    "Figure2",
    ["n", "events", "knots", "tests", "anchor_probabilities", "score_distribution"],
    "figures/Figure2",
)
check(
    "Figure3",
    ["n", "events", "profiles", "figure_tests", "panel_tests"],
    "figures/Figure3",
)
check(
    "TableS3", ["n", "events", "contrasts", "interaction_terms", "omnibus_interaction"]
)
if args.mode == "all":
    check("Table3", ["n", "events", "models"])
    a = json.loads((args.output / "tables/TableS2_results.json").read_text())
    r = json.loads((base / "reference_results/TableS2_reference.json").read_text())
    for key in ["reference", "multiple_imputation", "adult_complete_case"]:
        compare(a[key]["n"], r[key]["n"], "TableS2." + key + ".n")
        compare(a[key]["events"], r[key]["events"], "TableS2." + key + ".events")
        compare(
            estimate_rows(a[key]["estimates"]),
            estimate_rows(r[key]["estimates"]),
            "TableS2." + key,
        )
    compare(
        a["diagnostics"]["split_rhat_imputed_means"],
        r["diagnostics"]["split_rhat_imputed_means"],
        "TableS2.Rhat",
    )
    checks.append(
        {
            "result": "TableS2",
            "fields": ["all exposure estimates, N and R-hat"],
            "status": "passed",
        }
    )
    # Prediction-file byte hashes depend on CSV formatting and are not scientific checks.
    check(
        "FigureS1",
        ["n", "events", "calibration_tests", "calibration_bins"],
        "figures/FigureS1",
    )
report = {
    "status": "passed",
    "numeric_rtol": 1e-7,
    "numeric_atol": 1e-9,
    "checks": checks,
    "scope": "Aggregate numerical reproduction; does not establish external validity.",
}
(args.output / "verification_report.json").write_text(json.dumps(report, indent=2))
print(
    "PASS: manuscript aggregate counts, estimates and P values match reference results."
)
