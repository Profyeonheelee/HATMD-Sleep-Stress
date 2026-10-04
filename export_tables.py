"""Export manuscript-order, aggregate-only CSV tables from analysis JSON."""

import json
import pandas as pd
from common import paths

SOURCE, ROOT = paths("Export aggregate CSV tables")
folder = ROOT / "tables"


def get(name):
    path = folder / f"{name}_results.json"
    return json.loads(path.read_text()) if path.exists() else None


d = get("Table1")
if d:
    pd.DataFrame(
        [
            {
                "Variable": r["label"],
                "All": r["display"][0],
                "No HATMD": r["display"][1],
                "HATMD": r["display"][2],
                "Available N (all)": r["n"][0],
                "Missing N (all)": r["missing"][0],
                "P": r["p"],
                "FDR P": r["p_adjusted"],
                "Test": r["test"],
            }
            for r in d["rows"]
        ]
    ).to_csv(folder / "Table1.csv", index=False)
d = get("Table2")
if d:
    pd.DataFrame(
        [
            {"Model": model, "N": d[model]["n"], "HATMD N": d[model]["events"], **r}
            for model in ["model1", "model2"]
            for r in d[model]["estimates"]
        ]
    ).to_csv(folder / "Table2.csv", index=False)
d = get("Table3")
if d:
    rows = []
    for name, model in d["models"].items():
        for predictors in ["Baseline", "Expanded"]:
            rows.append(
                {
                    "Model": name,
                    "Predictors": predictors,
                    **model[predictors],
                    "Delta AUROC": (
                        model["delta_auc"] if predictors == "Expanded" else None
                    ),
                    "Delta AUROC CI": (
                        model["delta_auc_ci"] if predictors == "Expanded" else None
                    ),
                    "Delta AUROC P": model["p"] if predictors == "Expanded" else None,
                    "Delta AUROC FDR P": (
                        model["p_adjusted"] if predictors == "Expanded" else None
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(folder / "Table3.csv", index=False)
d = get("TableS1")
if d:
    pd.DataFrame(
        [
            {
                "Variable": r["label"],
                "Included": r["values"][0],
                "Excluded": r["values"][1],
                "Missing N (%)": r["missing_display"],
                "P": r["p"],
                "FDR P": r["p_adjusted"],
            }
            for r in d["rows"]
        ]
    ).to_csv(folder / "TableS1.csv", index=False)
d = get("TableS2")
if d:
    pd.DataFrame(
        [
            {"Analysis": label, "N": d[label]["n"], "HATMD N": d[label]["events"], **r}
            for label in ["reference", "multiple_imputation", "adult_complete_case"]
            for r in d[label]["estimates"]
        ]
    ).to_csv(folder / "TableS2.csv", index=False)
d = get("TableS3")
if d:
    pd.DataFrame(
        [
            {"Section": section, **r}
            for section in ["contrasts", "interaction_terms"]
            for r in d[section]
        ]
        + [{"Section": "omnibus_interaction", **d["omnibus_interaction"]}]
    ).to_csv(folder / "TableS3.csv", index=False)
