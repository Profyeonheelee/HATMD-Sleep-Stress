# HATMD analysis code

**Sleep Quality, Stress, and Oral Parafunction in Headache Attributed to Temporomandibular Disorders**

This package reproduces the statistical analyses and figures for the retrospective cross-sectional HATMD study (IRB KH-DT25033). The original cohort includes 3,672 patients, 861 with HATMD. Adjusted association analyses use 3,319 complete cases; classification uses all 3,672 patients with preprocessing inside training folds.

## Running the analyses

Python 3.12 is recommended. The source data are not included. Use the **Data** sheet of the original Excel file, or a CSV file with the same columns and patient order. `data_dictionary.csv` describes the variables, and `data_template.csv` provides an empty input template.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and run the full analysis:

```bash
python -m pip install -r requirements.txt
python run_analysis.py --source /path/to/TMD_Sleep_Regularity_Final.xlsx --output results --verify
```

Enclose paths containing spaces in double quotation marks. The command above runs all analyses. To avoid overwriting existing files, specify a new folder with `--output`. The default output folder is `results`.

To examine the main associations and Figures 1–3 first, use the following command. This mode does not run Table 3, Table S2, or Figure S1.

```bash
python run_analysis.py --source /path/to/data.xlsx --output results_core --mode core --verify
```

Individual scripts accept the same `--source` and `--output` options. For example, run Table 1 before Table 2:

```bash
python scripts/table1.py --source /path/to/data.xlsx --output results
python scripts/table2.py --source /path/to/data.xlsx --output results
```

## Manuscript-to-code mapping

| Manuscript item | Script | Content | Prerequisite |
| --- | --- | --- | --- |
| Table 1 | `scripts/table1.py` | Descriptive statistics by HATMD status; raw and FDR-adjusted P values | None |
| Table 2 | `scripts/table2.py` | Adjusted ORs for Models 1 and 2; HC0-based 95% CIs | Table 1 |
| Table 3 | `scripts/table3.py` | Nested CV: logistic regression, HGB, and MLP | Table 2 |
| Table S1 | `scripts/table_s1.py` | Missingness and comparison of included and excluded patients | None |
| Table S2 | `scripts/table_s2.py` | Multiple-imputation and adult-only sensitivity analyses | Table 2 |
| Table S3 | `scripts/table_s3.py` | Joint-exposure probability differences and interaction coefficients | Figure 3 |
| Figure 1 | `scripts/figure1.py` | Between-group distributions using violin, scatter, and box plots; proportions | Table 1 |
| Figure 2 | `scripts/figure2.py` | PSQI restricted cubic spline and score distribution | None |
| Figure 3 | `scripts/figure3.py` | Eight joint-exposure profiles for sleep, stress, and clenching | None |
| Figure S1 | `scripts/figure_s1.py` | ROC curves, 95% CIs, and calibration | Table 3 |

`association_models.py` fits the shared logistic models for Figures 2 and 3 and Table S3. `export_tables.py` converts the analysis JSON files into CSV tables in manuscript order. Word manuscript editing and the graphical abstract are outside the scope of this analysis package.

## Analysis settings

- **Table 1:** Continuous variables are summarized as medians and interquartile ranges (IQRs); means and SDs are also saved in the JSON output. Mann–Whitney U tests are two-sided and asymptotic, with tie and continuity corrections. Binary variables use nonmissing denominators and Pearson chi-square tests; Fisher's exact test is used when an expected cell count is below 5.
- **Table 2:** Unpenalized logistic regression with HC0 sandwich covariance. Model 1 includes the four exposures, age, sex, log(1 + symptom duration), and initial-visit calendar quarter. Model 2 additionally includes VAS and bilateral pain. Predictors are standardized for numerical stability, and coefficients and covariance are then transformed back to their original units. Coefficients are checked using an independent scikit-learn solver.
- **Figure 2:** Three knots are placed at the 10th, 50th, and 90th percentiles of PSQI (2, 5, and 11 in the original data). Estimates are restricted to the observed PSQI range. Overall and nonlinear associations are assessed using robust Wald tests.
- **Figure 3 and Table S3:** Poor sleep is defined as PSQI >5. The model includes all two-way and three-way interactions among the three exposures. Each exposure combination is assigned to the same full complete-case sample while retaining observed covariates, and marginal probabilities are calculated. CIs use HC0 covariance and the delta method. CIs for probability differences account for the joint covariance of the two probabilities. Individual gradients are checked using finite differences.
- **Table 3:** Five outer and three inner stratified nested CV folds, with three candidate configurations per model. Baseline predictors are age, sex, log(1 + symptom duration), and visit quarter. Expanded models add PSQI, stress, clenching, and grinding. VAS and bilateral pain are not included in the baseline classification models. Median imputation, missingness indicators, scaling, one-hot encoding, and tuning are performed within training folds. The MLP is a tabular neural network with hidden layers of (32, 16).
- **Classification inference:** CIs for AUROC and its increment use 2,000 paired, outcome-stratified bootstrap samples. P values for the increment use 10,000 within-patient prediction swaps. Figure S1 ROC bands are pointwise 95% CIs, shown in the same color as the curves at 20% opacity. The joint calibration null hypothesis is intercept = 0 and slope = 1, assessed using an HC0 Wald test (2 df). This inference is conditional on the fitted models; models are not refitted in each bootstrap sample.
- **Table S2:** Fifty independent fully conditional specification (FCS) chains, each run for 70 iterations. Five-donor type-1 predictive mean matching is used for individual PSQI components, log-transformed duration, and VAS; logistic parameter draws and Bernoulli sampling are used for bilateral pain. Observed values are held fixed, and PSQI global scores are recalculated as the sum of the components. The final state of each chain forms one of the 50 imputed datasets. Pooling uses Rubin's rules, Barnard–Rubin degrees of freedom, and HC0 within-imputation covariance. Split R-hat for the final 20 iterations and coefficient Monte Carlo standard errors (MCSEs) are saved. The deterministic random-number-generator (RNG) stream switches at iteration 35 to preserve the original reproducibility settings. The missing-at-random (MAR) assumption cannot be verified by running the code.
- **Adult-only sensitivity analysis:** Complete cases aged ≥18 years (3,174 patients in the original data).

## P-value adjustment families

All adjustments apply the Benjamini–Hochberg false discovery rate (FDR) procedure to unrounded, two-sided P values. CIs are nominal 95% intervals and are not adjusted for multiplicity.

| Analysis | Tests within one FDR family |
| --- | --- |
| Table 1 / Figure 1 | The 12 comparisons in Table 1; Figure 1 uses the corresponding P values directly |
| Table 2 | Four exposures × two models = eight tests |
| Figures 2 and 3 | Overall PSQI association, PSQI nonlinearity, and joint-exposure omnibus interaction = three tests |
| Figure 3 panel-wise tests | Omnibus group tests at each clenching level = two tests, in a separate family |
| Table 3 | AUROC increments for the three models = three tests |
| Figure S1 calibration | Calibration tests for the three expanded models = three tests, in a separate family |
| Table S1 | The 13 comparisons between included and excluded patients |
| Table S2 | Four MI tests + four adult-only tests = eight tests; reference-model results retain the Table 2 adjustments |
| Table S3 | The three probability contrasts and the four interaction coefficients form separate families |

The omnibus interaction P value in Table S3 retains the adjustment from the existing three-test family used for Figure 3. Significance of an individual interaction coefficient must be distinguished from the omnibus result. Supplementary profile contrasts and panel-wise tests are exploratory analyses.

## Inputs and outputs

- Each patient must have one baseline row and a unique `Study_ID`. IDs are used for linkage and duplicate checks, not as predictors. Preserve the original row order to reproduce the classification results.
- Leave missing values as empty cells. Do not encode missingness as 0, -999, or other numeric placeholders. Binary coding and PSQI calculations are checked during input processing.
- `tables/`: Aggregate JSON and CSV files for Tables 1–3 and S1–S3. JSON files contain unrounded values, denominators, and analysis and verification information.
- `figures/`: PNG, PDF, and TIFF files numbered according to the manuscript, plus aggregate CSV files for curves, exposure profiles, and calibration. Figure 1 is also saved as SVG.
- `private/`: **Patient-level out-of-fold (OOF) predictions and MI checkpoints.** These are needed for internal reproduction but must not be publicly distributed or uploaded to the code repository.
- `logs/` and `run_manifest.json`: Execution logs, Python information, run times, and exit codes.
- `reference_results/`: Aggregate results used to prepare the manuscript. No patient-level data are included.
- `verification_report.json`: When `--verify` is used, this file records comparisons against the manuscript reference values, including sample sizes, ORs and CIs, raw and FDR-adjusted P values, profile probabilities, and classification metrics. Figure 1 reads P values directly from the verified Table 1 results.

Use `--verify` when reproducing this study's original cohort. The code explicitly checks the original sample sizes. To apply it to another cohort, adjust these checks and plot limits to the new study design. Pinned requirements are recommended because software-version differences or Excel-to-CSV conversion can affect numerical results. The study evaluates cross-sectional associations and internal classification; it does not establish causality or validate prediction of future onset.

## Files excluded from the package

Patient source data, lists containing names or hospital IDs, patient-level predictions, imputed datasets, and checkpoints are not included. Data must be obtained separately under the research institution's ethical and data-access conditions. This code reproduces numerical results and figures; it does not replace the PSQI questionnaire or the DC/TMD clinical diagnostic instruments.

## Reproducibility checks

The full analysis was run on the original data on 2026-10-04. Aggregate values for Tables 1–3 and S1–S3 and Figures 2, 3, and S1 matched the manuscript reference results within the verification tolerances. Figure 1 uses the same P values as Table 1. Baseline and expanded AUROCs for all three classification models and the four MI ORs exactly matched the stored reference values. The validation items are recorded in `validation_report.json`.
