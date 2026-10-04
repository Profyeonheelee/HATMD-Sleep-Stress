# HATMD analysis code

**Sleep Quality, Stress, and Oral Parafunction in Headache Attributed to Temporomandibular Disorders**

This package reproduces the statistical analyses and figures for the retrospective cross-sectional HATMD study (IRB KH-DT25033). The original cohort includes 3,672 patients, 861 with HATMD. Adjusted association analyses use 3,319 complete cases; classification uses all 3,672 patients with preprocessing inside training folds.

## 실행 방법

Python 3.12 권장. 원자료는 포함하지 않았습니다. 원래 엑셀 파일의 **Data** 시트를 사용하거나, 같은 열과 환자 순서를 유지한 CSV를 사용하십시오. `data_dictionary.csv`는 변수 설명, `data_template.csv`는 데이터가 없는 입력 양식입니다.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_analysis.py --source /path/to/TMD_Sleep_Regularity_Final.xlsx --output results --verify
```

경로에 공백이 있으면 큰따옴표로 감싸십시오. 모든 분석은 위 명령 하나로 실행됩니다. 기존 파일을 덮어쓰지 않도록 `--output`에 새 폴더를 지정할 수 있습니다. 기본 출력 폴더는 `results`입니다.

주요 연관성과 Figure 1–3만 먼저 확인하려면 다음 명령을 사용하십시오. 이 모드는 Table 3, Table S2, Figure S1을 실행하지 않습니다.

```bash
python run_analysis.py --source /path/to/data.xlsx --output results_core --mode core --verify
```

개별 코드는 같은 `--source`, `--output` 옵션으로 실행할 수 있습니다. 예를 들어 Table 1을 실행한 다음 Table 2를 실행합니다.

```bash
python scripts/table1.py --source /path/to/data.xlsx --output results
python scripts/table2.py --source /path/to/data.xlsx --output results
```

## 원고와 코드의 대응

| 원고 항목 | 코드 | 내용 | 선행 실행 |
|---|---|---|---|
| Table 1 | `scripts/table1.py` | HATMD 유무에 따른 기술 통계, 원 P 및 FDR P | 없음 |
| Table 2 | `scripts/table2.py` | Model 1·2의 조정 OR, HC0 95% CI | Table 1 |
| Table 3 | `scripts/table3.py` | Nested CV: logistic regression, HGB, MLP | Table 2 |
| Table S1 | `scripts/table_s1.py` | 결측 현황, 포함·제외 환자 비교 | 없음 |
| Table S2 | `scripts/table_s2.py` | 다중 대체 및 성인만 포함한 민감도 분석 | Table 2 |
| Table S3 | `scripts/table_s3.py` | 공동 노출 확률 차이 및 상호작용 계수 | Figure 3 |
| Figure 1 | `scripts/figure1.py` | 군 간 분포: violin/scatter/box 및 비율 | Table 1 |
| Figure 2 | `scripts/figure2.py` | PSQI restricted cubic spline 및 점수 분포 | 없음 |
| Figure 3 | `scripts/figure3.py` | 수면·스트레스·이악물기 8개 공동 노출 | 없음 |
| Figure S1 | `scripts/figure_s1.py` | ROC, 95% CI, calibration | Table 3 |

`association_models.py`는 Figure 2·3과 Table S3의 공통 로지스틱 모형을 계산합니다. `export_tables.py`는 분석 JSON을 원고 순서의 CSV 표로 변환합니다. Word 원고 편집과 graphical abstract는 이 분석 코드의 범위에 포함되지 않습니다.

## 분석 설정

- **Table 1:** 연속 변수는 중앙값(IQR), 별도로 평균·SD를 JSON에 저장합니다. Mann–Whitney U 검정은 양측, 점근법, tie 및 continuity correction을 적용합니다. 이분 변수는 nonmissing 분모와 Pearson chi-square를 사용하며, 기대빈도 <5이면 Fisher exact test를 사용합니다.
- **Table 2:** 비벌점 로지스틱 회귀와 HC0 sandwich covariance. Model 1은 네 노출과 나이, 성별, log(1+증상 기간), 최초 방문 분기로 구성합니다. Model 2는 VAS와 양측성 통증을 추가합니다. 수치 안정성을 위해 표준화한 뒤 계수와 공분산을 원래 단위로 환산합니다. 독립된 scikit-learn solver로 계수를 확인합니다.
- **Figure 2:** PSQI의 10th/50th/90th percentile에서 3개 knot를 설정합니다(원자료: 2, 5, 11). 관찰된 PSQI 범위에서만 추정합니다. 전체·비선형 연관성은 robust Wald 검정입니다.
- **Figure 3 및 Table S3:** poor sleep은 PSQI >5입니다. 세 노출의 모든 2-way와 3-way interaction을 포함합니다. 각 노출 조합을 동일한 전체 complete-case 표본에 부여하고 공변량을 유지해 marginal probability를 계산합니다. CI는 HC0 covariance와 delta method를 사용합니다. 확률 차이의 CI는 두 확률의 공동 공분산을 사용합니다. 개별 gradient는 finite differences로 확인합니다.
- **Table 3:** 5 outer × 3 inner stratified nested CV; 3개 후보 설정/모형. Baseline은 나이, 성별, log(1+증상 기간), 방문 분기입니다. Expanded는 PSQI, 스트레스, 이악물기, 이갈이를 추가합니다. VAS와 양측 통증은 분류 모델의 baseline에 포함하지 않습니다. Median imputation, missing indicators, scaling, one-hot encoding과 tuning은 training folds 안에서 시행합니다. MLP는 hidden layers (32,16)의 tabular neural network입니다.
- **분류 추론:** AUROC와 증가량의 CI는 2,000 paired outcome-stratified bootstrap, 증가량 P는 10,000 within-patient prediction swaps입니다. Figure S1의 ROC band는 pointwise 95%, 같은 색의 20% opacity입니다. Calibration의 joint null은 intercept=0, slope=1이며 HC0 Wald(2 df)로 검정합니다. 이 추론은 이미 학습된 모델에 조건부이며 bootstrap마다 모델을 다시 학습하지 않습니다.
- **Table S2:** 50 independent FCS chains, chain당 70 iterations. PSQI 각 component, log-duration, VAS에 5-donor type-1 predictive mean matching, 양측 통증에 logistic parameter draws와 Bernoulli sampling을 사용합니다. 관찰값을 고정하고 PSQI global을 component 합으로 재계산합니다. 각 chain 마지막 값으로 50개 대체 자료를 구성합니다. Rubin pooling, Barnard–Rubin df와 HC0 within-imputation covariance를 사용합니다. 마지막 20 iterations의 split R-hat와 coefficient MCSE를 저장합니다. 원래 재현 설정을 유지하도록 iteration 35에서 deterministic RNG stream을 전환합니다. MAR 가정은 코드 실행으로 검증되지 않습니다.
- **성인 민감도 분석:** 나이 ≥18세 complete cases(원자료: 3,174명).

## P-value 보정 범위

모든 보정은 반올림 전 양측 P에 Benjamini–Hochberg FDR을 적용합니다. CI는 nominal 95%이며 multiplicity-adjusted CI가 아닙니다.

| 분석 | 하나의 FDR family |
|---|---|
| Table 1 / Figure 1 | Table 1의 12개 비교; Figure 1은 해당 P를 그대로 사용 |
| Table 2 | 4개 노출 × 2개 모델 = 8개 |
| Figure 2·3 | PSQI 전체, PSQI 비선형, 공동 노출 omnibus interaction = 3개 |
| Figure 3 panel-wise tests | 각 이악물기 수준의 omnibus group test = 2개, 별도 family |
| Table 3 | 3개 모델의 AUROC 증가량 = 3개 |
| Figure S1 calibration | 3개 expanded 모델의 calibration tests = 3개, 별도 family |
| Table S1 | 포함·제외 환자의 13개 비교 |
| Table S2 | MI 4개 + 성인 분석 4개 = 8개; 기준 모델은 Table 2 보정값 유지 |
| Table S3 | 3개 확률 차이와 4개 interaction 계수는 각각 별도 family |

Table S3의 omnibus interaction P는 Figure 3의 기존 3-test family 값을 그대로 사용합니다. 특정 interaction 계수의 유의성과 omnibus 결과를 구분해야 합니다. Supplementary profile contrasts와 panel-wise tests는 exploratory analyses입니다.

## 입력 및 출력

- 환자당 baseline 한 행, 고유한 `Study_ID`가 필요합니다. ID는 연결·중복 확인용이며 예측변수로 사용하지 않습니다. Classification 재현을 위해 원래 행 순서를 유지하십시오.
- 빈 셀은 결측으로 두십시오. 결측을 0, -999 등으로 입력하지 마십시오. Binary coding과 PSQI 계산식을 입력 단계에서 확인합니다.
- `tables/`: Table 1–3/S1–S3의 aggregate JSON 및 CSV. JSON에는 반올림 전 수치, 분모, 분석·검증 정보가 있습니다.
- `figures/`: 원고 번호별 PNG/PDF/TIFF 및 곡선·노출조합·calibration의 집계 CSV. Figure 1은 SVG도 저장합니다.
- `private/`: **환자별 OOF predictions와 MI checkpoint**. 내부 재현에 필요하지만 공개 배포하거나 코드 저장소에 올리지 마십시오.
- `logs/`, `run_manifest.json`: 실행 로그, Python 정보, 실행 시간과 종료 코드.
- `reference_results/`: 원고 작성에 사용한 집계 결과. 환자별 데이터는 없습니다.
- `verification_report.json`: `--verify`를 사용하면 원고 기준의 표본 수, OR/CI, raw/FDR P, profile probabilities, classification metrics 등을 비교한 결과를 저장합니다. Figure 1의 P 값은 검증된 Table 1을 직접 읽습니다.

`--verify`는 이 논문의 원래 cohort를 재현할 때 사용하십시오. 코드는 원래 표본 수를 명시적으로 확인합니다. 다른 cohort에 적용하려면 해당 확인 조건과 plot limits를 연구 설계에 맞게 조정해야 합니다. 다른 소프트웨어 버전이나 Excel→CSV 변환에서 수치가 달라질 수 있으므로 pinned requirements를 권장합니다. 연구 결과는 cross-sectional associations와 internal classification이며 인과성이나 미래 발생 예측을 검증하지 않습니다.

## 코드 묶음에 포함하지 않은 파일

환자 원자료, 실명·병원 ID 목록, 환자별 예측값, 대체 자료와 checkpoint는 포함하지 않았습니다. 데이터는 연구기관의 윤리·제공 조건에 따라 별도로 확보해야 합니다. 이 코드는 수치와 figure 재현용이며 PSQI 문항이나 DC/TMD 임상 진단 도구를 대체하지 않습니다.

## 재현 확인

2026-10-04 원자료로 전체 실행을 완료했습니다. Table 1–3/S1–S3, Figure 2·3/S1의 원고 기준 집계 수치가 검증 허용 오차 내에서 일치했고, Figure 1은 동일한 Table 1 P-value를 사용합니다. 세 분류 모형의 baseline/expanded AUROC와 MI의 네 OR는 저장된 기준값과 정확히 일치했습니다. 검증 항목은 `validation_report.json`에서 확인할 수 있습니다.
