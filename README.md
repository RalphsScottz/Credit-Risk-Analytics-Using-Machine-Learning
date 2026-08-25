# Credit Risk Analytics with Machine Learning

An end-to-end credit risk modeling project that builds a **Probability of Default (PD) model** on the Lending Club consumer loan dataset (2007–2014), following the Basel II / IFRS 9 style workflow used in banking: data preparation → Weight of Evidence (WoE) / Information Value (IV) feature engineering → Logistic Regression PD model → model evaluation (AUC, Gini, KS) → credit scorecard.

<a href="https://colab.research.google.com/drive/1p_gX-ZG268JBnMS_9tuCiYT-TSltkbyU" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## Objective

Lenders need to estimate **Expected Loss (EL)** on a loan portfolio:

```
EL = PD × LGD × EAD
```

This project focuses on the **PD** component: given a borrower's application and credit-bureau attributes, predict the probability that the loan will default, and translate that probability into a business-friendly **credit scorecard** that non-technical stakeholders (e.g. loan officers) can use.

The notebook is written as a teaching-style walkthrough — it pairs each modeling step with the underlying theory (credit risk fundamentals, Basel II capital requirements, WoE/IV, logistic regression, ROC/AUC, Gini, KS statistic, and scorecard construction) before implementing it in code.

## Dataset

- **Source:** [Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data) (Kaggle), consumer loans issued 2007–2015 by Lending Club, a US peer-to-peer lender.
- **File used in this repo:** `loan_data_sample_1M.zip` → `loan_data_sample_1M.csv`, a sampled subset of the full dataset for lighter-weight experimentation.
- **Full working set used in the notebook:** `loan_data_2007_2014.csv` — **466,285 rows × 75 columns** of loan-level data (loan amount, interest rate, grade/sub-grade, employment length, home ownership, DTI, FICO range, revolving utilization, delinquency history, public records, etc.).
- A full data dictionary for all 75+ raw fields is included in the notebook.

**Target variable — `good_loan`:**

```python
bad_status_l = ['Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off',
                'Late (31-120 days)']
loan_data['good_loan'] = np.where(loan_data['loan_status'].isin(bad_status_l), 0, 1)
```

`0` = bad loan (default/charged-off/late), `1` = good loan (current/fully paid) — the notebook models `good_loan` as a binary classification target.

## Project Workflow

1. **Data ingestion** — download the Lending Club dataset via `kaggle`/`gdown`, load with `pandas`.
2. **Preprocessing**
   - Clean/convert string fields to numeric (`emp_length`, `term`).
   - Derive date-based tenure features (`mths_since_earliest_cr_line`, `mths_since_issue_d`).
   - Define the `good_loan` / `bad_loan` binary target from `loan_status`.
   - Handle categorical variables (grade, sub_grade, home_ownership, verification_status, purpose, addr_state, initial_list_status).
3. **Feature engineering — Weight of Evidence (WoE) & Information Value (IV)**
   - Custom `calculate_woe_discrete` / `calculate_woe_iv_discrete` functions bucket each categorical/discrete variable and compute WoE and IV to quantify predictive power ahead of modeling.
   - IV-based variable screening (weak / medium / strong predictors) and WoE visualization (`plot_by_woe`, `plot_by_woe_prop`).
4. **Dimensionality/exploration** — `StandardScaler` + `PCA` on the normalized numeric features, including a 2D PCA projection colored by good/bad loan for visual separability.
5. **Modeling — Logistic Regression (PD model)**
   - `train_test_split` (80/20, `random_state=42`).
   - `LogisticRegression(class_weight='balanced')` to correct for the imbalance between good and bad loans.
   - A custom `log_summary_table()` function replicates a statsmodels-style output (coefficients, p-values via the Fisher Information Matrix, odds ratios, confidence intervals) on top of scikit-learn's `LogisticRegression`.
   - Iterative refinement: statistically insignificant feature groups (p-value > 0.05) are dropped and the model is refit.
6. **Evaluation**
   - Accuracy, precision, recall, F1 (per class) via confusion matrices at multiple probability cutoffs (0.5, 0.4).
   - ROC curve, **AUROC**.
   - **Gini coefficient** (`2 × AUROC − 1`).
   - **Kolmogorov–Smirnov (KS) statistic**.
7. **Credit Scorecard**
   - Converts logistic regression coefficients into a points-based scorecard (attribute → category → points), and demonstrates single-applicant PD scoring by summing coefficients for the applicant's attribute categories and converting log-odds → probability.

## Results

| Metric | Value |
|---|---|
| Test Accuracy (cutoff 0.5) | **79.6%** |
| Test Accuracy (cutoff 0.4) | **85.0%** |
| AUROC | **0.870** |
| Gini coefficient | **0.740** |
| Kolmogorov–Smirnov (KS) statistic | **0.581** |

**At the default 0.5 cutoff** (final refined model, test set):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bad loan (0) | 0.322 | 0.780 | 0.456 |
| Good loan (1) | 0.967 | 0.798 | 0.875 |

The model recovers a large share of actual bad loans (recall ≈ 0.78–0.80 on the minority "bad" class) at the cost of flagging some good borrowers as risky — a deliberate trade-off from `class_weight='balanced'`, since in credit risk missing a defaulter is typically costlier than over-flagging a safe borrower. Raising the cutoff to 0.4 improves overall accuracy to 85% while keeping bad-loan recall around 0.70.

An AUROC of 0.87 and Gini of 0.74 indicate **strong discriminatory power**, comfortably in the "good model" range (AUC 0.8–0.9) used as an industry rule of thumb for PD models.

## Tech Stack

- **Language:** Python
- **Core libraries:** `pandas`, `numpy`
- **Modeling:** `scikit-learn` (`LogisticRegression`, `PCA`, `StandardScaler`, `Pipeline`, `ColumnTransformer`, `train_test_split`, `metrics`)
- **Stats:** `scipy.stats` (for coefficient p-values)
- **Visualization:** `matplotlib`, `seaborn`
- **Data acquisition:** `kaggle`, `gdown`
- **Environment:** Jupyter / Google Colab

## Repository Structure

```
.
├── Credit_Risk_Analytics_ML.ipynb   # Main notebook: theory + full pipeline
├── loan_data_sample_1M.zip          # Sampled dataset (1M-row CSV, zipped)
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn gdown kaggle

# Unzip the sample dataset
unzip loan_data_sample_1M.zip

# Launch the notebook
jupyter notebook Credit_Risk_Analytics_ML.ipynb
```

> To use the full 466K-row dataset instead of the 1M-row sample, download `loan_data_2007_2014.csv` from the Kaggle source linked above and point `loan_fn` in the notebook to that file.

## Key Concepts Covered

- Credit risk fundamentals, Expected Credit Loss (PD × LGD × EAD)
- Basel II Accord: Standardized, Foundation IRB, and Advanced IRB approaches
- Weight of Evidence (WoE) and Information Value (IV) for feature selection
- Class imbalance handling in classification (class weighting)
- Logistic regression for PD modeling and statistical inference (coefficients, p-values, odds ratios)
- Model evaluation for credit risk: ROC/AUC, Gini coefficient, Kolmogorov–Smirnov statistic
- Translating a statistical model into a business-usable credit scorecard

## License

Add a license of your choice (e.g. MIT) here.

## Acknowledgements

- Dataset: [Lending Club Loan Data on Kaggle](https://www.kaggle.com/wendykan/lending-club-loan-data)
