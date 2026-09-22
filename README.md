# Customer Churn Prediction Using Machine Learning

**Team:** Kothuri Goutham (1601-24-737-314), Muggu Yashas (1601-24-737-319)

An end-to-end machine learning project that predicts telecom customer churn and serves the
prediction through a Streamlit web application.

## What's in this project

1. **A Google Colab / Jupyter notebook** (`notebooks/churn_analysis.ipynb`) covering the full
   ML development process: loading the real Telco Customer Churn dataset, cleaning it, EDA,
   feature engineering, feature selection, training and comparing 4 models, hyperparameter
   tuning, SHAP explainability, and saving the final pipeline. Every number in the notebook is
   produced by actually running the cells — nothing is hard-coded.
2. **A Streamlit web application** (`app/app.py`) where a user enters customer details and
   gets a live churn prediction, probability, risk level, and a SHAP-based explanation of
   *why* the model predicted that.

## Dataset

- **Source:** Telco Customer Churn dataset (the standard IBM sample dataset for this problem).
- **Size:** 7,043 rows, 21 original columns (20 features + `Churn` target, plus `customerID`).
- **Target:** `Churn` (Yes/No). Class balance: ~73.5% No, ~26.5% Yes (imbalanced).
- **Cleaning:** `TotalCharges` is stored as text and has 11 blank values, all belonging to
  customers with `tenure == 0` (new customers not yet billed) — these are filled with 0.
  `customerID` is dropped as a non-predictive identifier. No duplicate rows were found.

See `data/external/README.md` for why no external dataset was used for generalization testing.

## Project structure

```
customer-churn-prediction/
├── data/
│   ├── primary/telco_churn.csv       # real Telco Customer Churn dataset
│   └── external/README.md            # compatibility note (see above)
├── notebooks/
│   └── churn_analysis.ipynb          # full ML development notebook, already executed
├── src/
│   ├── preprocessing.py              # shared cleaning + ColumnTransformer (train & app)
│   ├── train.py                      # trains, compares, tunes, saves the pipeline
│   ├── explain.py                    # SHAP global + per-prediction explanations
│   └── predict.py                    # small prediction helper
├── models/
│   ├── churn_pipeline.pkl            # final saved preprocessing+model pipeline
│   ├── model_comparison.csv          # measured metrics for all 4 candidate models
│   ├── tuning_results.json           # before/after tuning metrics + best params
│   └── final_model_summary.json      # which model was selected and why
├── app/
│   └── app.py                        # Streamlit application
├── assets/
│   └── eda_overview.png              # EDA charts generated from the real data
├── requirements.txt
└── README.md
```

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Re)train the model (optional — a trained pipeline is already included)
```bash
cd src
python3 train.py
```
This regenerates `models/churn_pipeline.pkl`, `models/model_comparison.csv` and
`models/tuning_results.json` from scratch.

### 3. Run the Streamlit app
```bash
streamlit run app/app.py
```
The app loads the saved pipeline — it does not retrain the model when it starts.

## Results (measured on a held-out 20% test split, `random_state=42`)

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|---------:|----------:|-------:|------:|--------:|
| Logistic Regression |   0.738  |     0.504 |  0.783 | 0.614 |   0.842 |
| Decision Tree       |   0.740  |     0.507 |  0.789 | 0.617 |   0.828 |
| Random Forest       |   0.786  |     0.629 |  0.471 | 0.538 |   0.822 |
| XGBoost             |   0.772  |     0.568 |  0.588 | 0.578 |   0.809 |

**Why not just pick the highest-accuracy model?** Because churn is imbalanced (~73%/27%), a
model that predicts "No churn" for everyone would already score ~73% accuracy while missing
every real churner. Random Forest has the best accuracy here but the worst recall — it misses
the most actual churners, which defeats the purpose of a retention tool. The **Decision Tree**
was selected because it has the best F1-score, balancing precision and recall on the churn
class; Logistic Regression is a close second and has the best ROC-AUC, which is noted as a
legitimate trade-off rather than hidden.

Hyperparameter tuning (`GridSearchCV`, 5-fold CV, scored on F1) did not improve on the
untuned Decision Tree's F1 in this run, so the untuned model was kept as final — this is
reported honestly in `models/tuning_results.json` rather than forcing a "tuning improved
everything" narrative.

## Key EDA findings (from the real dataset)

- Month-to-month customers churn at ~42.7%, vs ~11.3% for one-year and ~2.8% for two-year
  contracts.
- Customers who churn have a lower average tenure (~18.0 months) than those who stay
  (~37.6 months).
- Electronic check payers churn at ~45.3%, much higher than the other three payment methods
  (15–19%).
- Fiber optic customers churn at ~41.9%, compared with ~19.0% for DSL and ~7.4% for customers
  with no internet service.

## Notes on academic use

This is a college-level case study project. The dataset, code, metrics and charts in this
repository are produced by actually running the included code against the real Telco Customer
Churn dataset — no results are invented or hard-coded. Students using this as a reference
should run the notebook and app themselves and write their own analysis/discussion of the
results in their own words for submission.
