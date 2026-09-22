"""
Train, compare, tune and save the churn prediction pipeline.

Run with:  python3 src/train.py
Produces:  models/churn_pipeline.pkl
           models/model_comparison.csv
           models/tuning_results.json
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import (
    build_preprocessor,
    clean_dataset,
    load_raw_dataset,
    split_features_target,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "primary" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def main():
    # 1-2. Load + clean the real dataset (no synthetic/fabricated data)
    raw = load_raw_dataset(DATA_PATH)
    clean = clean_dataset(raw)
    X, y = split_features_target(clean)

    print(f"Dataset shape after cleaning: {clean.shape}")
    print(f"Class balance -> No: {(y==0).sum()}  Yes: {(y==1).sum()}")

    # 3. Train/test split (stratified because churn is imbalanced ~73/27)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Candidate models. class_weight="balanced" is used where supported
    # because plain accuracy is a poor guide on a ~73/27 imbalanced target.
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=6
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", estimator)]
        )
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name}: {metrics}")

    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv")
    print("\nModel comparison:\n", comparison_df)

    # 5. Model selection: prioritise F1/Recall on the minority (churn) class
    # over raw accuracy, since missing a churner is the costlier error for
    # a retention use case. Rank by F1, use ROC-AUC as a tiebreaker.
    ranked = comparison_df.sort_values(by=["F1", "ROC-AUC"], ascending=False)
    best_name = ranked.index[0]
    print(f"\nSelected for tuning: {best_name}")

    # 6. Hyperparameter tuning on the selected model with GridSearchCV
    pre_tuning_metrics = results[best_name]

    if best_name == "XGBoost":
        base_estimator = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
        param_grid = {
            "model__n_estimators": [200, 300],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.05, 0.1],
        }
    elif best_name == "Random Forest":
        base_estimator = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )
        param_grid = {
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 2, 4],
        }
    elif best_name == "Decision Tree":
        base_estimator = DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )
        param_grid = {
            "model__max_depth": [4, 6, 8, 10],
            "model__min_samples_leaf": [1, 5, 10],
        }
    else:  # Logistic Regression
        base_estimator = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
        }

    tuning_pipe = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", base_estimator)]
    )

    grid = GridSearchCV(
        tuning_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    tuned_pipe = grid.best_estimator_
    post_tuning_metrics = evaluate(tuned_pipe, X_test, y_test)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Before tuning: {pre_tuning_metrics}")
    print(f"After tuning:  {post_tuning_metrics}")

    tuning_summary = {
        "selected_model": best_name,
        "best_params": grid.best_params_,
        "before_tuning": pre_tuning_metrics,
        "after_tuning": post_tuning_metrics,
    }
    with open(MODELS_DIR / "tuning_results.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)

    # 7. Final model = tuned pipeline if it does not regress F1; otherwise
    # keep the untuned version. This is a real comparison, not automatic.
    if post_tuning_metrics["F1"] >= pre_tuning_metrics["F1"]:
        final_pipe = tuned_pipe
        final_metrics = post_tuning_metrics
        print("\nUsing TUNED pipeline as final model.")
    else:
        final_pipe = fitted_pipelines[best_name]
        final_metrics = pre_tuning_metrics
        print("\nTuning did not improve F1; keeping UNTUNED pipeline as final model.")

    # 8. Persist the full pipeline (preprocessing + model together)
    joblib.dump(final_pipe, MODELS_DIR / "churn_pipeline.pkl")

    final_summary = {
        "final_model": best_name,
        "final_metrics": final_metrics,
        "feature_columns": list(X.columns),
    }
    with open(MODELS_DIR / "final_model_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\nSaved final pipeline to {MODELS_DIR / 'churn_pipeline.pkl'}")
    print(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
