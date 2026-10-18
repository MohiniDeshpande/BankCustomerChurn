"""Bank marketing: predict whether a client will subscribe to a term deposit.

Compares five classifiers (logistic regression, decision tree, random forest,
SVM, XGBoost) on a bank marketing dataset in the UCI Bank Marketing format.

Design choices that keep the evaluation honest:
  * The data is split into train and test sets BEFORE any resampling.
  * SMOTE sits inside the pipeline, so it only ever touches training folds
    during cross-validation and never sees the held-out test set.
  * Models are compared by cross-validated ROC-AUC on the training data.
    The test set is used once, at the end, to report performance.
  * The saved model is the fitted best estimator, including its preprocessing.
  * The `duration` column is dropped by default. It is only known after a call
    has finished, so a model that uses it cannot be used to decide whom to call.
  * The decision threshold is chosen from out-of-fold predictions on the TRAINING
    data, never from the test set. The test set then reports what that threshold
    achieves, next to the default 0.5, plus a lift table for ranking clients.

Usage:
    python bank_marketing_model.py --data bank.csv
"""

import argparse
import logging
from pathlib import Path

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "y"
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]


def load_data(file_path):
    """Load a CSV file, detecting the separator (comma or semicolon)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
    data.columns = data.columns.str.strip()
    return data


def preprocess_data(data, keep_duration=False):
    """Encode the target and features. Returns (X, y)."""
    data = data.copy()

    if TARGET not in data.columns:
        raise KeyError(f"Target column '{TARGET}' not found in the data.")

    if not keep_duration and "duration" in data.columns:
        data = data.drop(columns="duration")
        logger.info("Dropped 'duration' (known only after the call, so it leaks the outcome).")

    # In this dataset pdays == 999 means "not previously contacted".
    # Keep that information as its own flag instead of treating 999 as a number of days.
    if "pdays" in data.columns:
        data["previously_contacted"] = (data["pdays"] != 999).astype(int)
        data["pdays"] = data["pdays"].replace(999, -1)

    y = data[TARGET].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Target column contains values other than 'yes' and 'no'.")

    X = data.drop(columns=TARGET)
    categorical = [col for col in CATEGORICAL_COLS if col in X.columns]
    X = pd.get_dummies(X, columns=categorical, drop_first=True).astype(float)
    return X, y


def build_models(skip_svm=False):
    """Return {name: (estimator, parameter grid)}. Grid keys use the 'classifier__' prefix."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"classifier__C": [0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"classifier__max_depth": [3, 5, 10, None]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_leaf": [1, 2],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {},  # left at defaults: SVM is slow on large data
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
            },
        ),
    }
    if skip_svm:
        models.pop("SVM")
    return models


def make_pipeline(estimator):
    """SMOTE and scaling live inside the pipeline so they are fitted on training data only."""
    return Pipeline(
        [
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def evaluate_model(model, X_test, y_test):
    """Metrics on the held-out test set. Accuracy alone is misleading with imbalanced classes."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_pr_auc": average_precision_score(y_test, y_proba),
    }


def choose_threshold(y_true, proba, call_cost=None, subscriber_value=None):
    """Pick a decision threshold from out-of-fold predictions on the training data.

    By default the threshold maximises F1 for the "yes" class. If both a cost per
    call and a value per subscriber are given, it maximises expected net value instead.
    """
    y_true = np.asarray(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds

    if call_cost is not None and subscriber_value is not None:
        true_positives = recall * y_true.sum()
        contacted = true_positives / np.clip(precision, 1e-12, None)
        score = true_positives * subscriber_value - contacted * call_cost
        strategy = "expected net value"
    else:
        score = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        strategy = "F1"

    best = int(np.argmax(score))
    return float(thresholds[best]), strategy, precision, recall, thresholds


def metrics_at_threshold(y_true, proba, threshold, call_cost=None, subscriber_value=None):
    """Confusion matrix based metrics for a given decision threshold."""
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row = {
        "threshold": round(float(threshold), 4),
        "clients_contacted": int(pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "missed_subscribers": int(fn),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }
    if call_cost is not None and subscriber_value is not None:
        row["net_value"] = tp * subscriber_value - pred.sum() * call_cost
    return row


def build_lift_table(y_true, proba, steps=10):
    """Rank clients by predicted probability and report how many subscribers each
    contact depth captures. Lift 2.0 means twice as many subscribers as random calling."""
    y_true = np.asarray(y_true)
    y_sorted = y_true[np.argsort(-proba)]
    n, total_positives = len(y_sorted), y_sorted.sum()
    rows = []
    for k in range(1, steps + 1):
        cutoff = int(round(n * k / steps))
        captured = int(y_sorted[:cutoff].sum())
        share = cutoff / n
        rows.append(
            {
                "share_contacted": round(share, 2),
                "clients_contacted": cutoff,
                "subscribers_captured": captured,
                "share_of_subscribers_captured": captured / total_positives,
                "precision": captured / cutoff,
                "lift": (captured / total_positives) / share,
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_and_gains(precision, recall, thresholds, chosen, lift, path):
    """Left: precision-recall curve from training folds with the chosen threshold.
    Right: cumulative gains on the test set against random calling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(recall, precision, label="out-of-fold, training data")
    idx = int(np.argmin(np.abs(thresholds - chosen)))
    ax1.scatter([recall[idx]], [precision[idx]], color="red", zorder=3,
                label=f"chosen threshold {chosen:.2f}")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-recall trade-off")
    ax1.legend()

    x = np.concatenate([[0], lift["share_contacted"].to_numpy()])
    y = np.concatenate([[0], lift["share_of_subscribers_captured"].to_numpy()])
    ax2.plot(x, y, marker="o", label="model")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random calling")
    ax2.set_xlabel("Share of clients contacted")
    ax2.set_ylabel("Share of subscribers captured")
    ax2.set_title("Cumulative gains on the test set")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data)
    logger.info("Loaded data with shape %s", data.shape)

    X, y = preprocess_data(data, keep_duration=args.keep_duration)
    logger.info("Features: %d columns. Positive class rate: %.3f", X.shape[1], y.mean())

    # Split FIRST. Nothing below is allowed to see the test set until the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fitted = {}
    for name, (estimator, param_grid) in build_models(args.skip_svm).items():
        logger.info("Tuning and training %s ...", name)
        search = GridSearchCV(
            make_pipeline(estimator),
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_

        row = {"model": name, "cv_roc_auc": search.best_score_}
        row.update(evaluate_model(search.best_estimator_, X_test, y_test))
        row["best_params"] = str(search.best_params_)
        rows.append(row)
        logger.info("%s: CV ROC-AUC %.3f, test ROC-AUC %.3f",
                    name, row["cv_roc_auc"], row["test_roc_auc"])

    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    # Plot test-set results for comparison.
    ax = results.set_index("model")[["test_roc_auc", "test_f1"]].plot(
        kind="bar", figsize=(10, 6)
    )
    ax.set_title("Model comparison on the held-out test set")
    ax.set_ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()

    # Pick the best model by cross-validated ROC-AUC, not by looking at the test set.
    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    logger.info("Best model by cross-validated ROC-AUC: %s", best_name)

    y_pred = best_model.predict(X_test)
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["no", "yes"])
    plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


    # Threshold tuning and lift. The threshold comes from out-of-fold predictions on the
    # TRAINING data only. The test set just reports what that choice achieves.
    oof_proba = cross_val_predict(
        clone(best_model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold, strategy, precision_curve, recall_curve, thresholds = choose_threshold(
        y_train, oof_proba, args.call_cost, args.subscriber_value
    )
    test_proba = best_model.predict_proba(X_test)[:, 1]

    comparison = pd.DataFrame(
        [
            {"setting": "default 0.5",
             **metrics_at_threshold(y_test, test_proba, 0.5, args.call_cost, args.subscriber_value)},
            {"setting": f"tuned by {strategy}",
             **metrics_at_threshold(y_test, test_proba, threshold, args.call_cost, args.subscriber_value)},
        ]
    )
    comparison.to_csv(output_dir / "threshold_comparison.csv", index=False)

    lift = build_lift_table(y_test, test_proba)
    lift.to_csv(output_dir / "lift_table.csv", index=False)
    plot_threshold_and_gains(precision_curve, recall_curve, thresholds, threshold,
                             lift, output_dir / "threshold_and_gains.png")

    with open(output_dir / "decision_threshold.json", "w") as handle:
        json.dump({"model": best_name, "threshold": threshold, "chosen_by": strategy}, handle, indent=2)

    print(f"\nDecision threshold chosen by {strategy} on training folds: {threshold:.3f}")
    print(comparison.round(3).to_string(index=False))
    print("\nLift on the test set (clients ranked by predicted probability):")
    print(lift.round(3).to_string(index=False))

    # Save the fitted best pipeline (SMOTE is skipped at prediction time automatically).
    model_path = output_dir / f"best_model_{best_name.replace(' ', '_').lower()}.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Saved fitted model to %s", model_path)

    print("\nResults summary:")
    print(results.drop(columns="best_params").round(3).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Bank marketing prediction model comparison.")
    parser.add_argument("--data", default="bank.csv", help="Path to the CSV file.")
    parser.add_argument("--output-dir", default="results", help="Folder for results and the saved model.")
    parser.add_argument("--keep-duration", action="store_true",
                        help="Keep the 'duration' column (leaks the outcome; not recommended).")
    parser.add_argument("--skip-svm", action="store_true",
                        help="Skip the SVM, which is slow on large datasets.")
    parser.add_argument("--call-cost", type=float, default=None,
                        help="Cost of contacting one client. With --subscriber-value, the threshold "
                             "maximises expected net value instead of F1.")
    parser.add_argument("--subscriber-value", type=float, default=None,
                        help="Value of one client who subscribes.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
