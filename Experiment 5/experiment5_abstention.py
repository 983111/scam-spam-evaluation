"""Experiment 5: uncertainty-aware scam detection with abstention.

Run: python experiment5_abstention.py --data your_dataset.csv --text-col text --label-col label
Labels must identify scam vs legitimate. The script maps common labels and numeric 1/0
(1=scam, 0=safe). Outputs are saved in experiment5_outputs/ and zipped.
"""
import argparse
import json
import random
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
OUT = Path("experiment5_outputs")


def normalize_label(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        if value == 1:
            return 1
        if value == 0:
            return 0
    s = str(value).strip().lower()
    scam = {"scam", "spam", "fraud", "phishing", "malicious", "1", "true", "yes"}
    safe = {"ham", "legitimate", "legit", "safe", "not spam", "not_spam",
            "not scam", "not_scam", "benign", "0", "false", "no", "normal"}
    if s in scam:
        return 1
    if s in safe:
        return 0
    return np.nan


def choose_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup), None)


def make_model(min_df, max_features):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=min_df, max_features=max_features, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED,
            solver="liblinear"
        )),
    ])


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def classification_metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_scam": float(precision_score(y, pred, zero_division=0)),
        "scam_recall": float(recall_score(y, pred, zero_division=0)),
        "f1_scam": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "roc_auc": safe_auc(y, p),
        "scam_pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def selective_metrics(y, probabilities, confidence_threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    automatic = confidence >= confidence_threshold
    n = len(y)
    n_auto = int(automatic.sum())
    n_review = n - n_auto
    if n_auto:
        selective_error = float(np.mean(pred[automatic] != y[automatic]))
        tn = int(np.sum((pred == 0) & (y == 0) & automatic))
        fp = int(np.sum((pred == 1) & (y == 0) & automatic))
        fpr = fp / (fp + tn) if fp + tn else np.nan
    else:
        selective_error, fpr = np.nan, np.nan
    total_scams = int(np.sum(y == 1))
    scam_auto_correct = int(np.sum((pred == 1) & (y == 1) & automatic))
    return {
        "confidence_threshold": confidence_threshold,
        "coverage": n_auto / n if n else np.nan,
        "automatically_classified": n_auto,
        "sent_for_review": n_review,
        "review_workload_pct": n_review / n if n else np.nan,
        "selective_error_rate": selective_error,
        "scam_recall_over_all_test_scams": scam_auto_correct / total_scams if total_scams else np.nan,
        "false_positive_rate_among_auto_safe_decisions": fpr,
    }


def perturb_punctuation(text):
    return re.sub(r"([!?.,:$%])", r" \1 ", text)


def perturb_whitespace(text):
    return " " + re.sub(r"\s+", "  ", text.strip()) + "  "


def perturb_obfuscation(text):
    replacements = [
        (r"\bfree\b", "fr ee"), (r"\bclaim\b", "cl aim"),
        (r"\bverify\b", "ver ify"), (r"\burgent\b", "ur gent"),
        (r"\baccount\b", "acc ount"), (r"\bprize\b", "pr1ze"),
        (r"\bclick\b", "cl1ck"), (r"\bwon\b", "w0n"),
    ]
    out = text
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run ScamShield Experiment 5 locally.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV file")
    parser.add_argument("--text-col", default=None, help="Text/message column name (auto-detect if omitted)")
    parser.add_argument("--label-col", default=None, help="Label column name (auto-detect if omitted)")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=100000)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(args.data)
    print(f"Loaded {args.data}: {len(df)} rows")
    print("Columns:", list(df.columns))

    text_col = args.text_col or choose_column(
        df.columns, ["text", "message", "sms", "body", "content", "message_text", "sms_text"]
    )
    label_col = args.label_col or choose_column(
        df.columns, ["label", "class", "target", "category", "is_scam", "is_spam", "y"]
    )
    if text_col is None or label_col is None:
        raise ValueError("Could not identify columns. Pass --text-col YOUR_TEXT_COLUMN --label-col YOUR_LABEL_COLUMN")

    data = df[[text_col, label_col]].copy()
    data.columns = ["text", "raw_label"]
    data = data.dropna(subset=["text", "raw_label"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0].copy()
    data["label"] = data["raw_label"].map(normalize_label)
    unknown = int(data["label"].isna().sum())
    if unknown:
        print(f"Dropping {unknown} rows with unrecognized labels.")
    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)

    # Remove conflicting duplicate texts and exact duplicate rows before splitting.
    conflicts = data.groupby("text")["label"].nunique()
    conflict_texts = set(conflicts[conflicts > 1].index)
    if conflict_texts:
        print(f"Removing {len(conflict_texts)} duplicated texts with conflicting labels.")
        data = data[~data["text"].isin(conflict_texts)].copy()
    before = len(data)
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Removed {before - len(data)} repeated text rows.")

    if data["label"].nunique() != 2:
        raise ValueError("Need both classes after cleaning. Labels must map to 0=safe and 1=scam.")
    if data["label"].value_counts().min() < 10:
        raise ValueError("Need at least 10 examples in each class for reliable stratified splits and calibration.")

    X, y = data["text"], data["label"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval
    )
    print(f"Split sizes: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

    standard_model = make_model(args.min_df, args.max_features)
    standard_model.fit(X_train, y_train)

    # Fit calibration on validation data without retraining the base estimator.
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(standard_model), method="sigmoid"
        )
        calibrated_model.fit(X_val, y_val)
        calibration_method = "sigmoid calibration on validation set using FrozenEstimator"
    except (ImportError, TypeError, AttributeError):
        calibrated_model = CalibratedClassifierCV(
            estimator=standard_model, method="sigmoid", cv="prefit"
        )
        calibrated_model.fit(X_val, y_val)
        calibration_method = "sigmoid calibration on validation set using cv=prefit compatibility mode"

    y_test_arr = y_test.reset_index(drop=True).to_numpy()
    X_test_reset = X_test.reset_index(drop=True)
    conditions = {
        "original": X_test_reset,
        "punctuation_spacing": X_test_reset.map(perturb_punctuation),
        "irregular_whitespace": X_test_reset.map(perturb_whitespace),
        "keyword_obfuscation": X_test_reset.map(perturb_obfuscation),
        "combined_perturbations": X_test_reset.map(
            lambda s: perturb_obfuscation(perturb_punctuation(perturb_whitespace(s)))
        ),
    }

    summary_rows, tradeoff_rows, adversarial_rows, prediction_rows, calibration_rows = [], [], [], [], []
    for condition, texts in conditions.items():
        p_standard = standard_model.predict_proba(texts)[:, 1]
        p_calibrated = calibrated_model.predict_proba(texts)[:, 1]
        for model_name, probs in [
            ("standard_logistic_regression", p_standard),
            ("calibrated_logistic_regression", p_calibrated),
        ]:
            row = {"condition": condition, "model": model_name, "threshold": 0.5}
            row.update(classification_metrics(y_test_arr, probs))
            adversarial_rows.append(row)
            if condition == "original":
                summary_rows.append(row.copy())
            calibration_rows.append({
                "condition": condition,
                "model": model_name,
                "brier_score": brier_score_loss(y_test_arr, probs),
            })

        confidence = np.maximum(p_calibrated, 1 - p_calibrated)
        for threshold in THRESHOLDS:
            row = {"condition": condition, "model": "calibrated_with_abstention"}
            row.update(selective_metrics(y_test_arr, p_calibrated, threshold))
            tradeoff_rows.append(row)

        for i, (text, truth, ps, pc, conf) in enumerate(zip(texts, y_test_arr, p_standard, p_calibrated, confidence)):
            prediction_rows.append({
                "test_row_index": i, "condition": condition, "text": text,
                "true_label_scam_1_safe_0": int(truth),
                "standard_scam_probability": float(ps),
                "standard_prediction_at_0.5": int(ps >= 0.5),
                "calibrated_scam_probability": float(pc),
                "calibrated_prediction_at_0.5": int(pc >= 0.5),
                "calibrated_confidence": float(conf),
                "abstains_at_0.70_confidence": bool(conf < 0.70),
                "abstains_at_0.80_confidence": bool(conf < 0.80),
                "abstains_at_0.90_confidence": bool(conf < 0.90),
            })

    summary_df = pd.DataFrame(summary_rows)
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    adversarial_df = pd.DataFrame(adversarial_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    summary_df.to_csv(OUT / "metrics_summary.csv", index=False)
    tradeoff_df.to_csv(OUT / "abstention_tradeoff.csv", index=False)
    adversarial_df.to_csv(OUT / "adversarial_metrics.csv", index=False)
    predictions_df.to_csv(OUT / "predictions_test.csv", index=False)
    calibration_df.to_csv(OUT / "calibration_comparison.csv", index=False)

    # Plot coverage vs selective error for each text condition.
    fig, ax = plt.subplots(figsize=(9, 6))
    for condition in conditions:
        part = tradeoff_df[(tradeoff_df.condition == condition) & tradeoff_df.selective_error_rate.notna()].sort_values("coverage")
        if not part.empty:
            ax.plot(part.coverage, part.selective_error_rate, marker="o", label=condition)
    ax.set_xlabel("Coverage: fraction classified automatically")
    ax.set_ylabel("Error rate among automatically classified messages")
    ax.set_title("Coverage versus selective error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "coverage_vs_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for condition in conditions:
        part = tradeoff_df[tradeoff_df.condition == condition].sort_values("confidence_threshold")
        ax.plot(part.confidence_threshold, part.review_workload_pct, marker="o", label=condition)
    ax.set_xlabel("Minimum confidence required for automatic classification")
    ax.set_ylabel("Review workload (fraction of messages)")
    ax.set_title("Manual review workload versus confidence threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "review_workload.png", dpi=200)
    plt.close(fig)

    pivot = calibration_df.pivot(index="condition", columns="model", values="brier_score")
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Probability calibration under text perturbations")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "calibration_comparison.png", dpi=200)
    plt.close(ax.figure)

    metadata = {
        "experiment": "Experiment 5: uncertainty-aware scam detection with abstention",
        "dataset": str(args.data), "text_column": str(text_col), "label_column": str(label_col),
        "label_mapping": {"0": "safe/legitimate", "1": "scam"},
        "rows_after_cleaning": int(len(data)), "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)), "test_rows": int(len(X_test)),
        "random_seed": SEED, "calibration_method": calibration_method,
        "thresholds": THRESHOLDS, "scikit_learn_version": sklearn.__version__,
        "limitations": [
            "The results are specific to the dataset and split used.",
            "Synthetic text perturbations are limited stress tests, not comprehensive real-world attacks.",
            "Some transformations may alter meaning; inspect examples before assuming labels remain valid.",
            "Do not select a production threshold based on test results; choose it on validation data, then evaluate once on test data.",
            "Calibration depends on the validation set being representative of future messages.",
        ],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== ORIGINAL TEST SET ===")
    cols = ["condition", "model", "accuracy", "precision_scam", "scam_recall", "f1_scam", "false_positive_rate", "roc_auc", "scam_pr_auc", "brier_score"]
    print(summary_df[cols].round(4).to_string(index=False))
    print("\n=== ABSTENTION TRADE-OFF: ORIGINAL TEST SET ===")
    cols = ["confidence_threshold", "coverage", "selective_error_rate", "scam_recall_over_all_test_scams", "false_positive_rate_among_auto_safe_decisions", "review_workload_pct"]
    print(tradeoff_df.loc[tradeoff_df.condition == "original", cols].round(4).to_string(index=False))
    print("\n=== ADVERSARIAL RESULTS ===")
    print(adversarial_df[["condition", "model", "accuracy", "scam_recall", "false_positive_rate", "brier_score"]].round(4).to_string(index=False))

    zip_path = Path("experiment5_outputs.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in OUT.iterdir():
            zf.write(file, arcname=f"{OUT.name}/{file.name}")
    print(f"\nOutputs saved to: {OUT.resolve()}")
    print(f"ZIP archive: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
