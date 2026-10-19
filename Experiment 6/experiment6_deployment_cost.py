
import os
import sys
import json
import time
import platform
import argparse
import tempfile
import statistics
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# EXPERIMENT 6: REAL COST OF DEPLOYMENT
# ============================================================

SEED = 42
TEST_SIZE = 0.20
WARMUP_RUNS = 3
TIMING_REPEATS = 3
BATCH_SIZE = 32

OUTPUT_DIR = Path("experiment6_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. MODELS
# All models use the same text representation and test examples.
# Add/remove models here based on your earlier experiments.
# ------------------------------------------------------------

MODELS = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ]),

    "Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB(alpha=0.5)),
    ]),

    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("classifier", LinearSVC(
            class_weight="balanced",
            random_state=SEED,
        )),
    ]),

    "SGD Classifier": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
        ("classifier", SGDClassifier(
            loss="hinge",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=SEED,
        )),
    ]),
}


# ------------------------------------------------------------
# 2. DATA LOADING AND LABEL NORMALIZATION
# ------------------------------------------------------------

def normalize_label(value):
    """Return 1 for scam/spam and 0 for legitimate/ham."""
    if pd.isna(value):
        raise ValueError("Dataset contains a missing label.")

    text = str(value).strip().lower()

    scam_labels = {
        "1", "spam", "scam", "fraud", "phishing",
        "malicious", "smishing", "fraudulent", "true",
    }
    legitimate_labels = {
        "0", "ham", "legitimate", "legit", "safe",
        "benign", "normal", "not spam", "not scam", "false",
    }

    if text in scam_labels:
        return 1
    if text in legitimate_labels:
        return 0

    raise ValueError(
        f"Unknown label {value!r}. Edit normalize_label() "
        "to map your dataset's labels explicitly."
    )


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named 'text' and 'label'. "
            f"Found: {list(df.columns)}"
        )

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    df["label"] = df["label"].apply(normalize_label)

    # Remove repeated messages before splitting to reduce leakage.
    # If identical messages have conflicting labels, exclude them.
    conflicting = (
        df.groupby("text")["label"].nunique()
        .loc[lambda s: s > 1].index
    )
    if len(conflicting):
        print(f"Removing {len(conflicting)} texts with conflicting labels.")
        df = df[~df["text"].isin(conflicting)]

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"Removed {before - len(df)} duplicate rows.")

    if df["label"].nunique() != 2:
        raise ValueError("Both legitimate (0) and scam (1) classes are required.")

    counts = df["label"].value_counts()
    if counts.min() < 2:
        raise ValueError("Each class needs at least two samples.")

    return df.reset_index(drop=True)


# ------------------------------------------------------------
# 3. ADVERSARIAL INPUTS
# These are controlled text perturbations, not guaranteed
# to preserve the meaning of every individual message.
# ------------------------------------------------------------

def make_adversarial(text):
    """Apply deterministic, meaning-targeted formatting changes."""
    text = str(text)

    # Change case and add harmless spacing.
    transformed = "  " + text.swapcase() + "  "

    # Obfuscate common words/characters without changing labels.
    replacements = [
        ("free", "fr33"),
        ("urgent", "urg3nt"),
        ("account", "acc0unt"),
        ("verify", "ver1fy"),
        ("http", "hxxp"),
        ("www", "w w w"),
        ("win", "w1n"),
        ("claim", "cl41m"),
    ]

    lower = transformed.lower()
    for old, new in replacements:
        if old in lower:
            index = lower.find(old)
            transformed = (
                transformed[:index]
                + new
                + transformed[index + len(old):]
            )
            break

    return transformed


# ------------------------------------------------------------
# 4. METRICS
# ------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "scam_precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "scam_recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "scam_f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


# ------------------------------------------------------------
# 5. LATENCY, THROUGHPUT AND MEMORY
# ------------------------------------------------------------

def predict_in_batches(model, texts, batch_size=BATCH_SIZE):
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        predictions.extend(model.predict(batch))
    return np.asarray(predictions)


def measure_memory_mb():
    """Current process RSS; not the model's isolated memory."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def measure_latency(model, texts):
    """
    Per-message latency measured with single-message predictions.
    Warm-up is excluded from reported timings.
    """
    if len(texts) == 0:
        raise ValueError("Cannot benchmark an empty test set.")

    for _ in range(WARMUP_RUNS):
        model.predict([texts[0]])

    timings_ms = []

    for _ in range(TIMING_REPEATS):
        for text in texts:
            start = time.perf_counter_ns()
            model.predict([text])
            elapsed = time.perf_counter_ns() - start
            timings_ms.append(elapsed / 1_000_000)

    timings_ms = np.asarray(timings_ms)

    # Separate batch throughput measurement.
    start = time.perf_counter()
    predict_in_batches(model, texts)
    batch_elapsed = time.perf_counter() - start

    throughput = (
        len(texts) / batch_elapsed
        if batch_elapsed > 0 else float("nan")
    )

    return {
        "latency_p50_ms": float(np.percentile(timings_ms, 50)),
        "latency_p95_ms": float(np.percentile(timings_ms, 95)),
        "latency_mean_ms": float(np.mean(timings_ms)),
        "latency_std_ms": float(np.std(timings_ms, ddof=1))
        if len(timings_ms) > 1 else 0.0,
        "throughput_messages_per_second": float(throughput),
        "timing_repeats": TIMING_REPEATS,
        "warmup_runs": WARMUP_RUNS,
    }


def save_and_measure_model(model, model_name):
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    model_path = OUTPUT_DIR / f"{safe_name}.joblib"

    joblib.dump(model, model_path, compress=0)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    return model_path, size_mb


# ------------------------------------------------------------
# 6. HARDWARE AND SOFTWARE REPORT
# ------------------------------------------------------------

def environment_report():
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "Not reported",
        "logical_cpu_count": os.cpu_count(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "test_size_fraction": TEST_SIZE,
        "random_seed": SEED,
        "warmup_runs": WARMUP_RUNS,
        "timing_repeats": TIMING_REPEATS,
        "batch_size": BATCH_SIZE,
        "measurement_note": (
            "CPU/RAM environment reported by Python; no GPU acceleration "
            "is used by these scikit-learn models."
        ),
    }

    try:
        import psutil
        info["total_system_ram_gb"] = round(
            psutil.virtual_memory().total / (1024 ** 3), 2
        )
    except Exception:
        info["total_system_ram_gb"] = "Unavailable"

    with open(OUTPUT_DIR / "environment.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nENVIRONMENT")
    print(json.dumps(info, indent=2))


# ------------------------------------------------------------
# 7. MAIN EXPERIMENT
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ScamShield deployment efficiency experiment"
    )
    parser.add_argument(
        "--data", default="scam_dataset.csv",
        help="CSV containing text and label columns"
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"\nLoaded {len(df)} unique messages.")
    print("Label counts (0=legitimate, 1=scam):")
    print(df["label"].value_counts().sort_index())

    X = df["text"].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Same test examples for every model and condition.
    X_test_clean = list(X_test)
    X_test_adv = [make_adversarial(t) for t in X_test_clean]

    print(f"\nTraining messages: {len(X_train)}")
    print(f"Fixed test messages: {len(X_test)}")
    print(f"Adversarial test messages: {len(X_test_adv)}")

    environment_report()

    results = []
    predictions_for_plot = {}

    for model_name, base_model in MODELS.items():
        print("\n" + "=" * 65)
        print(f"MODEL: {model_name}")
        print("=" * 65)

        model = clone(base_model)

        rss_before = measure_memory_mb()
        model.fit(X_train, y_train)
        rss_after_training = measure_memory_mb()

        model_path, size_mb = save_and_measure_model(model, model_name)

        # The model artifact includes the fitted vectorizer and classifier.
        clean_predictions = predict_in_batches(model, X_test_clean)
        adv_predictions = predict_in_batches(model, X_test_adv)

        clean_metrics = calculate_metrics(y_test, clean_predictions)
        adv_metrics = calculate_metrics(y_test, adv_predictions)

        # Measure latency on a fixed, deterministic subset to limit runtime.
        timing_count = min(len(X_test_clean), 500)
        timing_texts = X_test_clean[:timing_count]
        timing_results = measure_latency(model, timing_texts)

        # Also report adversarial inference latency.
        adv_timing = measure_latency(
            model, X_test_adv[:timing_count]
        )

        # RSS is process-wide and includes Python, libraries and other models.
        # It is not an isolated peak-memory measurement for this model.
        row = {
            "model": model_name,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "clean_accuracy": clean_metrics["accuracy"],
            "clean_macro_f1": clean_metrics["macro_f1"],
            "clean_macro_precision": clean_metrics["macro_precision"],
            "clean_macro_recall": clean_metrics["macro_recall"],
            "clean_scam_precision": clean_metrics["scam_precision"],
            "clean_scam_recall": clean_metrics["scam_recall"],
            "clean_scam_f1": clean_metrics["scam_f1"],
            "adversarial_accuracy": adv_metrics["accuracy"],
            "adversarial_macro_f1": adv_metrics["macro_f1"],
            "adversarial_scam_recall": adv_metrics["scam_recall"],
            "adversarial_scam_f1": adv_metrics["scam_f1"],
            "macro_f1_drop": (
                clean_metrics["macro_f1"] - adv_metrics["macro_f1"]
            ),
            "latency_p50_ms": timing_results["latency_p50_ms"],
            "latency_p95_ms": timing_results["latency_p95_ms"],
            "latency_mean_ms": timing_results["latency_mean_ms"],
            "latency_std_ms": timing_results["latency_std_ms"],
            "throughput_messages_per_second":
                timing_results["throughput_messages_per_second"],
            "adversarial_latency_p50_ms":
                adv_timing["latency_p50_ms"],
            "adversarial_latency_p95_ms":
                adv_timing["latency_p95_ms"],
            "serialized_model_size_mb": size_mb,
            "rss_before_training_mb": rss_before,
            "rss_after_training_mb": rss_after_training,
            "rss_change_during_training_mb": (
                rss_after_training - rss_before
                if np.isfinite(rss_before) and np.isfinite(rss_after_training)
                else float("nan")
            ),
            "rss_after_benchmark_mb": measure_memory_mb(),
            "memory_measurement_note": (
                "Process-wide RSS snapshots; not isolated peak model memory."
            ),
            "timing_samples_per_condition": timing_count,
            "timing_repeats": TIMING_REPEATS,
            "warmup_runs": WARMUP_RUNS,
            "model_file": str(model_path),
        }

        results.append(row)
        predictions_for_plot[model_name] = row

        print("\nClean results:")
        print(json.dumps(clean_metrics, indent=2))
        print("\nAdversarial results:")
        print(json.dumps(adv_metrics, indent=2))
        print("\nLatency / deployment results:")
        print(json.dumps(timing_results, indent=2))
        print(f"Serialized model size: {size_mb:.4f} MB")

        # Save confusion matrices for the paper.
        for condition, truth, pred in [
            ("clean", y_test, clean_predictions),
            ("adversarial", y_test, adv_predictions),
        ]:
            cm = confusion_matrix(truth, pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["actual_legitimate", "actual_scam"],
                columns=["predicted_legitimate", "predicted_scam"],
            )
            cm_df.to_csv(
                OUTPUT_DIR / (
                    f"{model_name.lower().replace(' ', '_')}_"
                    f"{condition}_confusion_matrix.csv"
                )
            )

        # Store per-example outputs to enable error analysis.
        pd.DataFrame({
            "text_clean": X_test_clean,
            "text_adversarial": X_test_adv,
            "actual_label": y_test,
            "predicted_clean": clean_predictions,
            "predicted_adversarial": adv_predictions,
        }).to_csv(
            OUTPUT_DIR / (
                f"{model_name.lower().replace(' ', '_')}_predictions.csv"
            ),
            index=False,
        )

    # --------------------------------------------------------
    # 8. SAVE EFFICIENCY TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="clean_macro_f1", ascending=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "experiment6_model_efficiency.csv",
        index=False,
    )

    # Paper-friendly, compact table.
    compact_columns = [
        "model",
        "clean_macro_f1",
        "clean_scam_recall",
        "adversarial_macro_f1",
        "adversarial_scam_recall",
        "latency_p50_ms",
        "latency_p95_ms",
        "serialized_model_size_mb",
        "throughput_messages_per_second",
    ]

    results_df[compact_columns].to_csv(
        OUTPUT_DIR / "experiment6_paper_table.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9. PERFORMANCE VERSUS LATENCY PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    for _, row in results_df.iterrows():
        plt.scatter(
            row["latency_p50_ms"],
            row["clean_macro_f1"],
            s=90,
        )
        plt.annotate(
            row["model"],
            (
                row["latency_p50_ms"],
                row["clean_macro_f1"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Median single-message inference latency (ms)")
    plt.ylabel("Macro-F1 on clean test set")
    plt.title("ScamShield: Detection Performance vs Inference Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "experiment6_performance_vs_latency.png",
        dpi=300,
    )
    plt.close()

    # Plot clean vs adversarial macro-F1.
    positions = np.arange(len(results_df))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(
        positions - width / 2,
        results_df["clean_macro_f1"],
        width,
        label="Clean",
    )
    plt.bar(
        positions + width / 2,
        results_df["adversarial_macro_f1"],
        width,
        label="Adversarial",
    )
    plt.xticks(
        positions,
        results_df["model"],
        rotation=20,
        ha="right",
    )
    plt.ylabel("Macro-F1")
    plt.title("Clean vs Adversarial Detection Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "experiment6_clean_vs_adversarial.png",
        dpi=300,
    )
    plt.close()

    # Record reproducibility details.
    with open(
        OUTPUT_DIR / "experiment_protocol.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Experiment 6: Measure the Real Cost of Deployment\n"
            f"Dataset: {args.data}\n"
            f"Random seed: {SEED}\n"
            f"Test fraction: {TEST_SIZE}\n"
            f"Training samples: {len(X_train)}\n"
            f"Test samples: {len(X_test)}\n"
            f"Warm-up predictions per model: {WARMUP_RUNS}\n"
            f"Timing repeats: {TIMING_REPEATS}\n"
            f"Timing subset size: {min(len(X_test), 500)}\n"
            f"Batch size for throughput: {BATCH_SIZE}\n"
            "All models use the same stratified test set.\n"
            "Adversarial perturbations are deterministic and identical "
            "for all models.\n"
            "Latency is measured on CPU with single-message predictions; "
            "throughput uses batched predictions.\n"
            "Memory values are process-wide RSS snapshots, not isolated "
            "peak memory for each model.\n"
            "Serialized model size is measured from the saved joblib file.\n"
        )

    print("\n" + "=" * 65)
    print("EXPERIMENT 6 COMPLETED")
    print("=" * 65)
    print(f"Results folder: {OUTPUT_DIR.resolve()}")
    print("\nPaper table:")
    print(results_df[compact_columns].to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
