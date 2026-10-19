
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


SEED = 42
MAX_MESSAGES_PER_CLASS = 50
MAX_WORDS_TO_EXPLAIN = 24
TOP_K_FEATURES = 10

OUTPUT = Path("experiment7_results")
OUTPUT.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. TEXT TRANSFORMATIONS
# ---------------------------------------------------------

def formatting_variant(text):
    """Formatting-only change: case and extra whitespace."""
    return "  " + re.sub(r"\s+", "   ", text.swapcase()).strip() + "  "


def punctuation_variant(text):
    """Formatting-only change: add harmless outer whitespace."""
    return " \t" + text.strip() + " \n"


# Candidate paraphrases. These are NOT guaranteed to preserve
# meaning in every context. Review the exported variants manually.
PHRASE_REPLACEMENTS = [
    (r"\bplease\b", "kindly"),
    (r"\bkindly\b", "please"),
    (r"\bcall me\b", "contact me"),
    (r"\bcontact me\b", "call me"),
    (r"\bclick here\b", "open this link"),
    (r"\bopen this link\b", "click here"),
    (r"\bverify your account\b", "confirm your account"),
    (r"\bconfirm your account\b", "verify your account"),
    (r"\bclaim your prize\b", "collect your prize"),
    (r"\bcollect your prize\b", "claim your prize"),
]


def candidate_paraphrase(text):
    """Apply at most one predefined lexical transformation."""
    for pattern, replacement in PHRASE_REPLACEMENTS:
        changed, count = re.subn(
            pattern, replacement, text,
            count=1, flags=re.IGNORECASE
        )
        if count:
            return changed, pattern, replacement

    # If no phrase matched, return the original unchanged.
    return text, "", ""


def make_variants(text):
    paraphrase, pattern, replacement = candidate_paraphrase(text)

    return [
        {
            "condition": "formatting_case_whitespace",
            "text": formatting_variant(text),
            "transformation": "Case change and extra whitespace",
            "candidate_paraphrase": False,
        },
        {
            "condition": "formatting_whitespace",
            "text": punctuation_variant(text),
            "transformation": "Outer whitespace",
            "candidate_paraphrase": False,
        },
        {
            "condition": "candidate_paraphrase",
            "text": paraphrase,
            "transformation": (
                f"{pattern} -> {replacement}" if pattern else "No rule matched"
            ),
            "candidate_paraphrase": True,
        },
    ]


# ---------------------------------------------------------
# 2. LOAD EXISTING MODEL AND TEST PREDICTIONS
# ---------------------------------------------------------

def load_inputs(model_path, predictions_path):
    model = joblib.load(model_path)
    df = pd.read_csv(predictions_path)

    required = {
        "text_clean",
        "actual_label",
        "predicted_clean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["text_clean", "actual_label", "predicted_clean"]
    ).copy()

    df["text_clean"] = df["text_clean"].astype(str)
    df["actual_label"] = df["actual_label"].astype(int)
    df["predicted_clean"] = df["predicted_clean"].astype(int)

    # Use only correctly classified messages, as specified.
    df = df[
        df["actual_label"] == df["predicted_clean"]
    ].copy()

    if df.empty:
        raise ValueError("No correctly classified messages were found.")

    df = df.drop_duplicates(subset=["text_clean"])

    selected = []
    for label in [0, 1]:
        class_df = df[df["actual_label"] == label]
        n = min(MAX_MESSAGES_PER_CLASS, len(class_df))
        selected.append(
            class_df.sample(n=n, random_state=SEED)
        )

    selected_df = pd.concat(selected).reset_index(drop=True)

    if selected_df["actual_label"].nunique() < 2:
        raise ValueError(
            "The selected examples do not contain both classes. "
            "Check your predictions file."
        )

    return model, selected_df


# ---------------------------------------------------------
# 3. MODEL SCORES
# ---------------------------------------------------------

def predicted_class_score(model, text):
    """
    Signed decision margin for the predicted class.
    Positive values support the selected class.
    """
    pred = int(model.predict([text])[0])

    if hasattr(model, "decision_function"):
        score = float(np.asarray(
            model.decision_function([text])
        ).ravel()[0])

        classes = list(model.classes_)
        if len(classes) == 2:
            # sklearn's binary decision_function is positive
            # for classes_[1].
            return score if pred == classes[1] else -score

        return score

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        class_index = list(model.classes_).index(pred)
        return float(probabilities[class_index])

    raise TypeError(
        "Model must implement decision_function or predict_proba."
    )


# ---------------------------------------------------------
# 4. OCCLUSION-BASED FEATURE ATTRIBUTION
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def explain_text(model, text):
    """
    Estimate word importance by removing one word at a time.
    Attribution = original predicted-class score minus the score
    after removing the word.

    Positive attribution supports the original predicted class.
    Negative attribution means removing the word increases support
    for the original class.
    """
    text = str(text)
    predicted = int(model.predict([text])[0])
    original_score = predicted_class_score(model, text)

    matches = list(TOKEN_PATTERN.finditer(text))

    # Preserve unique words in order of first occurrence.
    words = []
    seen = set()
    for match in matches:
        word = match.group()
        key = word.lower()
        if key not in seen:
            seen.add(key)
            words.append(word)

    words = words[:MAX_WORDS_TO_EXPLAIN]
    attributions = []

    for word in words:
        # Remove this word as a whole token, case-insensitively.
        pattern = re.compile(
            r"(?<!\w)" + re.escape(word) + r"(?!\w)",
            flags=re.IGNORECASE,
        )
        modified = pattern.sub(" ", text)
        modified = re.sub(r"\s+", " ", modified).strip()

        if not modified:
            modified = " "

        modified_score = predicted_class_score(model, modified)
        contribution = original_score - modified_score

        attributions.append({
            "feature": word.lower(),
            "attribution": float(contribution),
        })

    # Consolidate repeated word forms, retaining the largest
    # absolute attribution for each feature.
    combined = {}
    for item in attributions:
        feature = item["feature"]
        value = item["attribution"]

        if (
            feature not in combined
            or abs(value) > abs(combined[feature])
        ):
            combined[feature] = value

    ordered = sorted(
        combined.items(),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    return {
        "predicted_label": predicted,
        "score": original_score,
        "attributions": [
            {"feature": feature, "attribution": value}
            for feature, value in ordered
        ],
        "top_features": [
            feature for feature, _ in ordered[:TOP_K_FEATURES]
        ],
    }


# ---------------------------------------------------------
# 5. EXPLANATION STABILITY METRICS
# ---------------------------------------------------------

def top_k_jaccard(features_a, features_b):
    a, b = set(features_a), set(features_b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def top_k_overlap(features_a, features_b):
    """Overlap divided by k; complements Jaccard similarity."""
    a, b = set(features_a), set(features_b)
    denominator = max(1, min(len(a), len(b)))
    return len(a & b) / denominator


def rank_biased_similarity(features_a, features_b):
    """
    Rank-weighted overlap. Higher-ranked features contribute more.
    This is a simple descriptive score, not a formal significance test.
    """
    if not features_a or not features_b:
        return 0.0

    rank_a = {f: i + 1 for i, f in enumerate(features_a)}
    rank_b = {f: i + 1 for i, f in enumerate(features_b)}

    shared = set(rank_a) & set(rank_b)
    if not shared:
        return 0.0

    scores = []
    for feature in shared:
        rank_difference = abs(rank_a[feature] - rank_b[feature])
        scores.append(1.0 / (1.0 + rank_difference))

    return float(np.mean(scores))


# ---------------------------------------------------------
# 6. PERFORMANCE METRICS
# ---------------------------------------------------------

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "scam_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "scam_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
    }


# ---------------------------------------------------------
# 7. FEATURE-IMPORTANCE PLOT
# ---------------------------------------------------------

def save_feature_plot(attribution_records, filename, title):
    totals = {}

    for record in attribution_records:
        for item in record["attributions"]:
            feature = item["feature"]
            value = abs(item["attribution"])
            totals[feature] = totals.get(feature, 0.0) + value

    top = sorted(
        totals.items(), key=lambda pair: pair[1], reverse=True
    )[:20]

    if not top:
        return

    plot_df = pd.DataFrame(top, columns=["feature", "importance"])
    plot_df = plot_df.sort_values("importance")

    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Sum of absolute occlusion attribution")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=300)
    plt.close()


# ---------------------------------------------------------
# 8. MAIN EXPERIMENT
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="linear_svm.joblib"
    )
    parser.add_argument(
        "--predictions", default="linear_svm_predictions.csv"
    )
    args = parser.parse_args()

    model, selected = load_inputs(
        args.model, args.predictions
    )

    print(f"Selected correctly classified messages: {len(selected)}")
    print("Class counts:")
    print(selected["actual_label"].value_counts().sort_index())
    print("Label convention: 0=legitimate, 1=scam")

    records = []
    explanation_records = []
    all_original_y = []
    all_original_pred = []
    variant_performance = {}

    for index, row in selected.iterrows():
        original = row["text_clean"]
        actual = int(row["actual_label"])

        original_explanation = explain_text(model, original)
        original_prediction = original_explanation["predicted_label"]

        all_original_y.append(actual)
        all_original_pred.append(original_prediction)

        base_record = {
            "message_id": int(index),
            "actual_label": actual,
            "original_text": original,
            "original_prediction": original_prediction,
            "original_score": original_explanation["score"],
            "original_top_features": ", ".join(
                original_explanation["top_features"]
            ),
        }

        # Save original feature attribution for plotting.
        explanation_records.append({
            "condition": "original",
            "actual_label": actual,
            "text": original,
            "attributions": original_explanation["attributions"],
        })

        variants = make_variants(original)

        for variant in variants:
            changed_text = variant["text"]
            changed_explanation = explain_text(model, changed_text)
            changed_prediction = changed_explanation["predicted_label"]

            unchanged = changed_text == original
            same_prediction = changed_prediction == original_prediction

            jaccard = top_k_jaccard(
                original_explanation["top_features"],
                changed_explanation["top_features"],
            )
            overlap = top_k_overlap(
                original_explanation["top_features"],
                changed_explanation["top_features"],
            )
            rank_similarity = rank_biased_similarity(
                original_explanation["top_features"],
                changed_explanation["top_features"],
            )

            record = {
                **base_record,
                "condition": variant["condition"],
                "transformation": variant["transformation"],
                "candidate_paraphrase": variant["candidate_paraphrase"],
                "variant_text": changed_text,
                "variant_prediction": changed_prediction,
                "prediction_consistent": same_prediction,
                "variant_unchanged": unchanged,
                "original_top_features": ", ".join(
                    original_explanation["top_features"]
                ),
                "variant_top_features": ", ".join(
                    changed_explanation["top_features"]
                ),
                "top_feature_jaccard": jaccard,
                "top_feature_overlap": overlap,
                "rank_weighted_similarity": rank_similarity,
                "original_score": original_explanation["score"],
                "variant_score": changed_explanation["score"],
                "absolute_score_change": abs(
                    changed_explanation["score"]
                    - original_explanation["score"]
                ),
                "actual_label": actual,
                "classification_correct_after_change": (
                    changed_prediction == actual
                ),
            }
            records.append(record)

            explanation_records.append({
                "condition": variant["condition"],
                "actual_label": actual,
                "text": changed_text,
                "attributions": changed_explanation["attributions"],
            })

            variant_performance.setdefault(
                variant["condition"], {"y": [], "pred": []}
            )
            variant_performance[variant["condition"]]["y"].append(actual)
            variant_performance[variant["condition"]]["pred"].append(
                changed_prediction
            )

    detail_df = pd.DataFrame(records)
    detail_df.to_csv(
        OUTPUT / "experiment7_message_level_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 9. SUMMARY BY TRANSFORMATION
    # -----------------------------------------------------

    summaries = []

    for condition, group in detail_df.groupby("condition"):
        summaries.append({
            "condition": condition,
            "messages_evaluated": len(group),
            "prediction_consistency": group[
                "prediction_consistent"
            ].mean(),
            "prediction_flip_rate": 1.0 - group[
                "prediction_consistent"
            ].mean(),
            "mean_top_feature_jaccard": group[
                "top_feature_jaccard"
            ].mean(),
            "mean_top_feature_overlap": group[
                "top_feature_overlap"
            ].mean(),
            "mean_rank_weighted_similarity": group[
                "rank_weighted_similarity"
            ].mean(),
            "mean_absolute_score_change": group[
                "absolute_score_change"
            ].mean(),
            "classification_accuracy_after_change": group[
                "classification_correct_after_change"
            ].mean(),
            "unchanged_variant_fraction": group[
                "variant_unchanged"
            ].mean(),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        OUTPUT / "experiment7_stability_summary.csv",
        index=False,
    )

    # Classification performance on the selected correctly classified
    # subset and its transformed variants. These are conditional metrics,
    # not unbiased estimates of full-test-set performance.
    performance_rows = []

    performance_rows.append({
        "condition": "original_selected_subset",
        **classification_metrics(
            np.asarray(all_original_y),
            np.asarray(all_original_pred),
        ),
    })

    for condition, values in variant_performance.items():
        performance_rows.append({
            "condition": condition,
            **classification_metrics(
                np.asarray(values["y"]),
                np.asarray(values["pred"]),
            ),
        })

    pd.DataFrame(performance_rows).to_csv(
        OUTPUT / "experiment7_classification_performance.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 10. ERROR ANALYSIS
    # -----------------------------------------------------

    # Prefer changed predictions, then the lowest feature overlap.
    errors = detail_df[
        (~detail_df["prediction_consistent"])
        | (~detail_df["classification_correct_after_change"])
        | (detail_df["top_feature_jaccard"] < 0.25)
    ].copy()

    errors["error_priority"] = (
        (~errors["prediction_consistent"]).astype(int) * 3
        + (~errors["classification_correct_after_change"]).astype(int) * 2
        + (errors["top_feature_jaccard"] < 0.25).astype(int)
    )

    errors = errors.sort_values(
        ["error_priority", "top_feature_jaccard"],
        ascending=[False, True],
    ).head(30)

    errors.to_csv(
        OUTPUT / "experiment7_error_analysis.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 11. FEATURE-IMPORTANCE PLOTS
    # -----------------------------------------------------

    for label, name in [(0, "legitimate"), (1, "scam")]:
        subset = [
            record for record in explanation_records
            if record["condition"] == "original"
            and record["actual_label"] == label
        ]
        save_feature_plot(
            subset,
            f"experiment7_{name}_feature_importance.png",
            f"Top Word Attributions: {name.title()} Messages",
        )

    # Plot explanation stability by condition.
    if not summary_df.empty:
        plot_df = summary_df.set_index("condition")

        plt.figure(figsize=(9, 5))
        plot_df["mean_top_feature_jaccard"].plot(
            kind="bar",
        )
        plt.ylabel("Mean top-feature Jaccard similarity")
        plt.ylim(0, 1)
        plt.title("Explanation Stability by Transformation")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            OUTPUT / "experiment7_explanation_stability.png",
            dpi=300,
        )
        plt.close()

        plt.figure(figsize=(9, 5))
        plot_df["prediction_consistency"].plot(kind="bar")
        plt.ylabel("Prediction consistency")
        plt.ylim(0, 1)
        plt.title("Prediction Consistency by Transformation")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            OUTPUT / "experiment7_prediction_consistency.png",
            dpi=300,
        )
        plt.close()

    protocol = {
        "experiment": "Experiment 7: Explanation Stability",
        "model_file": args.model,
        "predictions_file": args.predictions,
        "random_seed": SEED,
        "maximum_messages_per_class": MAX_MESSAGES_PER_CLASS,
        "selected_correctly_classified_messages": len(selected),
        "top_k_features": TOP_K_FEATURES,
        "maximum_unique_words_occluded_per_message": MAX_WORDS_TO_EXPLAIN,
        "explanation_method": (
            "Single-word occlusion; signed change in the original "
            "predicted-class decision margin"
        ),
        "stability_metrics": [
            "prediction consistency",
            "top-feature Jaccard similarity",
            "top-feature overlap",
            "rank-weighted similarity",
            "absolute decision-score change",
        ],
        "important_limitations": [
            "Candidate paraphrases are not guaranteed to preserve meaning.",
            "Manually review transformations before treating them as "
            "meaning-preserving.",
            "Only correctly classified original messages were selected.",
            "Word occlusion is a perturbation-based attribution method, "
            "not proof of causal feature importance.",
            "Classification metrics on the selected subset are conditional "
            "and must not replace full-test-set results.",
            "Attributions are restricted to the first configured number "
            "of unique words in each message.",
        ],
    }

    with open(
        OUTPUT / "experiment7_protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2)

    print("\n" + "=" * 65)
    print("EXPERIMENT 7 COMPLETED")
    print("=" * 65)
    print("\nSTABILITY SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nCLASSIFICATION PERFORMANCE")
    print(pd.DataFrame(performance_rows).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT.resolve()}")
    for path in sorted(OUTPUT.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
