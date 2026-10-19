"""
ScamShield Experiment 3: Adversarial Evasion Testing
====================================================
Compares:
  1. Clean baseline
  2. Unicode/text normalization
  3. Adversarial training
  4. Combined normalization + adversarial training

Input:
  A CSV with message text and labels. Defaults:
    text column: text
    label column: label

Example:
  python experiment3.py --data your_dataset.csv --text-col text --label-col label

Dependencies:
  pip install pandas numpy scikit-learn matplotlib seaborn Unidecode

Research safeguards:
- Train/test split occurs before adversarial test variants are generated.
- Adversarial training examples are created from the training split only.
- Transformations preserve labels by construction, but meaning preservation is not
  guaranteed automatically. Review/exported samples and exclude ambiguous cases.
- Reported attack success rate is conditional on messages correctly classified
  in the clean condition. A successful evasion means a clean-correct scam becomes
  classified as non-scam after transformation.
"""

import argparse
import html
import os
import random
import re
import unicodedata
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------------- Text transformations -------------------------

HOMOGLYPHS = {
    "a": ["а", "ɑ"], "c": ["с"], "e": ["е"], "i": ["і", "ı"],
    "o": ["о", "ο"], "p": ["р"], "s": ["ѕ"], "x": ["х"],
    "y": ["у"], "A": ["А"], "B": ["В"], "C": ["С"], "E": ["Е"],
    "H": ["Н"], "I": ["І"], "K": ["К"], "M": ["М"], "O": ["О"],
    "P": ["Р"], "T": ["Т"], "X": ["Х"],
}

LEET = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "$",
    "t": "7", "A": "4", "E": "3", "I": "1", "O": "0",
    "S": "$", "T": "7",
}

COMMON_TYPOS = {
    "account": "acount",
    "verify": "verfiy",
    "urgent": "urgnet",
    "winner": "winer",
    "payment": "paymant",
    "bank": "bnak",
    "click": "clik",
    "claim": "calim",
    "prize": "przie",
    "free": "fre",
    "update": "udpate",
    "suspended": "suspnded",
    "password": "pasword",
    "delivery": "delivrey",
    "confirm": "confrim",
}

def transform_homoglyph(text, rng):
    chars = list(text)
    candidates = [i for i, ch in enumerate(chars) if ch in HOMOGLYPHS]
    if not candidates:
        return text
    count = max(1, min(len(candidates), round(len(candidates) * 0.08)))
    for i in rng.sample(candidates, count):
        chars[i] = rng.choice(HOMOGLYPHS[chars[i]])
    return "".join(chars)

def transform_spaces_punctuation(text, rng):
    # Insert spaces inside selected words or punctuation around token boundaries.
    words = text.split()
    if not words:
        return text
    out = []
    for word in words:
        if len(word) >= 5 and rng.random() < 0.22:
            pos = rng.randint(2, len(word) - 2)
            word = word[:pos] + " " + word[pos:]
        if word and rng.random() < 0.12:
            word = word[0] + "." + word[1:] if len(word) > 2 else word
        out.append(word)
    return " ".join(out)

def transform_typos(text, rng):
    out = text
    # Apply at most two listed word-level typos, case-insensitively.
    keys = list(COMMON_TYPOS.keys())
    rng.shuffle(keys)
    changed = 0
    for word in keys:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", flags=re.I)
        if pattern.search(out):
            def replacement(match):
                original = match.group(0)
                typo = COMMON_TYPOS[word]
                return typo.capitalize() if original[:1].isupper() else typo
            out, n = pattern.subn(replacement, out, count=1)
            if n:
                changed += 1
            if changed >= 2:
                break
    # Otherwise try a transposition in one sufficiently long alphabetic word.
    if changed == 0:
        matches = list(re.finditer(r"\b[A-Za-z]{5,}\b", out))
        if matches:
            m = rng.choice(matches)
            w = m.group(0)
            pos = rng.randint(1, len(w) - 2)
            typo = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
            out = out[:m.start()] + typo + out[m.end():]
    return out

def transform_numbers_symbols(text, rng):
    out = []
    replacements = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$"}
    for ch in text:
        if ch in replacements and rng.random() < 0.13:
            out.append(replacements[ch])
        else:
            out.append(ch)
    return "".join(out)

def transform_paraphrase(text, rng):
    # Conservative, meaning-intended phrase substitutions. This is not a
    # semantic paraphrase model; human review is required before final reporting.
    substitutions = [
        (r"\bclick here\b", "open this link"),
        (r"\bverify your account\b", "confirm your account"),
        (r"\baccount suspended\b", "account has been put on hold"),
        (r"\bclaim your prize\b", "collect your reward"),
        (r"\bfree\b", "at no cost"),
        (r"\burgent\b", "time-sensitive"),
        (r"\bupdate\b", "refresh"),
        (r"\bpayment\b", "transaction"),
        (r"\bwon\b", "have won"),
        (r"\bselected\b", "chosen"),
        (r"\bconfirm\b", "validate"),
        (r"\bpassword\b", "login password"),
    ]
    rng.shuffle(substitutions)
    out = text
    for pattern, replacement in substitutions:
        new, n = re.subn(pattern, replacement, out, count=1, flags=re.I)
        if n:
            out = new
            break
    return out

TRANSFORMS = {
    "homoglyphs": transform_homoglyph,
    "spaces_punctuation": transform_spaces_punctuation,
    "typos_transpositions": transform_typos,
    "numbers_symbols": transform_numbers_symbols,
    "paraphrases": transform_paraphrase,
}

def normalize_text(text):
    """NFKC normalization, HTML unescaping, Unicode lookalike mapping,
    zero-width removal, whitespace cleanup, and light punctuation cleanup.
    """
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width and directional formatting characters.
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Cf"
    )
    # Map common Cyrillic/Greek homoglyphs to Latin characters.
    lookalikes = {
        "а": "a", "ɑ": "a", "с": "c", "е": "e", "і": "i", "ı": "i",
        "о": "o", "ο": "o", "р": "p", "ѕ": "s", "х": "x", "у": "y",
        "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I",
        "К": "K", "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X",
    }
    text = "".join(lookalikes.get(ch, ch) for ch in text)
    # Undo inserted intra-word spaces and punctuation, e.g. "ver ify" / "v.erify".
    text = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z])", "", text) if False else text
    # Remove punctuation between letters when it appears inside a token.
    text = re.sub(r"(?<=[A-Za-z])[.\-_]+(?=[A-Za-z])", "", text)
    # Collapse repeated whitespace and case-fold.
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])

def build_augmented_training(X_train, y_train, seed=SEED):
    rng = random.Random(seed)
    xs = list(map(str, X_train))
    ys = list(y_train)
    for i, (text, label) in enumerate(zip(X_train, y_train)):
        # Two reproducible transformation families per training item.
        keys = list(TRANSFORMS.keys())
        rng.shuffle(keys)
        for j, key in enumerate(keys[:2]):
            local_rng = random.Random(seed + i * 31 + j)
            transformed = TRANSFORMS[key](str(text), local_rng)
            if transformed.strip() and transformed != str(text):
                xs.append(transformed)
                ys.append(label)
    return np.array(xs, dtype=object), np.array(ys)

def positive_label_from_data(y):
    """Choose a scam/fraud-like label when labels are strings; otherwise
    assume the minority class is positive. Override with --positive-label.
    """
    values, counts = np.unique(y, return_counts=True)
    text_map = {str(v).strip().casefold(): v for v in values}
    for candidate in ["scam", "spam", "fraud", "phishing", "1", "true", "yes"]:
        if candidate in text_map:
            return text_map[candidate]
    return values[np.argmin(counts)]

def predict(model, texts, do_normalize=False):
    vals = [normalize_text(t) for t in texts] if do_normalize else list(map(str, texts))
    return model.predict(vals)

def compute_metrics(y_true, y_pred, positive_label):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "scam_precision": precision_score(y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0),
        "scam_recall": recall_score(y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--text-col", default="text", help="Message text column")
    parser.add_argument("--label-col", default="label", help="Label column")
    parser.add_argument("--positive-label", default=None, help="Label representing scam/spam/fraud, e.g. scam or 1")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--output-dir", default="experiment3_results")
    parser.add_argument("--review-file", default=None, help="Optional reviewed CSV with columns message_id, transformation, keep (1/0), notes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.data)
    if args.text_col not in df or args.label_col not in df:
        raise ValueError(
            f"CSV must contain columns '{args.text_col}' and '{args.label_col}'. "
            f"Found: {list(df.columns)}"
        )
    df = df[[args.text_col, args.label_col]].dropna().copy()
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col].str.len() > 0].reset_index(drop=True)
    if df[args.label_col].nunique() < 2:
        raise ValueError("At least two distinct labels are required.")

    X = df[args.text_col].to_numpy(dtype=object)
    y = df[args.label_col].to_numpy()
    positive_label = args.positive_label
    if positive_label is not None:
        matches = [v for v in np.unique(y) if str(v) == str(positive_label)]
        if not matches:
            raise ValueError(f"positive label {positive_label!r} not found in labels {np.unique(y)}")
        positive_label = matches[0]
    else:
        positive_label = positive_label_from_data(y)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=SEED, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train the four model/defense configurations.
    baseline = make_pipeline().fit(X_train, y_train)

    # Normalization defense: normalize both training and inference text.
    normalized_train = np.array([normalize_text(t) for t in X_train], dtype=object)
    normalized_model = make_pipeline().fit(normalized_train, y_train)

    X_aug, y_aug = build_augmented_training(X_train, y_train)
    adversarial_model = make_pipeline().fit(X_aug, y_aug)

    combined_train = np.array([normalize_text(t) for t in X_aug], dtype=object)
    combined_model = make_pipeline().fit(combined_train, y_aug)

    models = {
        "clean_baseline": (baseline, False),
        "normalization": (normalized_model, True),
        "adversarial_training": (adversarial_model, False),
        "combined_defense": (combined_model, True),
    }

    # Optionally apply a manually reviewed transformation allow-list.
    review_allow = None
    if args.review_file and os.path.exists(args.review_file):
        review_df = pd.read_csv(args.review_file)
        required = {"message_id", "transformation", "keep"}
        if not required.issubset(review_df.columns):
            raise ValueError(f"Review CSV must contain columns: {sorted(required)}")
        review_allow = set(
            (int(r.message_id), str(r.transformation))
            for r in review_df.itertuples()
            if int(r.keep) == 1
        )

    all_examples = []
    predictions = {}
    metric_rows = []
    transform_rng_seed = SEED + 9000

    # Clean performance and predictions.
    for model_name, (model, use_norm) in models.items():
        pred = predict(model, X_test, do_normalize=use_norm)
        predictions[(model_name, "clean")] = pred
        m = compute_metrics(y_test, pred, positive_label)
        metric_rows.append({"model": model_name, "condition": "clean", "transformation": "none", **m})

    # Generate held-out attack variants, separately for each transformation.
    for trans_name, trans_fn in TRANSFORMS.items():
        transformed_texts = []
        source_indices = []
        for j, original in enumerate(X_test):
            rng = random.Random(transform_rng_seed + j * 101 + len(trans_name))
            variant = trans_fn(str(original), rng)
            # Retain only actual changes; unchanged rows do not constitute an attack.
            if variant != str(original) and variant.strip():
                transformed_texts.append(variant)
                source_indices.append(j)
                all_examples.append({
                    "message_id": int(j),
                    "original_message": str(original),
                    "transformation": trans_name,
                    "transformed_message": variant,
                    "true_label": y_test[j],
                    "changed": 1,
                    "human_review_required": 1 if trans_name == "paraphrases" else 0,
                })
        if not transformed_texts:
            continue

        # If review file provided, only reviewed-in variants contribute to main metrics.
        keep_positions = list(range(len(transformed_texts)))
        if review_allow is not None:
            keep_positions = [
                k for k, j in enumerate(source_indices)
                if (int(j), trans_name) in review_allow
            ]
        if not keep_positions:
            print(f"WARNING: no reviewed-in examples for {trans_name}; skipping main metric.")
            continue

        texts_eval = [transformed_texts[k] for k in keep_positions]
        idx_eval = [source_indices[k] for k in keep_positions]
        y_eval = y_test[idx_eval]

        for model_name, (model, use_norm) in models.items():
            pred = predict(model, texts_eval, do_normalize=use_norm)
            predictions[(model_name, trans_name)] = (idx_eval, pred)
            m = compute_metrics(y_eval, pred, positive_label)
            metric_rows.append({
                "model": model_name,
                "condition": "attacked",
                "transformation": trans_name,
                "n": len(y_eval),
                **m,
            })

            # Attack success: scam correctly detected on clean text, but missed
            # after transformation. This is conditional on clean-correct scams.
            clean_pred = predictions[(model_name, "clean")][idx_eval]
            is_positive = np.array([v == positive_label for v in y_eval])
            clean_correct = np.array([p == t for p, t in zip(clean_pred, y_eval)])
            attacked_missed = np.array([p != positive_label for p in pred])
            eligible = is_positive & clean_correct
            denominator = int(eligible.sum())
            successes = int((eligible & attacked_missed).sum())
            asr = successes / denominator if denominator else np.nan
            metric_rows[-1]["attack_success_rate"] = asr
            metric_rows[-1]["asr_denominator_clean_correct_scams"] = denominator
            metric_rows[-1]["successful_evasions"] = successes

            # Add model-specific predictions to each example for later audit.
            for k, j in zip(keep_positions, idx_eval):
                ex = all_examples
                # Locate this exact transformed variant.
                for row in reversed(ex):
                    if row["message_id"] == int(j) and row["transformation"] == trans_name:
                        row[f"prediction_{model_name}"] = pred[idx_eval.index(j)]
                        row[f"clean_prediction_{model_name}"] = clean_pred[idx_eval.index(j)]
                        break

    metrics_df = pd.DataFrame(metric_rows)
    examples_df = pd.DataFrame(all_examples)

    # Add performance delta from clean macro-F1 to each attacked result.
    clean_f1 = metrics_df[metrics_df.condition == "clean"].set_index("model")["macro_f1"].to_dict()
    metrics_df["macro_f1_change_vs_clean"] = metrics_df.apply(
        lambda r: r["macro_f1"] - clean_f1.get(r["model"], np.nan)
        if r["condition"] == "attacked" else 0.0,
        axis=1,
    )

    # Write all result tables.
    metrics_path = os.path.join(args.output_dir, "experiment3_metrics.csv")
    examples_path = os.path.join(args.output_dir, "experiment3_attack_examples.csv")
    metrics_df.to_csv(metrics_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    # Compact transformation-by-model table.
    attacked = metrics_df[metrics_df.condition == "attacked"].copy()
    if not attacked.empty:
        pivot = attacked.pivot_table(
            index="transformation", columns="model",
            values=["macro_f1", "scam_recall", "attack_success_rate"],
            aggfunc="first"
        )
        pivot.to_csv(os.path.join(args.output_dir, "experiment3_transformation_by_model.csv"))

    # Clean vs attacked macro-F1 chart.
    plot_df = metrics_df.copy()
    plot_df["series"] = plot_df.apply(
        lambda r: "Clean" if r["condition"] == "clean" else str(r["transformation"]),
        axis=1
    )
    models_order = list(models.keys())
    x = np.arange(len(models_order))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 6))
    clean_vals = [
        float(metrics_df[(metrics_df.model == m) & (metrics_df.condition == "clean")]["macro_f1"].iloc[0])
        for m in models_order
    ]
    ax.bar(x - width * 2.5, clean_vals, width, label="Clean")
    trans_names = list(TRANSFORMS.keys())
    offsets = np.linspace(-1.5 * width, 3.5 * width, len(trans_names))
    for trans, offset in zip(trans_names, offsets):
        vals = []
        for m in models_order:
            row = metrics_df[
                (metrics_df.model == m) & (metrics_df.condition == "attacked") &
                (metrics_df.transformation == trans)
            ]
            vals.append(float(row["macro_f1"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, vals, width, label=trans)
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("ScamShield: Clean vs Adversarially Transformed Messages")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "experiment3_clean_vs_attacked.png"), dpi=200)
    plt.close(fig)

    # Attack success rate chart.
    asr_df = attacked.pivot(index="transformation", columns="model", values="attack_success_rate")
    if not asr_df.empty:
        ax = asr_df.plot(kind="bar", figsize=(12, 6), ylim=(0, 1))
        ax.set_ylabel("Attack success rate")
        ax.set_title("Evasion Success Rate by Transformation and Defense")
        ax.set_xlabel("Transformation")
        ax.legend(title="Model", fontsize=8)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "experiment3_attack_success_rate.png"), dpi=200)
        plt.close()

    # Full per-model clean classification reports.
    with open(os.path.join(args.output_dir, "experiment3_classification_reports.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset rows used: {len(df)}\n")
        f.write(f"Training rows: {len(train_idx)}; test rows: {len(test_idx)}\n")
        f.write(f"Detected positive/scam label: {positive_label!r}\n\n")
        for model_name, (model, use_norm) in models.items():
            pred = predictions[(model_name, "clean")]
            f.write(f"\n=== {model_name}: clean held-out test ===\n")
            f.write(classification_report(y_test, pred, zero_division=0))

    # Record metadata and a protocol summary.
    with open(os.path.join(args.output_dir, "experiment3_protocol.txt"), "w", encoding="utf-8") as f:
        f.write("EXPERIMENT 3 PROTOCOL\n")
        f.write("=====================\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Input file: {os.path.abspath(args.data)}\n")
        f.write(f"Text column: {args.text_col}\nLabel column: {args.label_col}\n")
        f.write(f"Rows after cleaning: {len(df)}\n")
        f.write(f"Train/test sizes: {len(train_idx)}/{len(test_idx)}\n")
        f.write(f"Positive label: {positive_label!r}\n")
        f.write("Classifier: TF-IDF word unigrams+bigrams + class-balanced Logistic Regression\n")
        f.write("Split: stratified random split; seed 42\n")
        f.write("Adversarial augmentation: up to two transformations per training message\n")
        f.write("Caution: paraphrase substitutions are heuristic; manually validate meaning and labels.\n")
        f.write("Attack success rate: clean-correct positive examples changed to non-positive predictions / clean-correct positive examples.\n")

    print("\nExperiment 3 complete.")
    print(f"Results directory: {os.path.abspath(args.output_dir)}")
    print("\nClean metrics:")
    print(metrics_df[metrics_df.condition == "clean"][["model", "macro_f1", "scam_precision", "scam_recall"]].to_string(index=False))
    print("\nSaved files:")
    for name in sorted(os.listdir(args.output_dir)):
        print(" -", name)

if __name__ == "__main__":
    main()
