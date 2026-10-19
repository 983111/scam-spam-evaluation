
import io
import json
import re
import hashlib
import zipfile
import urllib.request
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

BASE = Path(__file__).resolve().parent
RAW = BASE / "data_raw"
PROCESSED = BASE / "data_processed"
REPORTS = BASE / "reports"

for folder in [RAW, PROCESSED, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)


def download_dataset():
    dataset_path = RAW / "SMSSpamCollection"

    if dataset_path.exists():
        print("Using existing downloaded dataset.")
        return dataset_path

    print("Downloading UCI SMS Spam Collection...")

    request = urllib.request.Request(
        URL, headers={"User-Agent": "ScamShieldResearch/1.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.split("/")[-1] == "SMSSpamCollection"
        ]

        if not candidates:
            raise FileNotFoundError(
                "Dataset file was not found inside the downloaded ZIP."
            )

        dataset_path.write_bytes(archive.read(candidates[0]))

    print("Download complete.")
    return dataset_path


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def make_id(text):
    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()[:16]


def find_near_duplicates(df, threshold=0.92):
    """
    Finds candidate pairs for manual review.
    Not a semantic duplicate detector.
    """
    records = []
    texts = df["normalized_text"].tolist()

    # Limit comparison to messages of similar length and first character.
    # This keeps the audit practical for this dataset.
    buckets = {}

    for i, text in enumerate(texts):
        if not text or len(text) > 500:
            continue

        key = (len(text) // 20, text[:1])
        buckets.setdefault(key, []).append(i)

    seen = set()

    for indices in buckets.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)

                a, b = texts[i], texts[j]
                similarity = SequenceMatcher(None, a, b).ratio()

                if a != b and similarity >= threshold:
                    records.append({
                        "row_id_1": df.iloc[i]["row_id"],
                        "row_id_2": df.iloc[j]["row_id"],
                        "similarity": round(similarity, 4),
                        "label_1": int(df.iloc[i]["label"]),
                        "label_2": int(df.iloc[j]["label"]),
                        "text_1": a,
                        "text_2": b
                    })

    return pd.DataFrame(records)


def main():
    path = download_dataset()

    # UCI file format: label<TAB>message
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["original_label", "text"],
        encoding="utf-8",
        quoting=3,
        on_bad_lines="error"
    )

    initial_rows = len(df)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = (
        df["original_label"].map({"ham": 0, "spam": 1})
    )

    empty_text_count = int(df["text"].eq("").sum())
    invalid_label_count = int(df["label"].isna().sum())

    df = df[
        df["text"].ne("") & df["label"].notna()
    ].copy()

    df["label"] = df["label"].astype(int)
    df["source_dataset"] = "UCI SMS Spam Collection"
    df["source_file"] = "SMSSpamCollection"
    df["language"] = "English (dataset description)"
    df["normalized_text"] = df["text"].map(normalize_text)
    df["row_id"] = df["text"].map(make_id)

    # Exact duplicate audit.
    duplicate_mask = df.duplicated(
        subset=["normalized_text", "label"], keep="first"
    )

    exact_duplicates = df[duplicate_mask].copy()
    exact_duplicates.to_csv(
        REPORTS / "exact_duplicates.csv", index=False
    )

    clean = df[~duplicate_mask].copy()

    # Also identify identical text carrying conflicting labels.
    label_conflicts = (
        clean.groupby("normalized_text")["label"].nunique()
    )
    conflict_texts = label_conflicts[label_conflicts > 1].index
    conflicts = clean[
        clean["normalized_text"].isin(conflict_texts)
    ].copy()
    conflicts.to_csv(
        REPORTS / "conflicting_labels.csv", index=False
    )

    # Near-duplicate candidates for manual inspection.
    near = find_near_duplicates(clean)
    near.to_csv(
        REPORTS / "near_duplicate_candidates.csv", index=False
    )

    # Dataset statistics.
    class_stats = (
        clean["label"].value_counts()
        .rename(index={0: "legitimate_ham", 1: "spam"})
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_stats.to_csv(
        REPORTS / "class_distribution.csv", index=False
    )

    source_stats = (
        clean.groupby(["source_dataset", "label"])
        .size().reset_index(name="count")
    )
    source_stats.to_csv(
        REPORTS / "source_breakdown.csv", index=False
    )

    # Stratified 70/15/15 split.
    train, temp = train_test_split(
        clean,
        test_size=0.30,
        random_state=SEED,
        stratify=clean["label"]
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        stratify=temp["label"]
    )

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    result = pd.concat(
        [train, validation, test], ignore_index=True
    )

    # Dataset contains no campaign IDs, so this is a stratified
    # random split, not a campaign-grouped split.
    output_columns = [
        "row_id", "text", "label", "original_label",
        "source_dataset", "language", "source_file", "split"
    ]

    result[output_columns].to_csv(
        PROCESSED / "benchmark.csv", index=False
    )
    result[["row_id", "split"]].to_csv(
        PROCESSED / "split_manifest.csv", index=False
    )

    split_stats = (
        result.groupby(["split", "label"])
        .size().reset_index(name="count")
    )
    split_stats.to_csv(
        REPORTS / "split_statistics.csv", index=False
    )

    # Check for exact normalized texts across splits.
    split_leakage = (
        result.groupby("normalized_text")["split"].nunique()
    )
    leakage_count = int((split_leakage > 1).sum())

    summary = {
        "dataset": "UCI SMS Spam Collection",
        "dataset_url": URL,
        "license": "CC BY 4.0",
        "initial_rows": int(initial_rows),
        "empty_text_rows": empty_text_count,
        "invalid_label_rows": invalid_label_count,
        "exact_duplicate_rows_removed": int(len(exact_duplicates)),
        "final_rows": int(len(clean)),
        "near_duplicate_candidate_pairs": int(len(near)),
        "conflicting_label_text_groups": int(len(conflict_texts)),
        "split_sizes": {
            str(k): int(v)
            for k, v in result["split"].value_counts().items()
        },
        "exact_texts_crossing_splits": leakage_count,
        "split_method": "stratified random, seed 42",
        "campaign_metadata": "not provided in this dataset",
        "label_definition": "0=ham (legitimate), 1=spam",
        "warning": (
            "Spam is not synonymous with confirmed fraud or phishing. "
            "Near-duplicate candidates need manual review. "
            "Random splitting may still allow related templates across splits."
        )
    }

    with open(
        REPORTS / "dataset_audit_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nEXPERIMENT 1 FINISHED")
    print(json.dumps(summary, indent=2))
    print("\nFiles saved in:")
    print(PROCESSED)
    print(REPORTS)


if __name__ == "__main__":
    main()
