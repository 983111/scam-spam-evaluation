
import io
import zipfile
import urllib.request
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

print("Downloading official UCI dataset...")

request = urllib.request.Request(
    URL,
    headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(request, timeout=60) as response:
    archive_bytes = response.read()

with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    filename = next(
        name for name in archive.namelist()
        if name.split("/")[-1] == "SMSSpamCollection"
    )
    with archive.open(filename) as file:
        df = pd.read_csv(
            file,
            sep="\t",
            header=None,
            names=["original_label", "text"],
            encoding="utf-8"
        )

df["label"] = df["original_label"].map({
    "ham": "legitimate",
    "spam": "scam"
})

df = df.dropna(subset=["text", "label"])
df = df[["text", "label"]]
df.to_csv("sms_spam_dataset.csv", index=False, encoding="utf-8")

print("\nDataset downloaded successfully.")
print("Total messages:", len(df))
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nSaved to: sms_spam_dataset.csv")
