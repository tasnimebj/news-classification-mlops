
import pandas as pd

train = pd.read_parquet("data/bronze/train.parquet")
test = pd.read_parquet("data/bronze/test.parquet")

EXPECTED_LABELS = {0, 1, 2, 3}

def validate(df, name):
    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Nulls:", df.isnull().sum().to_dict())
    print("Duplicates:", df.duplicated().sum())

    # Check valid labels
    invalid_labels = ~df["label"].isin(EXPECTED_LABELS)
    print("Invalid labels:", invalid_labels.sum())

    # Check empty articles
    empty_text = df["text"].str.strip().eq("")
    print("Empty articles:", empty_text.sum())

    # Class distribution
    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    assert not df.isnull().any().any()
    assert df.duplicated().sum() == 0
    assert invalid_labels.sum() == 0
    assert empty_text.sum() == 0

validate(train, "TRAIN")
validate(test, "TEST")

# Check identical text across train and test
overlap = set(train["text"]) & set(test["text"])

print("\nTrain/Test text overlap:", len(overlap))

if overlap:
    print("WARNING: Review overlapping articles before training.")
else:
    print("No exact train/test text overlap detected.")
