
from pathlib import Path
from datasets import load_dataset

# 1. Define dataset information
DATASET_NAME = "fancyzhx/ag_news"
OUTPUT_DIR = Path("data/bronze")

# 2. Create the output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Download the dataset
print("Downloading AG News dataset...")

dataset = load_dataset(DATASET_NAME)

# 4. Display dataset information
print("\nDataset loaded successfully!")
print(dataset)

# 5. Display an example
print("\nFirst article:")
print(dataset["train"][0]["text"])

print("\nCategory:")
print(dataset["train"][0]["label"])

# 6. Save the original train and test splits
dataset["train"].to_parquet(OUTPUT_DIR / "train.parquet")
dataset["test"].to_parquet(OUTPUT_DIR / "test.parquet")

print("\nData saved successfully!")
print(f"Location: {OUTPUT_DIR.resolve()}")
