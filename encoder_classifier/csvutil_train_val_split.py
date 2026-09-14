import csv
import random

INPUT_FILE = "synthetic_dataset.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
SPLIT_RATIO = 0.8
SEED = 42

def split_csv(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    data = rows[1:]

    random.Random(seed).shuffle(data)

    split = int(len(data) * train_ratio)

    train_data = data[:split]
    val_data = data[split:]

    with open(train_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_data)

    with open(val_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_data)

    print(f"Total: {len(data)}")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")


split_csv(
    INPUT_FILE,
    TRAIN_FILE,
    VAL_FILE,
    train_ratio=SPLIT_RATIO,
    seed=SEED
)