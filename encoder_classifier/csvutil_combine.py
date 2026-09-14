import csv
from csvutil_fix_ids import fixIDs

FILE1 = "synthetic_dataset_raahul.csv"
FILE2 = "synthetic_dataset_ram.csv"
OUTPUT_FILE = "synthetic_dataset.csv"

with open(FILE1, "r", newline="", encoding="utf-8") as f:
    rows1 = list(csv.reader(f))

with open(FILE2, "r", newline="", encoding="utf-8") as f:
    rows2 = list(csv.reader(f))

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows1)
    writer.writerows(rows2[1:])

print(f"Appended {len(rows2) - 1} rows from {FILE2} to {FILE1} and wrote into {OUTPUT_FILE}")
fixIDs(OUTPUT_FILE, OUTPUT_FILE, 1)