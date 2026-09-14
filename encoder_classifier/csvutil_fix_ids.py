import csv

def fixIDs(input_file, output_file, start_id):
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    data = rows[1:]

    for i, row in enumerate(data, start=start_id):
        row[0] = str(i)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Renumbered {len(data)} rows.")
    print(f"IDs: {start_id} to {start_id + len(data) - 1}")
    print(f"Saved as {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "synthetic_dataset_raahul.csv"
    OUTPUT_FILE = "synthetic_dataset_raahul.csv"
    start_id = 1

    fixIDs(INPUT_FILE, OUTPUT_FILE, start_id)