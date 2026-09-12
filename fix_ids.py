import csv

input_file = "synthetic_dataset.csv"
output_file = "synthetic_dataset.csv"

start_id = 1001

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