import csv
import json
import glob
import os


CONTEXT_FILE = "../Golden_Dataset_Preparation/assignment_contexts_by_title.json"
INPUT_DIR = "../golden_dataset"
OUTPUT_FILE = "organic_dataset.csv"


def load_contexts():
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        item["filename_slug"]: item["assignment_details"]
        for item in data.values()
    }


def get_context(filename, contexts):
    filename = os.path.basename(filename).lower()

    matches = [
        context
        for slug, context in contexts.items()
        if slug.lower() in filename
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely match assignment context for: {filename}"
        )

    return matches[0]


def parse_conversation(conversation):
    if isinstance(conversation, str):
        conversation = json.loads(conversation)

    previous_conversation = conversation[:-2]
    user_query = conversation[-2]["content"]
    assistant_response = conversation[-1]["content"]

    return previous_conversation, user_query, assistant_response


def convert():
    contexts = load_contexts()

    records = {}

    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    for file in files:
        assignment_context = get_context(file, contexts)

        with open(file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            required = {
                "id",
                "criterion",
                "conversation",
                "score",
                "trace"
            }

            if not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{file} is missing required columns. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                old_id = row["id"]
                criterion = row["criterion"].strip().lower()
                score = row["score"].strip().lower()

                # Drift is not part of the current output format
                if criterion == "drift":
                    continue

                if criterion not in {
                    "irrelevancy",
                    "solution_proximity"
                }:
                    raise ValueError(
                        f"Unknown criterion '{row['criterion']}' "
                        f"in {file}, id={old_id}"
                    )

                if old_id not in records:
                    conversation = json.loads(row["conversation"])

                    previous_conversation, user_query, assistant_response = (
                        parse_conversation(conversation)
                    )

                    records[old_id] = {
                        "assignment_context": assignment_context,
                        "previous_conversation": json.dumps(
                            previous_conversation,
                            ensure_ascii=False
                        ),
                        "user_query": user_query,
                        "assistant_response": assistant_response,
                        "irrelevancy": "",
                        "solution_proximality": ""
                    }

                if criterion == "irrelevancy":
                    records[old_id]["irrelevancy"] = score

                elif criterion == "solution_proximity":
                    records[old_id]["solution_proximality"] = score

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "assignment_context",
            "previous_conversation",
            "user_query",
            "assistant_response",
            "irrelevancy",
            "solution_proximality"
        ])

        for new_id, record in enumerate(records.values(), start=1):
            writer.writerow([
                new_id,
                record["assignment_context"],
                record["previous_conversation"],
                record["user_query"],
                record["assistant_response"],
                record["irrelevancy"],
                record["solution_proximality"]
            ])

    print(f"Processed {len(files)} CSV files.")
    print(f"Created {len(records)} combined records.")
    print(f"Saved to: {OUTPUT_FILE}")


convert()