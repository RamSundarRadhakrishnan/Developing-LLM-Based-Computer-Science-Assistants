from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

#Configuration

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SCORED_DATASET_DIR = PROJECT_ROOT / "lab_exercise_dataset"
ASSIGNMENT_CONTEXTS_PATH = PROJECT_ROOT / "lab_exercise_raw" / "bacse102_assignments_final.json"
OUTPUT_DIR = PROJECT_ROOT / "encoder_training_data"

VAL_SIZE = 0.20
RANDOM_STATE = 42

EXPECTED_CRITERIA = {
    "irrelevancy",
    "solution_proximity",
    "drift",
}

#Assignment metadata

if not ASSIGNMENT_CONTEXTS_PATH.exists():
    raise FileNotFoundError(
        f"Assignment context JSON not found at:\n{ASSIGNMENT_CONTEXTS_PATH}"
    )

with ASSIGNMENT_CONTEXTS_PATH.open(encoding="utf-8") as f:
    ASSIGNMENT_DATA = json.load(f)

if "assignments" not in ASSIGNMENT_DATA:
    raise ValueError("Assignment context JSON does not contain an 'assignments' key.")

ASSIGNMENTS = ASSIGNMENT_DATA["assignments"]


#Basic helpers

def get_assignment_id(chat_csv_path: Path) -> str:
    """
    Convert a scored CSV filename into the canonical assignment ID.

    Example:
        bacse102_week8_ips_2_p04_p2_scored.csv

    becomes:
        bacse102_week8_ips_2_p04_p2
    """

    assignment_id = chat_csv_path.stem

    if assignment_id.endswith("_scored"):
        assignment_id = assignment_id.removesuffix("_scored")

    return assignment_id


def get_assignment_context(chat_csv_path: Path) -> str:
    """
    Retrieve the assignment context corresponding to a scored conversation CSV.
    """

    assignment_id = get_assignment_id(chat_csv_path)

    if assignment_id not in ASSIGNMENTS:
        raise ValueError(
            f"No assignment entry found for CSV: {chat_csv_path.name}. "
            f"Expected JSON key: {assignment_id!r}"
        )

    entry = ASSIGNMENTS[assignment_id]

    expected_csv = entry.get("csv_file")

    if expected_csv:
        expected_stem = Path(expected_csv).stem

        if expected_stem != assignment_id:
            raise ValueError(
                f"CSV filename mismatch for {assignment_id!r}: "
                f"JSON specifies {expected_csv!r}, but scored file is {chat_csv_path.name!r}"
            )

    title = str(entry.get("title", "")).strip()
    question = str(entry.get("question", "")).strip()

    if not question:
        raise ValueError(f"Assignment {assignment_id!r} has no question text.")

    if title:
        return f"{title}\n\n{question}"

    return question


def parse_label(value) -> int:
    """
    Convert Granite Guardian yes/no labels to binary integers.

        yes -> 1
        no  -> 0
    """

    value = str(value).strip().lower()

    if value == "yes":
        return 1

    if value == "no":
        return 0

    raise ValueError(f"Unexpected label value: {value!r}")


#Conversation parsing

def format_previous_conversation(messages: list) -> str:
    """
    Convert all turns before the final user/assistant exchange
    into deterministic plain text.
    """

    if not messages:
        return ""

    formatted_messages = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Conversation message {index} is not a dictionary.")

        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if role == "user":
            speaker = "USER"
        elif role == "assistant":
            speaker = "ASSISTANT"
        else:
            raise ValueError(f"Unexpected message role {role!r} at message {index}.")

        formatted_messages.append(f"{speaker}:\n{content}")

    return "\n\n".join(formatted_messages)


def split_conversation(raw_conversation):
    """
    Decompose a complete conversation into:

        previous_conversation
        user_query
        assistant_response

    The Granite prompts classify:
        irrelevancy        -> final user message
        solution proximity -> final assistant response
    while using the preceding turns as context.
    """

    try:
        messages = json.loads(raw_conversation)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Conversation field is not valid JSON.") from exc

    if not isinstance(messages, list):
        raise ValueError("Conversation JSON is not a list.")

    if len(messages) < 2:
        raise ValueError("Conversation contains fewer than two messages.")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Conversation message {index} is not a dictionary.")

        if "role" not in message:
            raise ValueError(f"Conversation message {index} has no 'role' field.")

        if "content" not in message:
            raise ValueError(f"Conversation message {index} has no 'content' field.")

    if messages[-2]["role"] != "user":
        raise ValueError(
            f"Expected penultimate message to be 'user', got {messages[-2]['role']!r}."
        )

    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Expected final message to be 'assistant', got {messages[-1]['role']!r}."
        )

    previous_messages = messages[:-2]

    previous_conversation = format_previous_conversation(previous_messages)
    user_query = str(messages[-2]["content"]).strip()
    assistant_response = str(messages[-1]["content"]).strip()

    if not user_query:
        raise ValueError("Final user query is empty.")

    if not assistant_response:
        raise ValueError("Final assistant response is empty.")

    return previous_conversation, user_query, assistant_response


#Individual scored CSV processing

def process_file(csv_path: Path) -> pd.DataFrame:
    """
    Convert one scored assignment CSV into the canonical
    per-conversation representation.

    Input:
        id
        criterion
        conversation
        score
        trace

    Output:
        assignment_id
        conversation_id
        assignment_context
        previous_conversation
        user_query
        assistant_response
        irrelevancy
        solution_proximality
        drift
        source_file
    """

    df = pd.read_csv(csv_path)

    required_columns = {
        "id",
        "criterion",
        "conversation",
        "score",
        "trace",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"{csv_path.name} contains no rows.")

    assignment_id = get_assignment_id(csv_path)
    assignment_context = get_assignment_context(csv_path)

    df["criterion"] = df["criterion"].astype(str).str.strip()

    observed_criteria = set(df["criterion"].unique())
    unexpected_criteria = observed_criteria - EXPECTED_CRITERIA

    if unexpected_criteria:
        raise ValueError(
            f"{csv_path.name} contains unexpected criteria: {sorted(unexpected_criteria)}"
        )

    duplicate_mask = df.duplicated(subset=["id", "criterion"], keep=False)

    if duplicate_mask.any():
        duplicate_rows = (
            df.loc[duplicate_mask, ["id", "criterion"]]
            .sort_values(["id", "criterion"])
        )

        raise ValueError(
            f"Duplicate criterion scores found in {csv_path.name}:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    conversation_variants = df.groupby("id")["conversation"].nunique(dropna=False)
    inconsistent_ids = conversation_variants[conversation_variants != 1]

    if not inconsistent_ids.empty:
        raise ValueError(
            f"{csv_path.name} contains conversation IDs whose transcript differs "
            f"between criterion rows: {inconsistent_ids.index.tolist()}"
        )

    criteria_per_conversation = df.groupby("id")["criterion"].apply(set)

    for conversation_id, criteria_set in criteria_per_conversation.items():
        missing = EXPECTED_CRITERIA - criteria_set

        if missing:
            raise ValueError(
                f"{csv_path.name}, conversation {conversation_id!r} "
                f"is missing criteria: {sorted(missing)}"
            )

    wide = (df.pivot(index=["id", "conversation"],columns="criterion",values="score").reset_index())
    wide.columns.name = None

    rows = []

    for _, row in wide.iterrows():
        try:
            previous_conversation, user_query, assistant_response = split_conversation(
                row["conversation"]
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse conversation {row['id']!r} in {csv_path.name}."
            ) from exc

        rows.append({
            "assignment_id": assignment_id,
            "conversation_id": row["id"],
            "assignment_context": assignment_context,
            "previous_conversation": previous_conversation,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "irrelevancy": parse_label(row["irrelevancy"]),
            "solution_proximality": parse_label(row["solution_proximity"]),
            "drift": parse_label(row["drift"]),
            "source_file": csv_path.name,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(f"No conversations were produced from {csv_path.name}.")

    return result



#Load complete real dataset

def load_all_assignments() -> pd.DataFrame:
    """
    Process every scored CSV and concatenate them into one
    canonical master dataset.
    """

    if not SCORED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Scored dataset directory not found:\n{SCORED_DATASET_DIR}"
        )

    csv_files = sorted(SCORED_DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{SCORED_DATASET_DIR}"
        )

    print(f"Found {len(csv_files)} scored CSV files.")

    all_frames = []

    for file_number, csv_path in enumerate(csv_files, start=1):
        print(f"[{file_number}/{len(csv_files)}] Processing {csv_path.name}")

        assignment_df = process_file(csv_path)
        all_frames.append(assignment_df)

    master_df = pd.concat(all_frames, ignore_index=True)

    duplicate_samples = master_df.duplicated(
        subset=["assignment_id", "conversation_id"],
        keep=False
    )

    if duplicate_samples.any():
        duplicates = master_df.loc[
            duplicate_samples,
            ["assignment_id", "conversation_id", "source_file"]
        ]

        raise ValueError(
            "Duplicate assignment/conversation pairs found after concatenation:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return master_df



#Assignment-level train/validation split

def split_by_assignment(master_df: pd.DataFrame):
    """
    Split at assignment level.

    No assignment may appear in both train and validation.
    """

    assignment_ids = (
        master_df["assignment_id"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(assignment_ids) < 2:
        raise ValueError("Need at least two assignments to create train/validation splits.")

    train_ids, val_ids = train_test_split(
        assignment_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    if not train_ids.isdisjoint(val_ids):
        raise RuntimeError("Assignment leakage detected between train and validation.")

    train_df = (
        master_df[master_df["assignment_id"].isin(train_ids)]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        master_df[master_df["assignment_id"].isin(val_ids)]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, val_df, train_ids, val_ids



#Classifier-specific datasets

def make_irrelevancy_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset consumed by the irrelevancy encoder classifier.
    """

    result = df[
        [
            "conversation_id",
            "assignment_id",
            "assignment_context",
            "previous_conversation",
            "user_query",
            "irrelevancy",
            "source_file",
        ]
    ].copy()

    result = result.rename(columns={"conversation_id": "id"})
    return result


def make_solution_proximality_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset consumed by the solution proximality encoder classifier.
    """

    result = df[
        [
            "conversation_id",
            "assignment_id",
            "assignment_context",
            "previous_conversation",
            "user_query",
            "assistant_response",
            "solution_proximality",
            "source_file",
        ]
    ].copy()

    result = result.rename(columns={"conversation_id": "id"})
    return result


#Output

def save_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, train_ids: set, val_ids: set):
    """
    Save:
        master train/val
        irrelevancy train/val
        solution proximality train/val
        assignment split manifest
    """

    master_dir = OUTPUT_DIR / "master"
    irrelevancy_dir = OUTPUT_DIR / "irrelevancy"
    proximality_dir = OUTPUT_DIR / "solution_proximality"

    master_dir.mkdir(parents=True, exist_ok=True)
    irrelevancy_dir.mkdir(parents=True, exist_ok=True)
    proximality_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(master_dir / "train.csv", index=False)
    val_df.to_csv(master_dir / "val.csv", index=False)

    irrelevancy_train = make_irrelevancy_dataset(train_df)
    irrelevancy_val = make_irrelevancy_dataset(val_df)

    irrelevancy_train.to_csv(irrelevancy_dir / "train.csv", index=False)
    irrelevancy_val.to_csv(irrelevancy_dir / "val.csv", index=False)

    proximality_train = make_solution_proximality_dataset(train_df)
    proximality_val = make_solution_proximality_dataset(val_df)

    proximality_train.to_csv(proximality_dir / "train.csv", index=False)
    proximality_val.to_csv(proximality_dir / "val.csv", index=False)

    manifest_rows = []

    for assignment_id in sorted(train_ids):
        manifest_rows.append({
            "assignment_id": assignment_id,
            "split": "train",
        })

    for assignment_id in sorted(val_ids):
        manifest_rows.append({
            "assignment_id": assignment_id,
            "split": "val",
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)


#Reporting

def print_label_distribution(df: pd.DataFrame, label_column: str, split_name: str):
    counts = df[label_column].value_counts().sort_index()
    proportions = df[label_column].value_counts(normalize=True).sort_index()

    negative_count = int(counts.get(0, 0))
    positive_count = int(counts.get(1, 0))

    negative_pct = float(proportions.get(0, 0.0)) * 100
    positive_pct = float(proportions.get(1, 0.0)) * 100

    print(f"  {split_name}:")
    print(f"    no  / 0: {negative_count:,} ({negative_pct:.2f}%)")
    print(f"    yes / 1: {positive_count:,} ({positive_pct:.2f}%)")


def print_summary(master_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Print dataset sizes and label balance after splitting.
    """

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    total_assignments = master_df["assignment_id"].nunique()
    train_assignments = train_df["assignment_id"].nunique()
    val_assignments = val_df["assignment_id"].nunique()

    print("\nAssignments")
    print(f"  Total:      {total_assignments}")
    print(f"  Train:      {train_assignments}")
    print(f"  Validation: {val_assignments}")

    print("\nConversations")
    print(f"  Total:      {len(master_df):,}")
    print(f"  Train:      {len(train_df):,}")
    print(f"  Validation: {len(val_df):,}")

    total_conversations = len(master_df)

    if total_conversations:
        train_pct = len(train_df) / total_conversations * 100
        val_pct = len(val_df) / total_conversations * 100

        print("\nConversation split")
        print(f"  Train:      {train_pct:.2f}%")
        print(f"  Validation: {val_pct:.2f}%")

    for label_column in ["irrelevancy", "solution_proximality", "drift"]:
        print(f"\n{label_column.upper()}")

        print_label_distribution(train_df, label_column, "Train")
        print_label_distribution(val_df, label_column, "Validation")

    print("\n" + "=" * 70)

def main():
    print(f"Project root:\n  {PROJECT_ROOT}")
    print(f"\nScored dataset:\n  {SCORED_DATASET_DIR}")
    print(f"\nAssignment contexts:\n  {ASSIGNMENT_CONTEXTS_PATH}")
    print(f"\nOutput directory:\n  {OUTPUT_DIR}\n")

    master_df = load_all_assignments()

    print(
        f"\nLoaded {len(master_df):,} conversations from "
        f"{master_df['assignment_id'].nunique()} assignments."
    )

    train_df, val_df, train_ids, val_ids = split_by_assignment(master_df)

    train_assignment_ids = set(train_df["assignment_id"].unique())
    val_assignment_ids = set(val_df["assignment_id"].unique())

    overlap = train_assignment_ids & val_assignment_ids

    if overlap:
        raise RuntimeError(
            f"Assignment leakage detected after split: {sorted(overlap)}"
        )

    save_datasets(train_df, val_df, train_ids, val_ids)
    print_summary(master_df, train_df, val_df)

    print(f"\nProcessed datasets saved to:\n  {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()