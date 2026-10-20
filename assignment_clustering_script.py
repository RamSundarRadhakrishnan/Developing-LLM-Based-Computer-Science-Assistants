#!/usr/bin/env python3
"""Cluster assignment questions using semantic embeddings and average linkage.

Place this file in Developing-LLM-Based-Computer-Science-Assistants/, alongside
the lab_exercise_dataset/ directory.

Install:
    python -m pip install "sentence-transformers>=3.0" "scikit-learn>=1.2" numpy

Run:
    python cluster_assignments.py --clusters 10

The embedding model is downloaded on the first run. Source CSVs are not changed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def load_assignments(dataset_dir):
    """Read each assignment once, irrespective of its conversation-row count."""
    assignments = {}
    files = sorted(dataset_dir.rglob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {dataset_dir}")

    # Conversation fields can exceed the csv module's default field-size limit.
    csv.field_size_limit(100_000_000)
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "assignment_context" not in reader.fieldnames:
                raise ValueError(f"Missing assignment_context column: {path}")
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV record near row {line_number}: {path}")
                context = row["assignment_context"].strip()
                if not context:
                    raise ValueError(f"Empty assignment context near row {line_number}: {path}")
                assignment_id = (row.get("assignment_id") or path.stem).strip()
                # Normalize whitespace only; retain technical terms and requirements.
                text = " ".join(context.split())
                if assignment_id not in assignments:
                    title = context.splitlines()[0].strip()
                    if title.startswith("Assignment Title:"):
                        title = title[len("Assignment Title:"):].strip()
                    assignments[assignment_id] = {
                        "assignment_id": assignment_id,
                        "title": title,
                        "assignment_context": context,
                        "text": text,
                        "source_files": set(),
                    }
                elif assignments[assignment_id]["text"] != text:
                    raise ValueError(
                        f"Assignment {assignment_id!r} has different contexts. "
                        "Resolve these before clustering."
                    )
                assignments[assignment_id]["source_files"].add(
                    str(path.relative_to(dataset_dir))
                )

    if not assignments:
        raise ValueError("The CSV files contain no assignments.")
    result = []
    for assignment_id in sorted(assignments):
        record = assignments[assignment_id]
        record["source_files"] = sorted(record["source_files"])
        result.append(record)
    return result


def embed_assignments(assignments, model, batch_size):
    """Chunk long questions to avoid silently losing their final requirements."""
    special_tokens = model.tokenizer.num_special_tokens_to_add(pair=False)
    # Leave a small margin for re-tokenization after decoding chunks.
    chunk_size = int(model.max_seq_length) - special_tokens - 16
    if chunk_size < 1:
        raise ValueError("The embedding model has an invalid sequence-length limit.")

    chunks, owners, weights = [], [], []
    for index, assignment in enumerate(assignments):
        token_ids = model.tokenizer.encode(
            assignment["text"], add_special_tokens=False, truncation=False
        )
        for start in range(0, len(token_ids), chunk_size):
            part = token_ids[start:start + chunk_size]
            chunks.append(model.tokenizer.decode(part, skip_special_tokens=True))
            owners.append(index)
            weights.append(len(part))

    chunk_embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Token-weighted mean yields one vector per assignment, then normalize again.
    embeddings = np.zeros(
        (len(assignments), chunk_embeddings.shape[1]), dtype=np.float64
    )
    totals = np.zeros(len(assignments), dtype=np.float64)
    for owner, weight, vector in zip(owners, weights, chunk_embeddings):
        embeddings[owner] += weight * vector
        totals[owner] += weight
    if np.any(totals == 0):
        raise ValueError("An assignment produced no tokens.")
    embeddings /= totals[:, None]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding vector encountered.")
    return embeddings / norms


def save_clusters(assignments, labels, output_dir, model_name):
    groups = {}
    for assignment, label in zip(assignments, labels):
        groups.setdefault(int(label), []).append(assignment)
    # Give groups readable, stable IDs ordered by their first assignment ID.
    ordered_groups = sorted(
        groups.values(), key=lambda members: members[0]["assignment_id"]
    )
    clusters = []
    for cluster_id, members in enumerate(ordered_groups, start=1):
        clusters.append({
            "cluster_id": cluster_id,
            "assignment_count": len(members),
            "assignments": [
                {key: value for key, value in item.items() if key != "text"}
                for item in members
            ],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "clusters.json"
    json_path.write_text(json.dumps({
        "embedding_model": model_name,
        "clustering": {"metric": "cosine", "linkage": "average"},
        "assignment_count": len(assignments),
        "cluster_count": len(clusters),
        "clusters": clusters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "assignment_clusters.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "assignment_id", "title", "source_files"])
        for cluster in clusters:
            for assignment in cluster["assignments"]:
                writer.writerow([
                    cluster["cluster_id"], assignment["assignment_id"],
                    assignment["title"], "; ".join(assignment["source_files"]),
                ])

    for cluster in clusters:
        print(f"\nCluster {cluster['cluster_id']} ({cluster['assignment_count']} assignments)")
        for assignment in cluster["assignments"]:
            print(f"  - {assignment['assignment_id']}: {assignment['title']}")
    print(f"\nSaved: {json_path}\nSaved: {csv_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters (default: 10).")
    parser.add_argument("--dataset-dir", type=Path, default=base_dir / "lab_exercise_dataset")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "assignment_clusters")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu or cuda.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_dir}")
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        parser.error("Choose an output directory outside the input dataset directory.")
    assignments = load_assignments(dataset_dir)
    if not 1 <= args.clusters <= len(assignments):
        parser.error(f"--clusters must be between 1 and {len(assignments)}.")
    print(f"Loaded {len(assignments)} unique assignments.")
    model = SentenceTransformer(args.model, device=args.device)
    embeddings = embed_assignments(assignments, model, args.batch_size)
    if len(assignments) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=args.clusters,
            metric="cosine",
            linkage="average",
        ).fit_predict(embeddings)
    save_clusters(assignments, labels, output_dir, args.model)


if __name__ == "__main__":
    main()
