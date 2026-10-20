#!/usr/bin/env python3
"""Evaluate the existing drift detector against turnwise CSV labels.

Run: python test_drift_detection.py
Inspect input without loading a model: python test_drift_detection.py --validate-only
Use an existing similarity cutoff: python test_drift_detection.py --threshold 0.35
Dependencies: numpy matplotlib scikit-learn torch transformers python-dotenv

Default calibration uses encoder_classifier/drift_synthetic_dataset_raahul.csv
when present; otherwise 20% of assignment groups are reserved for calibration.
Use --calibration-csv for an explicit separate calibration dataset. Never fit on
test labels. Each test row is the full conversation prefix through its
current exchange. The decision is min(anchor similarities) < threshold, matching
drift.py. Labels describe current-turn drift, so recovery after an earlier off-topic
turn may disagree with this cumulative detector. Metrics are turn-weighted, not
independent conversation estimates. Scores are negative similarities, NOT probabilities.
"""

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DIRS = [ROOT / "golden_dataset_turnwise_openai",
                ROOT / "lab_exercise_dataset_turnwise_openai"]
ROLE = re.compile(r"^(USER|ASSISTANT):[ \t]*\r?$", re.MULTILINE)


def parse_history(text):
    """Parse the exported standalone USER:/ASSISTANT: message delimiters."""
    if not text.strip():
        return []
    markers = list(ROLE.finditer(text))
    if not markers or text[:markers[0].start()].strip() or len(markers) % 2:
        raise ValueError("History must contain alternating USER:/ASSISTANT: blocks")
    messages = []
    for i, marker in enumerate(markers):
        expected = "USER" if i % 2 == 0 else "ASSISTANT"
        if marker.group(1) != expected:
            raise ValueError(f"Expected {expected} history block at position {i}")
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        messages.append(text[marker.end():end].strip())
    return list(zip(messages[::2], messages[1::2]))


def load_rows(directories):
    csv.field_size_limit(100_000_000)
    required = {"id", "assignment_context", "previous_conversation",
                "user_query", "assistant_response", "drift"}
    result = []
    for directory in directories:
        files = sorted(directory.rglob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files in {directory}")
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing columns in {path}: {required - set(reader.fieldnames or [])}")
                seen = set()
                for line, row in enumerate(reader, 2):
                    try:
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError("Malformed CSV record")
                        label = row["drift"].strip().lower()
                        if label not in {"yes", "no"}:
                            raise ValueError(f"Invalid drift label: {label!r}")
                        if not row["id"].strip() or row["id"] in seen:
                            raise ValueError("Empty or duplicate row ID within file")
                        seen.add(row["id"])
                        if not row["assignment_context"].strip() or not row["assistant_response"].strip():
                            raise ValueError("Empty assignment context or assistant response")
                        turns = parse_history(row["previous_conversation"])
                        turns.append((row["user_query"].strip(), row["assistant_response"].strip()))
                        result.append({"dataset": directory.name,
                                       "source_file": str(path.relative_to(directory)),
                                       "id": row["id"], "assignment_id": row.get("assignment_id", ""),
                                       "anchor": row["assignment_context"].strip(),
                                       "turns": turns, "y_true": int(label == "yes")})
                    except ValueError as exc:
                        raise ValueError(f"{path}, CSV record {line}: {exc}") from exc
    return result


def turn_text(turn, mode):
    user, assistant = turn
    return assistant if mode == "assistant" else user if mode == "user" else f"User: {user}\nAssistant: {assistant}"


def score_rows(rows, embeddings, mode):
    predictions = []
    for row in rows:
        anchor = embeddings[row["anchor"]]
        vectors = [embeddings[turn_text(t, mode)] for t in row["turns"]]
        sims = [float(np.dot(anchor, vector)) for vector in vectors]
        adjacent = [float(np.dot(a, b)) for a, b in zip(vectors, vectors[1:])]
        predictions.append({k: row[k] for k in ("dataset", "source_file", "id", "assignment_id", "y_true")}
                           | {"turn_index": len(vectors), "min_anchor_similarity": min(sims),
                              "current_anchor_similarity": sims[-1], "risk_score": -min(sims),
                              "worst_turn_index": int(np.argmin(sims)) + 1,
                              "min_adjacent_similarity": min(adjacent) if adjacent else None})
    return predictions


def choose_threshold(labels, similarities):
    """Youden J; convert ROC's inclusive boundary to drift.py's strict '<'."""
    fpr, tpr, cutoffs = roc_curve(labels, -np.asarray(similarities))
    index = int(np.argmax(tpr - fpr))
    raw = float(-cutoffs[index])
    threshold = float(np.nextafter(raw, np.inf)) if np.isfinite(raw) else -1.000001
    return threshold, {"youden_j": float(tpr[index] - fpr[index]),
                       "tpr": float(tpr[index]), "fpr": float(fpr[index])}


def metrics(rows):
    y = np.array([r["y_true"] for r in rows])
    pred = np.array([r["y_pred"] for r in rows])
    score = np.array([r["risk_score"] for r in rows])
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = len(np.unique(y)) == 2
    specificity = tn / (tn + fp) if tn + fp else None
    sensitivity = recall_score(y, pred, zero_division=0)
    return {"n": len(rows), "positive_count": int(y.sum()), "prevalence": float(y.mean()),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(sensitivity), "f1": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "specificity": specificity,
            "balanced_accuracy": (float(sensitivity) + specificity) / 2 if both else None,
            "mcc": float(matthews_corrcoef(y, pred)) if both else None,
            "roc_auc": float(roc_auc_score(y, score)) if both else None,
            "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "classification_report": classification_report(y, pred, labels=[0, 1],
                target_names=["no_drift", "drift"], output_dict=True, zero_division=0)}


def save_plots(rows, report, threshold, directory, training_rows=None, evaluation_label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    y = np.array([r["y_true"] for r in rows])
    score = np.array([r["risk_score"] for r in rows])
    cm = np.asarray(report["confusion_matrix"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    ConfusionMatrixDisplay(cm, display_labels=["No drift", "Drift"]).plot(ax=axes[0, 0], colorbar=False)
    axes[0, 0].set_title("Confusion matrix (counts)")
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True),
                           out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    ConfusionMatrixDisplay(normalized, display_labels=["No drift", "Drift"]).plot(
        ax=axes[0, 1], colorbar=False, values_format=".2f")
    axes[0, 1].set_title("Confusion matrix (row proportions)")
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, score)
        axes[0, 2].plot(fpr, tpr, label=f"AUC = {report['roc_auc']:.3f}")
        axes[0, 2].scatter([report["fp"] / (report["fp"] + report["tn"])],
                           [report["recall"]], label="Fixed operating threshold")
        axes[0, 2].legend()
    else:
        axes[0, 2].text(.1, .5, "ROC undefined: only one class")
    axes[0, 2].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0, 2].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    if y.sum():
        precision, recall, _ = precision_recall_curve(y, score)
        axes[1, 0].plot(recall, precision, label=f"AP = {report['average_precision']:.3f}")
        axes[1, 0].scatter([report["recall"]], [report["precision"]])
        axes[1, 0].axhline(y.mean(), linestyle="--", color="gray", label="Prevalence baseline")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(.1, .5, "PR undefined: no positive labels")
    axes[1, 0].set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision", ylim=(0, 1.05))
    bins = np.linspace(-1, 1, 41)
    for label, name in [(0, "No drift"), (1, "Drift")]:
        if np.any(y == label):
            axes[1, 1].hist(-score[y == label], bins=bins, alpha=.55, density=True, label=name)
    axes[1, 1].axvline(threshold, color="black", linestyle="--", label="Threshold")
    axes[1, 1].set(title="Minimum anchor similarity by true class", xlabel="Similarity", ylabel="Density")
    axes[1, 1].legend()
    names = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"]
    values = [report[n] or 0 for n in names]
    axes[1, 2].barh(names, values)
    axes[1, 2].set(title="Metrics at the fixed threshold", xlim=(0, 1.12))
    for i, name in enumerate(names):
        axes[1, 2].text(values[i] + .01, i, f"{report[name]:.3f}" if report[name] is not None else "N/A", va="center")
    fig.suptitle(f"{evaluation_label + ' — ' if evaluation_label else ''}{directory.name}: {len(rows)} turns; drift prevalence {y.mean():.1%}")
    fig.savefig(directory / "evaluation_dashboard.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sweep_rows = rows if training_rows is None else training_rows
    sweep_y = np.array([r["y_true"] for r in sweep_rows])
    sweep_similarity = np.array([r["min_anchor_similarity"] if "min_anchor_similarity" in r else -r["risk_score"]
                                 for r in sweep_rows], dtype=np.float64)
    thresholds = np.linspace(-1, 1, 201)
    for name, fn in [("Precision", precision_score), ("Recall", recall_score), ("F1", f1_score)]:
        axes[0].plot(thresholds, [fn(sweep_y, sweep_similarity < t, zero_division=0) for t in thresholds], label=name)
    axes[0].axvline(threshold, color="black", linestyle="--", label="Calibration threshold")
    axes[0].set(title="In-sample threshold sweep" if evaluation_label == "In-sample evaluation" else
                "Test threshold sweep (diagnostic only)" if training_rows is None else "Training-only threshold sweep",
                xlabel="Similarity threshold", ylabel="Metric")
    axes[0].legend()
    turns = np.array([r["turn_index"] for r in rows])
    indices = sorted(set(turns))
    axes[1].plot(indices, [y[turns == t].mean() for t in indices], label="Labeled drift rate")
    axes[1].plot(indices, [np.mean(-score[turns == t] < threshold) for t in indices], label="Predicted drift rate")
    axes[1].set(title="Drift rate by prefix length (support varies)", xlabel="Turn index", ylabel="Proportion", ylim=(0, 1.05))
    axes[1].legend()
    if evaluation_label:
        fig.suptitle(evaluation_label)
    fig.savefig(directory / "threshold_and_turn_diagnostics.png", dpi=170)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dirs", nargs="+", type=Path, default=DEFAULT_DIRS)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, help="Fixed anchor similarity cutoff; bypass calibration")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--turn-repr", choices=["assistant", "user", "full"], default="assistant")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "drift_test_results")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.threshold is not None and not -1 <= args.threshold <= 1):
        parser.error("Batch size must be positive; threshold must be between -1 and 1")
    if not 0 < args.calibration_fraction < 1:
        parser.error("Calibration fraction must be between 0 and 1")
    if args.threshold is not None and args.calibration_csv is not None:
        parser.error("Specify either --threshold or --calibration-csv, not both")
    directories = [p.resolve() for p in args.dataset_dirs]
    if len({p.name for p in directories}) != len(directories) or any(p.name == "combined" for p in directories):
        parser.error("Dataset directory names must be unique and cannot be 'combined'")
    output = args.output_dir.resolve()
    if any(output == p or p in output.parents or output in p.parents for p in directories):
        parser.error("Output directory must be separate from the input directories")
    rows = load_rows(directories)
    for directory in directories:
        subset = [r for r in rows if r["dataset"] == directory.name]
        print(f"{directory.name}: {len(subset)} turns, labels {dict(Counter(r['y_true'] for r in subset))}", flush=True)
    if not rows:
        parser.error("No labeled rows found")
    if args.validate_only:
        print("Input validation passed; no model loaded or output written.")
        return

    from encoder_classifier.drift import embed_texts, load_embedding_model, load_labeled_examples_from_csv, parse_conversation
    calibration = []
    calibration_source = None
    input_rows = len(rows)
    if args.threshold is None:
        conventional = ROOT / "encoder_classifier/drift_synthetic_dataset_raahul.csv"
        if args.calibration_csv is None and conventional.exists():
            args.calibration_csv = conventional
    if args.threshold is None and args.calibration_csv is None:
        from sklearn.model_selection import GroupShuffleSplit
        groups = [" ".join(r["anchor"].split()) for r in rows]
        if len(set(groups)) < 2:
            parser.error("Need at least two assignment groups, or supply a calibration CSV/threshold")
        splitter = GroupShuffleSplit(n_splits=100, test_size=args.calibration_fraction, random_state=args.seed)
        for test_indices, calibration_indices in splitter.split(rows, groups=groups):
            candidate = [rows[i] for i in calibration_indices]
            held_out = [rows[i] for i in test_indices]
            if (len({r['y_true'] for r in candidate}) == 2
                    and len({r['y_true'] for r in held_out}) == 2
                    and {r['dataset'] for r in held_out} == {d.name for d in directories}):
                calibration, rows = candidate, held_out
                break
        else:
            parser.error("Could not create a grouped split with both classes; supply --calibration-csv or --threshold")
        calibration_source = "assignment-group holdout from input folders"
        print(f"Reserved {len(calibration)} calibration turns; evaluating {len(rows)} held-out turns", flush=True)
    elif args.threshold is None:
        if not args.calibration_csv.is_file():
            parser.error(f"Calibration CSV not found: {args.calibration_csv}")
        if args.calibration_csv.resolve() in {p.resolve() for d in directories for p in d.rglob('*.csv')}:
            parser.error("Calibration CSV must not be a test input")
        for example in load_labeled_examples_from_csv(args.calibration_csv):
            turns = [(t.user, t.assistant) for t in parse_conversation(example["conversation"])]
            if not turns:
                raise ValueError("Calibration contains an empty conversation")
            calibration.append({"dataset": "calibration", "source_file": str(args.calibration_csv),
                                "id": example["id"], "assignment_id": "", "anchor": example["assignment_context"].strip(),
                                "turns": turns, "y_true": int(example["drift"] == "yes")})
        if len({r["y_true"] for r in calibration}) != 2:
            raise ValueError("Calibration requires both drift and no-drift examples")
        fingerprint = lambda r: (r["anchor"], tuple(r["turns"]))
        if {fingerprint(r) for r in calibration} & {fingerprint(r) for r in rows}:
            raise ValueError("Exact conversation overlap between calibration and test data")
        calibration_source = str(args.calibration_csv.resolve())

    texts = list(dict.fromkeys(text for r in rows + calibration
                              for text in [r["anchor"], *[turn_text(t, args.turn_repr) for t in r["turns"]]]))
    print(f"Loading {args.model}; embedding {len(texts)} unique texts (max_length=256)...", flush=True)
    tokenizer, model, device = load_embedding_model(args.model, args.device)
    embeddings = {}
    truncated = 0
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        lengths = tokenizer(batch, truncation=False, padding=False, verbose=False)["input_ids"]
        truncated += sum(len(ids) > 256 for ids in lengths)
        vectors = embed_texts(batch, tokenizer, model, device, batch_size=args.batch_size, max_length=256)
        embeddings.update(zip(batch, vectors))
        if start % (args.batch_size * 20) == 0:
            print(f"Embedded {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    calibration_info = {"method": "supplied threshold"}
    threshold = args.threshold
    if calibration:
        scores = score_rows(calibration, embeddings, args.turn_repr)
        threshold, calibration_info = choose_threshold([r["y_true"] for r in scores],
                                                       [r["min_anchor_similarity"] for r in scores])
        calibration_info.update(method="Youden J", source=calibration_source, n=len(scores),
                                seed=args.seed, requested_group_fraction=args.calibration_fraction)
    predictions = score_rows(rows, embeddings, args.turn_repr)
    for row in predictions:
        row["y_pred"] = int(row["min_anchor_similarity"] < threshold)
        row["outcome"] = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[row["y_true"], row["y_pred"]]
    output.mkdir(parents=True, exist_ok=True)
    if calibration:
        write_csv(output / "calibration_scores.csv", scores)
    write_csv(output / "predictions.csv", predictions)
    groups = {d.name: [r for r in predictions if r["dataset"] == d.name] for d in directories}
    groups["combined"] = predictions
    summary = []
    for name, subset in groups.items():
        if not subset:
            continue
        directory = output / name
        directory.mkdir(exist_ok=True)
        report = metrics(subset)
        (directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        save_plots(subset, report, threshold, directory)
        summary.append({"dataset": name, **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}})
        print(f"{name}: precision={report['precision']:.4f}, recall={report['recall']:.4f}, "
              f"F1={report['f1']:.4f}, accuracy={report['accuracy']:.4f}, ROC AUC={report['roc_auc']}")
    write_csv(output / "metrics_summary.csv", summary)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
                "device": str(device), "turn_repr": args.turn_repr, "max_length": 256,
                "similarity_threshold": threshold, "comparison": "min_anchor_similarity < threshold",
                "calibration": calibration_info, "unique_texts": len(texts), "truncated_texts": truncated,
                "datasets": [str(p) for p in directories],
                "input_rows": input_rows, "evaluated_rows": len(rows), "calibration_rows": len(calibration),
                "notes": ["Metrics are turn-weighted; overlapping prefixes are correlated.",
                          "Current-turn labels may disagree with cumulative worst-turn detection.",
                          "ROC and PR use negative minimum similarity, not predicted labels.",
                          "Test threshold sweeps are diagnostic only; do not tune on test results.",
                          "Undefined ROC/MCC/balanced accuracy are null; undefined precision/recall/F1 use zero."]}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Threshold: {threshold:.8f}; truncated unique texts: {truncated}/{len(texts)}")
    print(f"Reports saved to {output}")


if __name__ == "__main__":
    main()
