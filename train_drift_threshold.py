#!/usr/bin/env python3
"""Train a drift F1 threshold on grouped half-data; evaluate on the other half.

Run: python train_drift_threshold.py
Uses the existing drift detector, with no encoder training. Dependencies:
numpy scikit-learn matplotlib torch transformers python-dotenv.
All artifacts go to drift_threshold_training_results/ by default.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from test_drift_detection import (
    DEFAULT_DIRS, ROOT, load_rows, metrics, save_plots, score_rows, turn_text, write_csv,
)


def group_key(row):
    return " ".join(row["anchor"].split())


def split_rows(rows, seed=42):
    """Choose by count balance alone, never by model scores or performance."""
    groups = np.array([group_key(r) for r in rows])
    labels = np.array([r["y_true"] for r in rows])
    sources = np.array([r["dataset"] for r in rows])
    source_names = sorted(set(sources))
    best = None
    splitter = GroupShuffleSplit(n_splits=1000, test_size=.5, random_state=seed)
    for order, (train, test) in enumerate(splitter.split(rows, groups=groups)):
        if any(len(set(labels[indices])) != 2 or set(sources[indices]) != set(source_names)
               for indices in (train, test)):
            continue
        deviation = (abs(len(train) - len(rows) / 2)
                     + abs(int(labels[train].sum()) - labels.sum() / 2)
                     + sum(abs(int((sources[train] == name).sum()) - (sources == name).sum() / 2)
                           for name in source_names))
        if best is None or (deviation, order) < best[:2]:
            best = (float(deviation), order, train, test)
    if best is None:
        raise ValueError("No valid grouped split among 1,000 candidates: need both classes and sources in both halves")
    deviation, candidate, train, test = best
    assert not set(groups[train]) & set(groups[test])
    assert set(train) | set(test) == set(range(len(rows))) and not set(train) & set(test)
    return train, test, {"seed": seed, "candidate_count": 1000, "selected_candidate_zero_based": candidate,
                        "count_deviation_objective": deviation, "train_rows": len(train), "test_rows": len(test),
                        "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])),
                        "train_positives": int(labels[train].sum()), "test_positives": int(labels[test].sum())}


def threshold_candidates(scores):
    """Enumerate every attainable decision set; keep equal similarities together."""
    similarities = np.array([r["min_anchor_similarity"] for r in scores], dtype=np.float64)
    labels = np.array([r["y_true"] for r in scores])
    if not np.isfinite(similarities).all() or set(labels) != {0, 1}:
        raise ValueError("Threshold training needs finite scores and both classes")
    unique = np.unique(similarities)
    thresholds = np.concatenate(([unique[0]], np.nextafter(unique, np.inf)))
    result = []
    for threshold in thresholds:
        pred = similarities < threshold
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        tn = int(np.sum(~pred & (labels == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn)
        result.append({"threshold": float(threshold), "precision": precision, "recall": recall,
                       "f1": 2 * tp / (2 * tp + fp + fn), "youden_j": recall - fp / (fp + tn),
                       "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def fit_thresholds(scores):
    candidates = threshold_candidates(scores)
    best_f1 = max(candidates, key=lambda r: (r["f1"], r["precision"], -r["threshold"]))
    best_j = max(candidates, key=lambda r: (r["youden_j"], -r["threshold"]))
    return {"f1": best_f1["threshold"], "youden_j": best_j["threshold"]}, candidates


def predict(scores, threshold):
    predictions = []
    for score in scores:
        row = dict(score)
        row["y_pred"] = int(row["min_anchor_similarity"] < threshold)
        row["outcome"] = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[row["y_true"], row["y_pred"]]
        predictions.append(row)
    return predictions


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def comparison_plots(summary, candidates, thresholds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name in ("precision", "recall", "f1"):
        ax.step([r["threshold"] for r in candidates], [r[name] for r in candidates], where="post", label=name)
    for name, threshold in thresholds.items():
        ax.axvline(threshold, linestyle="--", label=f"{name}: {threshold:.6f}")
    ax.set(title="Training-only threshold selection", xlabel="Anchor similarity threshold", ylabel="Metric", ylim=(0, 1.05))
    ax.legend()
    fig.savefig(output / "training_threshold_selection.png", dpi=170)
    plt.close(fig)
    datasets = list(dict.fromkeys(r["dataset"] for r in summary))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False, constrained_layout=True)
    names = ["precision", "recall", "f1", "accuracy"]
    for ax, dataset in zip(axes[0], datasets):
        for offset, method in [(-.2, "f1"), (.2, "youden_j")]:
            row = next(r for r in summary if r["dataset"] == dataset and r["method"] == method)
            ax.bar(np.arange(len(names)) + offset, [row[n] for n in names], width=.4, label=method)
        ax.set(xticks=np.arange(len(names)), xticklabels=names, ylim=(0, 1.05), title=dataset.replace("_turnwise_openai", ""))
        ax.legend()
    fig.suptitle("Two thresholds evaluated on the same held-out test rows")
    fig.savefig(output / "test_metric_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dirs", nargs="+", type=Path, default=DEFAULT_DIRS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "drift_threshold_training_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    directories = [p.resolve() for p in args.dataset_dirs]
    output = args.output_dir.resolve()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if len({d.name for d in directories}) != len(directories) or any(d.name == "combined" for d in directories):
        parser.error("Dataset names must be unique and cannot be combined")
    if any(output == d or output in d.parents or d in output.parents for d in directories):
        parser.error("Output directory must be separate from input directories")
    rows = load_rows(directories)
    train_indices, test_indices, split = split_rows(rows, args.seed)
    print(f"Split: {split}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    train_set = set(train_indices)
    manifest = []
    for index, row in enumerate(rows):
        manifest.append({"dataset": row["dataset"], "source_file": row["source_file"], "id": row["id"],
                         "assignment_id": row["assignment_id"], "y_true": row["y_true"],
                         "assignment_group": hashlib.sha256(group_key(row).encode()).hexdigest(),
                         "split": "train" if index in train_set else "test"})
    write_csv(output / "split_manifest.csv", manifest)

    from encoder_classifier.drift import embed_texts, load_embedding_model
    texts = list(dict.fromkeys(text for row in rows for text in
                 [row["anchor"], *[turn_text(turn, "assistant") for turn in row["turns"]]]))
    print(f"Loading {args.model}; embedding {len(texts)} unique texts", flush=True)
    tokenizer, model, device = load_embedding_model(args.model, args.device)
    embeddings = {}
    truncated = 0
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        ids = tokenizer(batch, truncation=False, padding=False, verbose=False)["input_ids"]
        truncated += sum(len(tokens) > 256 for tokens in ids)
        vectors = embed_texts(batch, tokenizer, model, device, batch_size=args.batch_size, max_length=256)
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite embedding encountered")
        embeddings.update(zip(batch, vectors))
        if start % (args.batch_size * 20) == 0:
            print(f"Embedded {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    train_scores = score_rows([rows[i] for i in train_indices], embeddings, "assistant")
    thresholds, candidates = fit_thresholds(train_scores)
    write_csv(output / "training_scores.csv", train_scores)
    write_csv(output / "training_threshold_candidates.csv", candidates)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "similarity_threshold": thresholds["f1"],
                "baseline_threshold": thresholds["youden_j"], "objective": "maximize drift F1",
                "f1_tie_break": "higher precision, then lower threshold",
                "youden_tie_break": "lower threshold", "comparison": "min_anchor_similarity < threshold",
                "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
                "turn_repr": "assistant", "pooling": "masked mean, L2-normalized", "max_length": 256,
                "device": str(device), "split": split, "unique_texts": len(texts), "truncated_texts": truncated,
                "training_metrics": {name: metrics(predict(train_scores, threshold)) for name, threshold in thresholds.items()},
                "datasets": [str(d) for d in directories],
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in directories for p in sorted(d.rglob('*.csv'))},
                "notes": ["Previously explored data; this is not an untouched external benchmark.",
                          "Turn-weighted metrics; conversation prefixes are correlated.",
                          "Current-turn labels can disagree with cumulative worst-turn detection.",
                          "Scores are negative similarities, not probabilities.",
                          "Undefined class precision/recall/F1 use zero; undefined ROC AUC uses null."]}
    # Persist learned cutoffs BEFORE inspecting test labels in any metric calculation.
    write_json(output / "threshold.json", metadata)
    test_scores = score_rows([rows[i] for i in test_indices], embeddings, "assistant")
    summary = []
    for method, threshold in thresholds.items():
        method_dir = output / method
        method_dir.mkdir(exist_ok=True)
        predictions = predict(test_scores, threshold)
        write_csv(method_dir / "test_predictions.csv", predictions)
        for name in [d.name for d in directories] + ["combined"]:
            subset = predictions if name == "combined" else [r for r in predictions if r["dataset"] == name]
            report = metrics(subset)
            dest = method_dir / name
            dest.mkdir(exist_ok=True)
            write_json(dest / "metrics.json", report)
            save_plots(subset, report, threshold, dest, training_rows=train_scores)
            summary.append({"method": method, "dataset": name, "threshold": threshold,
                            **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}})
            print(f"{method} / {name}: P={report['precision']:.4f}, R={report['recall']:.4f}, "
                  f"F1={report['f1']:.4f}, accuracy={report['accuracy']:.4f}, AUC={report['roc_auc']}", flush=True)
    write_csv(output / "metrics_summary.csv", summary)
    comparison_plots(summary, candidates, thresholds, output)
    print(f"Thresholds: {thresholds}; truncated texts: {truncated}/{len(texts)}\nSaved to {output}")


if __name__ == "__main__":
    main()
