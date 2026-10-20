#!/usr/bin/env python3
"""Fit and evaluate three thresholds on ALL rows: IN-SAMPLE EVALUATION.

Run: python compare_drift_threshold_objectives.py
No encoder training and no train/test split. Requires numpy, scipy, scikit-learn,
matplotlib, torch, transformers and python-dotenv. Similarities are recomputed
from the model, avoiding reliance on previously saved scores.
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from test_drift_detection import DEFAULT_DIRS, ROOT, load_rows, metrics, save_plots, score_rows, turn_text, write_csv
from train_drift_threshold import predict, threshold_candidates, write_json

EVALUATION = "In-sample evaluation"
TEMPERATURE = .1
BOUNDS = (-1.000001, 1.000001)
OPT_OPTIONS = {"xatol": 1e-12, "maxiter": 1000}


def class_weights(labels):
    y = np.asarray(labels, dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Both classes are required")
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (2 * counts.astype(float))
    return weights[y], weights


def weighted_bce(threshold, similarities, labels, weights):
    logits = (threshold - similarities) / TEMPERATURE
    # log(1 + exp(z)) - y*z, stable for both large and small z.
    return float(np.average(np.logaddexp(0., logits) - labels * logits, weights=weights))


def fit_objectives(scores):
    similarities = np.array([r["min_anchor_similarity"] for r in scores], dtype=np.float64)
    y = np.array([r["y_true"] for r in scores])
    weights, per_class = class_weights(y)
    candidates = threshold_candidates(scores)
    max_equivalence_error = 0.
    for row in candidates:
        threshold = row["threshold"]
        pred = similarities < threshold
        j = (weights[(y == 1) & pred].sum() / weights[y == 1].sum()
             - weights[(y == 0) & pred].sum() / weights[y == 0].sum())
        max_equivalence_error = max(max_equivalence_error, abs(j - row["youden_j"]))
        if not np.isclose(j, row["youden_j"], atol=1e-12, rtol=0):
            raise AssertionError("Class-weighted Youden differs from ordinary Youden")
        # Use the algebraically equivalent rate calculation to avoid floating
        # summation noise changing the lower-threshold tie break.
        row["weighted_youden_j"] = row["youden_j"]
        negative_f1 = 2 * row["tn"] / (2 * row["tn"] + row["fp"] + row["fn"])
        row["macro_f1"] = (negative_f1 + row["f1"]) / 2
        row["weighted_bce"] = weighted_bce(threshold, similarities, y, weights)
    loss = lambda threshold: weighted_bce(threshold, similarities, y, weights)
    optimum = minimize_scalar(loss, bounds=BOUNDS, method="bounded", options=OPT_OPTIONS)
    if not optimum.success:
        raise RuntimeError(f"BCE threshold optimization failed: {optimum.message}")
    bce_threshold = min([BOUNDS[0], float(optimum.x), BOUNDS[1]], key=lambda t: (loss(t), t))
    selected = {
        "class_weighted_youden": max(candidates, key=lambda r: (r["weighted_youden_j"], -r["threshold"]))["threshold"],
        "weighted_cross_entropy": bce_threshold,
        "macro_f1": max(candidates, key=lambda r: (r["macro_f1"], -r["threshold"]))["threshold"],
    }
    details = {"class_weights": {"no_drift": float(per_class[0]), "drift": float(per_class[1])},
               "class_weight_totals": {str(c): float(weights[y == c].sum()) for c in (0, 1)},
               "youden_equivalence_max_error": max_equivalence_error,
               "temperature": TEMPERATURE, "optimizer": {"method": "bounded", "bounds": BOUNDS,
               "options": OPT_OPTIONS, "success": bool(optimum.success), "nfev": int(optimum.nfev),
               "minimum_weighted_bce": loss(bce_threshold)},
               "tie_break": "lower threshold", "thresholds": selected}
    return selected, candidates, details


def make_comparison_plots(summary, candidates, thresholds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, method, field in zip(axes, thresholds, ["weighted_youden_j", "weighted_bce", "macro_f1"]):
        x = [r["threshold"] for r in candidates]
        values = [r[field] for r in candidates]
        ax.plot(x, values)
        ax.axvline(thresholds[method], color="black", linestyle="--", label=f"Selected: {thresholds[method]:.6f}")
        ax.set(title=method.replace('_', ' '), xlabel="Similarity threshold", ylabel=field)
        ax.legend()
    fig.suptitle(f"{EVALUATION}: threshold fitting objectives")
    fig.savefig(output / "objective_curves.png", dpi=170)
    plt.close(fig)
    datasets = list(dict.fromkeys(r["dataset"] for r in summary))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False, constrained_layout=True)
    names = ["precision", "recall", "f1", "macro_f1", "accuracy"]
    for ax, dataset in zip(axes[0], datasets):
        for index, method in enumerate(thresholds):
            row = next(r for r in summary if r["dataset"] == dataset and r["method"] == method)
            ax.bar(np.arange(len(names)) + (index - 1) * .25, [row[n] for n in names], width=.25, label=method)
        ax.set(xticks=np.arange(len(names)), xticklabels=names, ylim=(0, 1.05), title=dataset.replace('_turnwise_openai', ''))
        ax.legend(fontsize=8)
    fig.suptitle(f"{EVALUATION}: all methods fitted and evaluated on identical rows")
    fig.savefig(output / "metric_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dirs", nargs="+", type=Path, default=DEFAULT_DIRS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "drift_threshold_objective_comparison")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    directories = [d.resolve() for d in args.dataset_dirs]
    output = args.output_dir.resolve()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if len({d.name for d in directories}) != len(directories) or any(d.name == 'combined' for d in directories):
        parser.error("Dataset names must be unique and cannot be combined")
    if any(output == d or output in d.parents or d in output.parents for d in directories):
        parser.error("Output must be separate from input folders")
    rows = load_rows(directories)
    keys = [(r['dataset'], r['source_file'], r['id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate row identities")
    class_weights([r['y_true'] for r in rows])
    print(f"{EVALUATION}: fitting AND evaluating on all {len(rows)} rows; no split.", flush=True)
    # Always recompute: no saved score is trusted without provenance validation.
    from encoder_classifier.drift import embed_texts, load_embedding_model
    texts = list(dict.fromkeys(text for row in rows for text in
                 [row['anchor'], *[turn_text(t, 'assistant') for t in row['turns']]]))
    tokenizer, model, device = load_embedding_model(args.model, args.device)
    embeddings = {}
    truncated = 0
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        ids = tokenizer(batch, truncation=False, padding=False, verbose=False)['input_ids']
        truncated += sum(len(tokens) > 256 for tokens in ids)
        vectors = embed_texts(batch, tokenizer, model, device, batch_size=args.batch_size, max_length=256)
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite embeddings")
        embeddings.update(zip(batch, vectors))
        if start % (args.batch_size * 20) == 0:
            print(f"Embedded {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    scores = score_rows(rows, embeddings, 'assistant')
    thresholds, candidates, fitting = fit_objectives(scores)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'scores.csv', [{'evaluation': EVALUATION, **r} for r in scores])
    write_csv(output / 'threshold_candidates.csv', [{'evaluation': EVALUATION, **r} for r in candidates])
    metadata = {'evaluation': EVALUATION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'fit_rows': len(rows), 'evaluation_rows': len(rows), 'split': None,
                'model': args.model, 'model_revision': getattr(model.config, '_commit_hash', None),
                'encoder_trained': False, 'device': str(device), 'max_length': 256,
                'turn_repr': 'assistant', 'pooling': 'masked mean, L2-normalized',
                'decision': 'min_anchor_similarity < threshold', 'unique_texts': len(texts),
                'truncated_texts': truncated, 'score_source': 'recomputed from model', 'fitting': fitting,
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in directories for p in sorted(d.rglob('*.csv'))},
                'notes': ['In-sample fit, not evidence of unseen-conversation performance.',
                          'Inverse-frequency class weights do not balance assignments.',
                          'Class-weighted Youden equals ordinary Youden because within-class constants cancel.',
                          'BCE sigmoid outputs are not assumed to be calibrated probabilities.',
                          'Current-turn labels may disagree with cumulative minimum similarity.',
                          'Overlapping conversation prefixes are correlated.',
                          'Undefined precision/recall/F1 use zero; undefined ROC AUC uses null.']}
    write_json(output / 'thresholds.json', metadata)
    summary = []
    for method, threshold in thresholds.items():
        method_dir = output / method
        method_dir.mkdir(exist_ok=True)
        predictions = predict(scores, threshold)
        write_csv(method_dir / 'predictions.csv', [{'evaluation': EVALUATION, **r} for r in predictions])
        for name in [d.name for d in directories] + ['combined']:
            subset = predictions if name == 'combined' else [r for r in predictions if r['dataset'] == name]
            if not subset:
                continue
            report = metrics(subset)
            dest = method_dir / name
            dest.mkdir(exist_ok=True)
            write_json(dest / 'metrics.json', {'evaluation': EVALUATION, **report})
            save_plots(subset, report, threshold, dest, training_rows=scores, evaluation_label=EVALUATION)
            summary.append({'evaluation': EVALUATION, 'method': method, 'dataset': name, 'threshold': threshold,
                            **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}})
            print(f"{EVALUATION} / {method} / {name}: P={report['precision']:.4f}, R={report['recall']:.4f}, "
                  f"F1={report['f1']:.4f}, macro-F1={report['macro_f1']:.4f}", flush=True)
    write_csv(output / 'metrics_summary.csv', summary)
    combined = sorted([r for r in summary if r['dataset'] == 'combined'], key=lambda r: -r['macro_f1'])
    ranks = []
    rank = 1
    previous = None
    for i, row in enumerate(combined):
        if previous is not None and not np.isclose(row['macro_f1'], previous, atol=1e-12, rtol=0):
            rank = i + 1
        ranks.append({'evaluation': EVALUATION, 'rank': rank, 'method': row['method'], 'macro_f1': row['macro_f1']})
        previous = row['macro_f1']
    write_csv(output / 'ranking.csv', ranks)
    make_comparison_plots(summary, candidates, thresholds, output)
    print(f"In-sample ranking: {ranks}\nSaved to {output}")


if __name__ == '__main__':
    main()
