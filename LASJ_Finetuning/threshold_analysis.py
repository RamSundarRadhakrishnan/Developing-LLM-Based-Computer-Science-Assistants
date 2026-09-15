from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PREDICTIONS_PATH = Path(
    "../granite_guardian_results/solution_proximity_predictions.csv"
)
TRUE_LABEL_COL = "solution_proximality_label"
SCORE_COL = "solution_proximity_risk_probability"
OUTPUT_PATH = Path(
    "../granite_guardian_results/solution_proximity_threshold_sweep.csv"
)


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    required = {TRUE_LABEL_COL, SCORE_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=[SCORE_COL]).copy()

    y_true = df[TRUE_LABEL_COL].astype(int).to_numpy()
    scores = df[SCORE_COL].astype(float).to_numpy()

    rows = []

    for threshold in np.linspace(0.00, 1.00, 101):
        y_pred = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        rows.append({
            "threshold": round(float(threshold), 2),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            ),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUTPUT_PATH, index=False)

    print(sweep.to_string(index=False))

    candidates = sweep[sweep["recall"] >= 0.90]

    if not candidates.empty:
        chosen = candidates.sort_values(
            ["false_positive_rate", "threshold"]
        ).iloc[0]

        print("\nBest threshold with recall >= 0.90:")
        print(chosen.to_string())
    else:
        print("\nNo threshold reached recall >= 0.90.")


if __name__ == "__main__":
    main()