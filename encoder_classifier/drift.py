"""
drift.py

Session-level context-drift detection for the tutoring guardrail.

Approach:
  - Sentence embeddings via a model fine-tuned for semantic similarity
    (sentence-transformers/all-MiniLM-L6-v2), using mean pooling over
    token embeddings (NOT the raw [CLS] token, which is anisotropic and
    unreliable for cosine similarity on vanilla pretrained encoders).
  - Anchor-distance tracking: each turn is compared against the assignment
    context embedding, to catch cumulative drift away from the assignment
    even when no single adjacent-turn jump looks large.
  - Adjacent-turn distance: each turn is also compared against the
    previous turn, to catch sudden topic jumps as a complementary signal.
  - Threshold calibration: rather than guessing a cosine-similarity cutoff,
    the threshold is derived empirically from a labeled multi-turn dataset
    (drift: yes / no), using ROC analysis (Youden's J statistic) on the
    per-conversation worst-case (minimum) anchor similarity.

Usage:
    python drift.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Model loading and embedding
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
    """Load tokenizer + model for sentence embeddings.

    Returns (tokenizer, model, device).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean pooling over token embeddings.

    This is the pooling strategy sentence-transformers models are trained
    and evaluated with. Using the raw [CLS] token here would reintroduce
    the anisotropy problem this whole module exists to avoid.
    """
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """Embed a list of texts, mean-pooled and L2-normalized.

    Returns an (N, hidden_size) numpy array. Normalized vectors mean cosine
    similarity reduces to a plain dot product downstream.
    """
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        model_output = model(**encoded)
        pooled = _mean_pooling(model_output, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())

    return np.vstack(all_embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two already-normalized vectors (dot product)."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Conversation representation
# ---------------------------------------------------------------------------

TurnRepr = Literal["assistant", "user", "full"]


@dataclass
class Turn:
    user: str
    assistant: str

    def text(self, repr_mode: TurnRepr = "assistant") -> str:
        if repr_mode == "assistant":
            return self.assistant
        if repr_mode == "user":
            return self.user
        return f"User: {self.user}\nAssistant: {self.assistant}"


@dataclass
class DriftTrace:
    """Per-turn drift signals for one conversation."""

    anchor_similarities: List[float] = field(default_factory=list)   # turn[i] vs assignment context
    adjacent_similarities: List[float] = field(default_factory=list)  # turn[i] vs turn[i-1], len = n_turns - 1
    min_anchor_similarity: float = 1.0
    flagged_turns: List[int] = field(default_factory=list)  # turn indices below threshold


# ---------------------------------------------------------------------------
# Core drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Embeds a conversation and scores drift against an assignment anchor.

    threshold semantics: a turn is flagged as drifted if its anchor
    cosine similarity falls BELOW `anchor_threshold` (i.e. it has moved
    too far, in embedding space, from the assignment context).
    """

    def __init__(
        self,
        tokenizer=None,
        model=None,
        device: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        anchor_threshold: Optional[float] = None,
        adjacent_threshold: Optional[float] = None,
        turn_repr: TurnRepr = "assistant",
    ):
        if tokenizer is None or model is None:
            tokenizer, model, device = load_embedding_model(model_name, device)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor_threshold = anchor_threshold
        self.adjacent_threshold = adjacent_threshold
        self.turn_repr = turn_repr

    def _embed(self, texts: List[str]) -> np.ndarray:
        return embed_texts(texts, self.tokenizer, self.model, self.device)

    def trace(self, conversation: List[Turn], assignment_context: str) -> DriftTrace:
        """Compute anchor-distance and adjacent-turn-distance signals for one conversation."""
        turn_texts = [t.text(self.turn_repr) for t in conversation]
        anchor_embedding = self._embed([assignment_context])[0]
        turn_embeddings = self._embed(turn_texts)

        anchor_sims = [cosine_sim(turn_embeddings[i], anchor_embedding) for i in range(len(turn_embeddings))]
        adjacent_sims = [
            cosine_sim(turn_embeddings[i], turn_embeddings[i - 1]) for i in range(1, len(turn_embeddings))
        ]

        flagged = []
        if self.anchor_threshold is not None:
            flagged = [i for i, sim in enumerate(anchor_sims) if sim < self.anchor_threshold]

        return DriftTrace(
            anchor_similarities=anchor_sims,
            adjacent_similarities=adjacent_sims,
            min_anchor_similarity=min(anchor_sims) if anchor_sims else 1.0,
            flagged_turns=flagged,
        )

    def detect(self, conversation: List[Turn], assignment_context: str) -> Dict:
        """Return a drift verdict for the conversation, plus the underlying trace."""
        if self.anchor_threshold is None:
            raise ValueError(
                "anchor_threshold is not set. Calibrate it with "
                "calibrate_threshold_from_labeled_data(...) first, or pass one explicitly."
            )

        t = self.trace(conversation, assignment_context)
        drifted = t.min_anchor_similarity < self.anchor_threshold

        # Adjacent-turn distance is reported as a complementary signal (sudden jumps),
        # not folded into the primary verdict, since it catches a different failure shape.
        sudden_jump = False
        if self.adjacent_threshold is not None and t.adjacent_similarities:
            sudden_jump = min(t.adjacent_similarities) < self.adjacent_threshold

        return {
            "drift": "yes" if drifted else "no",
            "min_anchor_similarity": t.min_anchor_similarity,
            "worst_turn_index": int(np.argmin(t.anchor_similarities)) if t.anchor_similarities else None,
            "anchor_similarities": t.anchor_similarities,
            "adjacent_similarities": t.adjacent_similarities,
            "sudden_jump_detected": sudden_jump,
            "flagged_turns": t.flagged_turns,
        }


# ---------------------------------------------------------------------------
# Threshold calibration from labeled data
# ---------------------------------------------------------------------------

def parse_conversation(raw_conversation) -> List[Turn]:
    """Convert a [{"role": "user"/"assistant", "content": ...}, ...] list into Turn pairs.

    Assumes alternating user/assistant messages starting with user.
    """
    if isinstance(raw_conversation, str):
        raw_conversation = json.loads(raw_conversation)

    turns = []
    i = 0
    while i + 1 < len(raw_conversation):
        user_msg = raw_conversation[i]
        assistant_msg = raw_conversation[i + 1]
        assert user_msg["role"] == "user" and assistant_msg["role"] == "assistant", (
            f"Expected alternating user/assistant at index {i}, got "
            f"{user_msg['role']}/{assistant_msg['role']}"
        )
        turns.append(Turn(user=user_msg["content"], assistant=assistant_msg["content"]))
        i += 2
    return turns


def load_labeled_examples_from_csv(csv_path: str | Path) -> List[Dict]:
    """Load drift calibration examples from a CSV dataset.

    The CSV must contain ``id``, ``assignment_context``, ``conversation``,
    and ``drift`` columns. ``conversation`` is expected to be a JSON-encoded
    list of alternating user and assistant messages.
    """
    path = Path(csv_path)
    required_columns = {"id", "assignment_context", "conversation", "drift"}

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is not None:
            reader.fieldnames = [field.strip() for field in reader.fieldnames]
        columns = set(reader.fieldnames or [])
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )

        examples = []
        for row_number, row in enumerate(reader, start=2):
            row = {key.strip(): value for key, value in row.items() if key is not None}
            label = str(row["drift"]).strip().lower()
            if label not in {"yes", "no"}:
                raise ValueError(
                    f"Invalid drift label {row['drift']!r} on CSV row {row_number}; "
                    "expected 'yes' or 'no'"
                )
            if not row["assignment_context"].strip() or not row["conversation"].strip():
                raise ValueError(f"Empty assignment context or conversation on CSV row {row_number}")

            try:
                conversation = json.loads(row["conversation"])
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid conversation JSON on CSV row {row_number}") from error

            # Parse now so malformed or non-alternating records fail before
            # the embedding model is loaded.
            parse_conversation(conversation)
            examples.append(
                {
                    "id": row["id"],
                    "assignment_context": row["assignment_context"],
                    "conversation": conversation,
                    "drift": label,
                }
            )

    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def compute_min_anchor_similarities(
    labeled_examples: List[Dict],
    detector: DriftDetector,
) -> Dict[str, List[float]]:
    """For each labeled conversation, compute its worst-case (minimum) anchor similarity.

    labeled_examples: list of dicts with keys "conversation" (raw messages list or JSON
    string), "assignment_context", and "drift" ("yes"/"no").

    Returns {"yes": [...], "no": [...]}, the per-group distributions of min anchor similarity.
    """
    grouped: Dict[str, List[float]] = {"yes": [], "no": []}
    for ex in labeled_examples:
        turns = parse_conversation(ex["conversation"])
        trace = detector.trace(turns, ex["assignment_context"])
        label = str(ex["drift"]).strip().lower()
        grouped[label].append(trace.min_anchor_similarity)
    return grouped


def calibrate_threshold_from_labeled_data(
    labeled_examples: List[Dict],
    detector: DriftDetector,
) -> Dict:
    """Derive an anchor-similarity threshold from labeled drift:yes/no conversations.

    Method: reduce each conversation to its worst-case (minimum) anchor similarity,
    then find the threshold that maximizes Youden's J statistic (TPR - FPR) via
    sklearn's ROC curve, treating "drift: yes" as the positive class and
    "low similarity" as the positive-predicting direction.

    Mutates `detector.anchor_threshold` to the selected value and also returns it,
    along with the group distributions and achieved TPR/FPR, for inspection.
    """
    from sklearn.metrics import roc_curve

    grouped = compute_min_anchor_similarities(labeled_examples, detector)
    yes_scores = grouped["yes"]
    no_scores = grouped["no"]

    if not yes_scores or not no_scores:
        raise ValueError(
            f"Need at least one example of each label to calibrate a threshold. "
            f"Got {len(yes_scores)} 'yes' and {len(no_scores)} 'no'."
        )

    y_true = [1] * len(yes_scores) + [0] * len(no_scores)
    # Lower similarity => more likely drift, so use negative similarity as the score
    # roc_curve expects "higher score = more positive".
    y_score = [-s for s in yes_scores] + [-s for s in no_scores]

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_neg_sim_threshold = roc_thresholds[best_idx]
    similarity_threshold = -best_neg_sim_threshold

    detector.anchor_threshold = float(similarity_threshold)

    return {
        "threshold": float(similarity_threshold),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "youden_j": float(j_scores[best_idx]),
        "yes_group_stats": {
            "n": len(yes_scores),
            "mean": float(np.mean(yes_scores)),
            "std": float(np.std(yes_scores)),
            "min": float(np.min(yes_scores)),
            "max": float(np.max(yes_scores)),
        },
        "no_group_stats": {
            "n": len(no_scores),
            "mean": float(np.mean(no_scores)),
            "std": float(np.std(no_scores)),
            "min": float(np.min(no_scores)),
            "max": float(np.max(no_scores)),
        },
    }


# ---------------------------------------------------------------------------
# Demonstration / sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
        "drift_synthetic_dataset_raahul.csv"
    )
    labeled_examples = load_labeled_examples_from_csv(dataset_path)
    label_counts = {
        label: sum(example["drift"] == label for example in labeled_examples)
        for label in ("yes", "no")
    }
    print(f"Loaded {len(labeled_examples)} examples from {dataset_path}")
    print(f"Labels: {label_counts}")

    print("Loading embedding model...")
    tokenizer, model, device = load_embedding_model()
    print(f"Loaded {DEFAULT_MODEL_NAME} on {device}\n")

    detector = DriftDetector(tokenizer=tokenizer, model=model, device=device)

    print("=" * 70)
    print("Calibrating threshold from drift_synthetic_dataset_raahul.csv")
    print("=" * 70)
    calib = calibrate_threshold_from_labeled_data(labeled_examples, detector)
    print(json.dumps(calib, indent=2))

    print("=" * 70)
    print(f"Calibrated threshold = {detector.anchor_threshold:.4f}")
    print("=" * 70)

    print("="*28, "DEMONSTRATION", "="*28)

    ASSIGNMENT_CONTEXT = (
            "Assignment: write a Java program that calculates a household's carbon "
            "footprint. Read miles driven, kWh used, and waste generated from user "
            "input using Scanner, and compute emissions using the given arithmetic "
            "formulas for each category."
        )
    

    # -- drift: yes ------------------------------------------------------------
    # Assistant gradually gives ground turn by turn.
    convo_drift = [
        {
            "role": "user",
            "content": "I need help with the subscriber final fee calculation part of the assignment. How do I apply both the festival discount and the additional discount?"
        },
        {
            "role": "assistant",
            "content": "You calculate it using the formula provided: Final Fee = Monthly Fee x [1 - (Festival Discount + Additional Discount)/100]. Make sure your festival discount is validated to be between 0 and 20%."
        },
        {
            "role": "user",
            "content": "Thanks. Also, since we're dealing with monthly subscriptions and fees, do you know how real streaming services like Netflix handle subscription tiers and billing cycles?"
        },
        {
            "role": "assistant",
            "content": "Real streaming services typically use complex billing engines that handle recurring payments, prorated upgrades or downgrades mid-cycle, and localized pricing tiers depending on the region and currency."
        },
        {
            "role": "user",
            "content": "That's pretty interesting. How do those billing systems handle currency conversions if a user travels internationally?"
        },
        {
            "role": "assistant",
            "content": "International billing systems usually rely on real-time foreign exchange rate APIs at the exact timestamp of the transaction authorization, while maintaining a primary billing currency for each user account."
        },
        {
            "role": "user",
            "content": "That makes sense for global apps. By the way, what's the best API library in Java to fetch those live currency exchange rates easily?"
        },
        {
            "role": "assistant",
            "content": "For Java, libraries like Jackson or Gson are great for parsing JSON responses from exchange rate APIs like Open Exchange Rates or Fixer.io, and you can use Java's built-in `HttpClient` to make the GET requests easily."
        }
    ]

    # -- drift: no ------------------------------------------------------------
    # Assistant stays on context.
    convo_no_drift = [
        {
            "role": "user",
            "content": "I have my miles, electricity, and waste inputs in Java. How should I calculate each emission before finding the total footprint?"
        },
        {
            "role": "assistant",
            "content": "Calculate vehicle emission as miles multiplied by 0.40, electricity emission as kWh multiplied by 0.85, and waste emission as waste multiplied by 1.20. Then add those three values to obtain the total footprint."
        },
        {
            "role": "user",
            "content": "How do I check whether the person is Eco-Friendly after calculating the total?"
        },
        {
            "role": "assistant",
            "content": "Use a logical AND condition: the total footprint must be less than 20, the recycled flag must be 1, and the public-transport flag must be 1. Keep those checks in your Java if statement rather than changing the assignment's formula."
        },
        {
            "role": "user",
            "content": "What should I do if the input values are read as integers but the emission multipliers contain decimals?"
        },
        {
            "role": "assistant",
            "content": "Use a numeric type that preserves decimals, such as double, for the calculated emissions and total. The input variables can be converted or stored as doubles before applying the multipliers."
        }
    ]

    labeled_examples = [
        {"conversation": convo_drift, "assignment_context": ASSIGNMENT_CONTEXT, "drift": "yes"},
        {"conversation": convo_no_drift, "assignment_context": ASSIGNMENT_CONTEXT, "drift": "no"},
    ]
    
    for i, example in enumerate(labeled_examples):
        result = detector.detect(
            parse_conversation(example["conversation"]),
            example["assignment_context"],
        )
        print(
            f"  id={i}: expected={example['drift']} "
            f"predicted={result['drift']} "
            f"min_anchor_sim={result['min_anchor_similarity']:.4f} "
            f"worst_turn={result['worst_turn_index']}"
        )
