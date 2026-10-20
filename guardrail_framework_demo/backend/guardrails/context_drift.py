"""Mean-pooled semantic drift detection with fingerprinted dataset calibration."""
import csv
import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from ..contracts import GuardrailRequest, GuardrailResult
from ..settings import BACKEND_DIR


def mean_pool_embeddings(hidden, mask):
    """Attention-mask-weighted mean pooling, followed by L2 normalization."""
    import torch
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def select_anchor_threshold(labels: list[int], similarities: list[float]) -> float:
    from sklearn.metrics import roc_curve
    if set(labels) != {0, 1}:
        raise ValueError("Drift calibration requires both yes and no labels.")
    fpr, tpr, thresholds = roc_curve(labels, -np.asarray(similarities))
    valid = np.isfinite(thresholds)
    index = np.flatnonzero(valid)[np.argmax((tpr - fpr)[valid])]
    return float(-thresholds[index])


class ContextDriftDetector:
    def __init__(self, settings: dict):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name = settings.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.lock = threading.Lock()
        self.embeddings: dict[str, np.ndarray] = {}
        self.threshold = settings.get("anchor_threshold")
        if self.threshold is None:
            self.threshold = self.calibrate(Path(settings["dataset_path"]))

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch
        if not texts:
            return np.empty((0, self.model.config.hidden_size))
        with self.lock, torch.inference_mode():
            missing = list(dict.fromkeys(text for text in texts if text not in self.embeddings))
            for start in range(0, len(missing), 32):
                batch = missing[start:start + 32]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256,
                                         return_tensors="pt").to(self.device)
                output = self.model(**encoded)
                values = mean_pool_embeddings(output.last_hidden_state, encoded["attention_mask"]).cpu().numpy()
                self.embeddings.update(zip(batch, values))
            result = np.stack([self.embeddings[text] for text in texts])
            if len(self.embeddings) > 4096:
                self.embeddings.clear()
            return result

    def calibrate(self, path: Path) -> float:
        fingerprint = hashlib.sha256(path.read_bytes() + self.model_name.encode() + b"assistant:mean:256:v1").hexdigest()
        cache = BACKEND_DIR / ".runtime" / f"drift-{fingerprint}.json"
        if cache.exists():
            try:
                threshold = json.loads(cache.read_text())["anchor_threshold"]
                if isinstance(threshold, (int, float)) and np.isfinite(threshold) and -1 <= threshold <= 1:
                    return float(threshold)
            except (OSError, ValueError, KeyError, TypeError):
                pass  # An interrupted cache write is recoverable by recalibration.
        labels, scores = [], []
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, skipinitialspace=True)
            for row in reader:
                label = row["drift"].strip().lower()
                if label not in {"yes", "no"}:
                    raise ValueError("Invalid drift calibration label.")
                conversation = json.loads(row["conversation"])
                responses = [item["content"] for item in conversation if item["role"] == "assistant"]
                if not responses or not row["assignment_context"].strip():
                    raise ValueError("Drift calibration contains an empty conversation or assignment.")
                vectors = self.embed([row["assignment_context"], *responses])
                scores.append(float(np.min(vectors[1:] @ vectors[0])))
                labels.append(int(label == "yes"))
        threshold = select_anchor_threshold(labels, scores)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"anchor_threshold": threshold, "examples": len(labels)}), encoding="utf-8")
        return threshold

    def evaluate(self, request: GuardrailRequest, settings: dict) -> GuardrailResult:
        turns = [turn.response for turn in request.conversation if not turn.overridden]
        turns.append(request.response or "")
        vectors = self.embed([request.assignment_context, *turns])
        anchor = vectors[1:] @ vectors[0]
        adjacent = np.sum(vectors[2:] * vectors[1:-1], axis=1)
        score = float(np.min(anchor))
        triggered = score < self.threshold
        adjacent_threshold = settings.get("adjacent_threshold")
        return GuardrailResult(status="triggered" if triggered else "passed", label="drift" if triggered else "on_topic",
            score=score, reason="Conversation drifted from its assignment anchor." if triggered else "Conversation remains near its assignment anchor.",
            details={"anchor_threshold": self.threshold, "anchor_similarities": anchor.tolist(),
                     "adjacent_similarities": adjacent.tolist(), "worst_turn_index": int(np.argmin(anchor)),
                     "sudden_jump": bool(adjacent_threshold is not None and len(adjacent) and np.min(adjacent) < adjacent_threshold)})


def check_context_drift(request: GuardrailRequest, settings: dict, detector: ContextDriftDetector) -> GuardrailResult:
    return detector.evaluate(request, settings)
