"""Recreate the repository's trained encoder architecture and input formatting."""
import json
import threading
from pathlib import Path

from ..contracts import GuardrailRequest, GuardrailResult


def format_classifier_text(request: GuardrailRequest, stage: str) -> str:
    previous = "\n".join(f"User: {t.prompt}\nAssistant: {t.response}" for t in request.conversation)
    text = (f"[ASSIGNMENT]\n{request.assignment_context}\n\n"
            f"[PREVIOUS CONVERSATION]\n{previous}\n\n"
            f"[CURRENT USER QUERY]\n{request.prompt}\n\n")
    if stage == "output":
        text += f"[CURRENT ASSISTANT RESPONSE]\n{request.response or ''}"
    return text


class EncoderClassifier:
    def __init__(self, checkpoint_dir: str):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        directory = Path(checkpoint_dir)
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        backbone_config = AutoConfig.from_pretrained(self.config["encoder"])

        class Classifier(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.encoder = AutoModel.from_config(backbone_config, attn_implementation="eager")
                inner.head = nn.Sequential(
                    nn.Linear(backbone_config.hidden_size, self.config["head_hidden"]), nn.GELU(),
                    nn.Dropout(self.config["dropout_p"]), nn.Linear(self.config["head_hidden"], 2))

            def forward(inner, input_ids, attention_mask):
                hidden = inner.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                return inner.head(hidden)

        self.model = Classifier()
        state = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def predict(self, text: str, risk_label: str) -> tuple[float, bool]:
        import torch

        risk_id = next(int(key) for key, value in self.config["label_map"].items() if value == risk_label)
        limit = int(self.config["max_length"])
        with self.lock, torch.inference_mode():
            truncated = len(self.tokenizer(text, truncation=False)["input_ids"]) > limit
            inputs = self.tokenizer(text, max_length=limit, truncation=True, return_tensors="pt")
            logits = self.model(input_ids=inputs["input_ids"].to(self.device),
                                attention_mask=inputs["attention_mask"].to(self.device))
            return float(torch.softmax(logits, dim=-1)[0, risk_id].cpu()), truncated


def check_encoder_classification(request: GuardrailRequest, settings: dict, classifier: EncoderClassifier) -> GuardrailResult:
    label = settings["label"]
    score, truncated = classifier.predict(format_classifier_text(request, settings["stage"]), label)
    threshold = settings.get("threshold", 0.5)
    triggered = score >= threshold
    return GuardrailResult(status="triggered" if triggered else "passed", label=label if triggered else "clear",
        score=score, reason=f"Encoder {label} probability {'meets' if triggered else 'is below'} the threshold.",
        details={"threshold": threshold, "truncated": truncated, "device": classifier.device})
