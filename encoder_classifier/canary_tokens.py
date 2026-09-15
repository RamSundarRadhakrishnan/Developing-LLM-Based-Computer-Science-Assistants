"""Canary-token prompt exfiltration guardrail.

A canary is inserted into the model prompt and checked in model output. Keep
the token secret and create a new one per request when possible.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Optional, Tuple


DEFAULT_CANARY_FORMAT = "<!-- {canary_word} -->"
QWEN_CODER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def generate_canary_word(length: int = 8) -> str:
    """Generate a cryptographically random hexadecimal canary.

    ``length`` is the number of characters, matching Rebuff's default.
    """
    if not isinstance(length, int) or length < 2 or length % 2:
        raise ValueError("length must be an even integer >= 2")
    return secrets.token_hex(length // 2)


def add_canary_word(
    prompt: str,
    canary_word: Optional[str] = None,
    canary_format: str = DEFAULT_CANARY_FORMAT,
) -> Tuple[str, str]:
    """Prepend a canary comment to a string prompt and return both values."""

    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, not {type(prompt).__name__}")
    token = canary_word or generate_canary_word()
    if not token or any(character.isspace() for character in token):
        raise ValueError("canary_word must be a non-empty, whitespace-free string")
    return f"{canary_format.format(canary_word=token)}\n{prompt}", token


def is_canary_word_leaked(completion: str, canary_word: str) -> bool:
    """Return ``True`` when the exact canary occurs in model output."""

    if not isinstance(completion, str):
        raise TypeError(f"completion must be str, not {type(completion).__name__}")
    return bool(canary_word) and canary_word in completion


def is_canary_tokens_leaked(
    completion: str,
    canary_word: str,
    tokenizer: Any = None,
) -> float:
    """Estimate leakage as the fraction of canary token pieces in ``completion``.

    The default tokenizer is Qwen3 Coder's tokenizer.  Each token ID from the
    canary is decoded separately and searched in the completion.  Whitespace-
    only pieces are ignored because they are not useful leakage evidence.

    Args:
        completion: Model-generated text to inspect.
        canary_word: Secret canary inserted into the protected prompt.
        tokenizer: Optional compatible tokenizer. If omitted, Transformers
            loads ``QWEN_CODER_MODEL`` on first use.

    Returns:
        A float from ``0.0`` to ``1.0``. This is a heuristic evidence score,
        not a calibrated statistical probability.
    """
    if not isinstance(completion, str):
        raise TypeError(f"completion must be str, not {type(completion).__name__}")
    if not isinstance(canary_word, str) or not canary_word:
        return 0.0

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "is_canary_tokens_leaked requires transformers. Install it "
                "with `pip install transformers`."
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(QWEN_CODER_MODEL)

    token_ids = tokenizer.encode(canary_word, add_special_tokens=False)
    token_pieces = {
        tokenizer.decode([token_id], skip_special_tokens=True)
        for token_id in token_ids
    }
    token_pieces = {piece for piece in token_pieces if piece.strip()}
    if not token_pieces:
        return 0.0

    matched_pieces = sum(piece in completion for piece in token_pieces)
    return matched_pieces / len(token_pieces)


@dataclass(frozen=True)
class CanaryCheck:
    canary: str
    leaked: bool


class CanaryTokenGuardrail:
    """Stateful convenience wrapper for one prompt/model interaction."""

    def __init__(self, length: int = 8, canary_format: str = DEFAULT_CANARY_FORMAT) -> None:
        self.length = length
        self.canary_format = canary_format

    def protect(self, prompt: str) -> Tuple[str, str]:
        return add_canary_word(prompt, generate_canary_word(self.length), self.canary_format)

    def check(self, output: str, canary: str) -> CanaryCheck:
        return CanaryCheck(canary, is_canary_word_leaked(output, canary))


__all__ = [
    "CanaryCheck",
    "CanaryTokenGuardrail",
    "DEFAULT_CANARY_FORMAT",
    "add_canary_word",
    "generate_canary_word",
    "is_canary_word_leaked",
    "is_canary_tokens_leaked",
    "QWEN_CODER_MODEL",
]


if __name__ == "__main__":
    test_cases = [
        {
            "prompt": "Summarize this algorithm.",
            "canary": "alpha1234",
            "output": "The algorithm runs in linear time.",
            "expected": False,
        },
        {
            "prompt": "Return the requested result.",
            "canary": "bravo5678",
            "output": "The hidden marker was bravo5678.",
            "expected": True,
        },
        {
            "prompt": "Calculate the requested result.",
            "canary": "charlie9012",
            "output": "The result is 42.",
            "expected": False,
        },
        {
            "prompt": "Inspect the document and summarize it.",
            "canary": "delta3456",
            "output": "The document contains delta3456.",
            "expected": True,
        },
        {
            "prompt": "Generate a short answer.",
            "canary": None,
            "output": "A short answer without the marker.",
            "expected": False,
        },
    ]

    guardrail = CanaryTokenGuardrail(length=12)
    for index, case in enumerate(test_cases, start=1):
        protected_prompt, canary = add_canary_word(
            case["prompt"], case["canary"]
        )
        assert canary in protected_prompt
        check = guardrail.check(case["output"], canary)
        assert check.leaked == case["expected"], (
            f"case {index} expected {case['expected']}, got {check}"
        )
        print(
            f"case {index}: canary={canary} leaked={check.leaked}"
        )

    print("canary token demonstration passed")

