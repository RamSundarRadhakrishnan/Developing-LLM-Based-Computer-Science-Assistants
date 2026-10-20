"""Distributed system-prompt canaries with full and contiguous-fragment detection."""
import secrets
from difflib import SequenceMatcher

from ..contracts import GuardrailRequest, GuardrailResult


def is_canary_tokens_leaked(completion: str, canary_word: str, tokenizer=None) -> float:
    """Fraction of the secret exposed in the longest contiguous match.

    The optional tokenizer argument is retained for caller compatibility; token
    pieces are deliberately not used because common characters match prose.
    """
    if not isinstance(completion, str):
        raise TypeError(f"completion must be str, not {type(completion).__name__}")
    if not isinstance(canary_word, str) or not canary_word:
        return 0.0
    token = canary_word.casefold()
    match = SequenceMatcher(None, token, completion.casefold(), autojunk=False).find_longest_match()
    return match.size / len(token)


def protect_system_prompt(system_prompt: str, request: GuardrailRequest, settings: dict) -> str:
    """Repeat the existing canary at both ends and between fixed-size source chunks."""
    token = request.metadata.get("canary")
    if not token:
        raise ValueError("Canary preparation is missing for this turn.")
    spacing = settings.get("spacing_chars", 50)
    if not isinstance(spacing, int) or isinstance(spacing, bool) or spacing < 1:
        raise ValueError("Canary spacing_chars must be a positive integer.")
    marker = f"[{token}]"
    return marker + "".join(system_prompt[start:start + spacing] + marker
                            for start in range(0, len(system_prompt), spacing))


def prepare_canary(request: GuardrailRequest, settings: dict) -> None:
    request.metadata.setdefault("canary", secrets.token_hex(int(settings.get("length", 32)) // 2))


def check_canary_leakage(request: GuardrailRequest, settings: dict, tokenizer=None) -> GuardrailResult:
    token = request.metadata.get("canary")
    if not token:
        raise ValueError("Canary preparation is missing for this turn.")
    response = request.response or ""
    full_match = token.casefold() in response.casefold()
    score = 1.0 if full_match else is_canary_tokens_leaked(response, token)
    threshold = settings.get("partial_threshold", 0.5)
    # A low configured fraction must not make individual letters evidence.
    minimum_fraction = min(8, len(token)) / len(token)
    leaked = full_match or score >= max(threshold, minimum_fraction)
    return GuardrailResult(status="triggered" if leaked else "passed",
        label="leaked" if full_match else "partial_leak" if leaked else "clear", score=score,
        reason="Private canary leaked in model output." if full_match else
               "A contiguous private canary fragment reached the leakage threshold." if leaked else "No canary leakage detected.",
        details={"full_match": full_match, "partial_threshold": threshold})
