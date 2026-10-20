"""Real tokenizer parity and synthetic full/partial leakage verification."""
from .contracts import GuardrailRequest
from .guardrails.canary import check_canary_leakage, protect_system_prompt


def main():
    # Public synthetic fixture, never a real session marker.
    token = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    request = GuardrailRequest("prompt", "assignment", response=token[8:24], metadata={"canary": token})
    partial_result = check_canary_leakage(request, {})
    assert partial_result.status == "triggered" and not partial_result.details["full_match"]
    for text in ("A conceptual hint.", "0123456789 abcdef " * 100,
                 " | ".join(token[index:index + 4] for index in range(0, len(token), 4))):
        request.response = text
        assert check_canary_leakage(request, {}).status == "passed"
    request.response = token
    assert check_canary_leakage(request, {}).status == "triggered"
    protected = protect_system_prompt("Assignment tutoring policy. " * 30, request, {})
    marker = f"[{token}]"
    assert protected.startswith(marker) and protected.endswith(marker)
    assert all(len(chunk) <= 50 for chunk in protected.split(marker))
    print("Full/partial detection, false-positive regressions, and system-prompt distribution passed.", flush=True)
    print("Partial score:", round(partial_result.score, 3), flush=True)


if __name__ == "__main__":
    main()
