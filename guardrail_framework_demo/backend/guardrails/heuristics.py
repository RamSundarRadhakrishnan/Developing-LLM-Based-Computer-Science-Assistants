"""Local input attack checks and output-specific leakage/handover checks."""
import re
from difflib import SequenceMatcher

from ..contracts import GuardrailRequest, GuardrailResult

INPUT_RULES = {
    "instruction_override": r"\b(?:ignore|disregard|forget|bypass|skip|omit)\b.{0,80}\b(?:previous|prior|above|earlier|system)\b.{0,40}\b(?:instructions?|rules?|prompt|context)\b",
    "disobey_instructions": r"\b(?:do not|don't)\s+(?:follow|obey)\b.{0,80}\b(?:instructions?|rules?|prompt)\b",
    "prompt_extraction": r"\b(?:reveal|show|print|repeat|output|leak|disclose)\b.{0,80}\b(?:system prompt|hidden prompt|secret instructions?|chain of thought|developer message)\b",
    "unrestricted_role": r"\b(?:act|pretend|roleplay)\s+as\b.{0,80}\b(?:unrestricted|uncensored|jailbreak|DAN)\b",
    "fake_system_role": r"\b(?:developer|system)\s*:",
    "disable_safety": r"\b(?:ignore all safeguards|disable safety|jailbreak)\b",
    "assignment_handover": r"\b(?:write|give|provide|generate|submit|hand me)\b.{0,80}\b(?:complete|full|entire|final)\b.{0,40}\b(?:code|solution|implementation|assignment|project)\b",
    "graded_work": r"\b(?:do|solve|finish|complete)\b.{0,50}\b(?:my|this|the)\b.{0,30}\b(?:homework|assignment|lab|project|coursework)\b",
    "answer_only": r"\b(?:just|only)\b.{0,30}\b(?:give|send|output|return)\b.{0,30}\b(?:the )?(?:code|answer|solution)\b",
}
OUTPUT_RULES = {
    "system_disclosure": r"\b(?:my|the|our)\s+(?:hidden\s+)?(?:system prompt|secret instructions|developer message)\s+(?:is|says|reads|contains)\s*[:=]",
    "submission_handover": r"\b(?:here is|below is|here's)\b.{0,50}\b(?:complete|full|final|ready.to.submit)\b.{0,40}\b(?:assignment|solution|implementation|project)\b",
    "submit_verbatim": r"\b(?:copy and paste|submit this|submit the following)\b.{0,50}\b(?:assignment|homework|submission|code)\b",
}
COMPILED = {stage: {name: re.compile(pattern, re.I | re.S) for name, pattern in rules.items()}
            for stage, rules in [("input", INPUT_RULES), ("output", OUTPUT_RULES)]}
PHRASES = tuple(" ".join(filter(None, (verb, adjective, obj)))
    for verb in ("ignore", "disregard", "skip", "forget", "bypass", "do not follow", "do not obey", "implement", "solve", "provide")
    for adjective in ("", "prior", "previous", "above", "earlier", "complete", "fully")
    for obj in ("instructions", "rules", "context", "conversation", "code", "pseudocode", "python", "java", "solution"))


def phrase_score(text: str) -> float:
    """Sliding-window Rebuff-style evidence score, not a probability."""
    words = re.sub(r"[^\w\s]|_", "", text.lower()).split()
    highest = 0.0
    for phrase in PHRASES:
        parts = phrase.split()
        for index in range(max(0, len(words) - len(parts) + 1)):
            window = words[index:index + len(parts)]
            matched = sum(a == b for a, b in zip(parts, window))
            score = (0.5 + 0.5 * min(matched / 5, 1) -
                     SequenceMatcher(None, phrase, " ".join(window)).ratio() / 10) if matched else 0
            highest = max(highest, score)
    return highest


def check_heuristics(request: GuardrailRequest, settings: dict) -> GuardrailResult:
    stage = settings["stage"]
    text = request.prompt if stage == "input" else request.response or ""
    rules = settings.get("rules", list(COMPILED[stage]))
    matched = [name for name in rules if COMPILED[stage][name].search(text)]
    score = max(float(bool(matched)), phrase_score(text) if stage == "input" else 0)
    triggered = score >= settings.get("threshold", 0.75)
    return GuardrailResult(status="triggered" if triggered else "passed",
        label="policy_match" if triggered else "clear", score=score,
        reason="Heuristic policy indicators detected." if triggered else "No heuristic policy violation found.",
        details={"matched_rules": matched, "threshold": settings.get("threshold", 0.75)})
