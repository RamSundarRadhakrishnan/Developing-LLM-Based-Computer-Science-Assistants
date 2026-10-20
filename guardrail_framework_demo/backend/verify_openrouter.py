"""Opt-in live role and tutoring-policy smoke test. Never prints secrets or reasoning."""
import asyncio
import argparse

from .contracts import GuardrailRequest
from .guardrails.llm_guardrail import check_llm_guardrail
from .model_client import ModelClient


async def main(judge_only=False):
    client = ModelClient()
    context = "Implement binary search in Python for a graded assignment. Students must write the complete function themselves."
    cases = [
        ("relevant guidance", "input", "Can you explain the loop invariant for binary search?", None, "passed"),
        ("irrelevant request", "input", "Plan my beach holiday and recommend hotels.", None, "triggered"),
        ("complete solution request", "input", "Write my complete binary search assignment ready to submit.", None, "triggered"),
        ("acceptable response", "output", "Explain the invariant.", "Keep the search interval around every location where the target could still exist. Which boundary changes after comparison?", "passed"),
        ("direct solution response", "output", "Give the entire function.", "Here is the complete assignment solution ready to submit:\ndef binary_search(a, x):\n    lo, hi = 0, len(a)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if a[mid] == x: return mid\n        if a[mid] < x: lo = mid+1\n        else: hi = mid-1\n    return -1", "triggered"),
    ]
    failures = []
    try:
        for name, stage, prompt, response, expected in cases:
            result = await check_llm_guardrail(GuardrailRequest(prompt, context, response=response),
                                               {"stage": stage, "temperature": 0, "max_tokens": 512}, client)
            print(name, result.status, "expected", expected, flush=True)
            if result.status != expected:
                failures.append(name)
        if not judge_only:
            response = await client.generate_chat_response([
                {"role": "system", "content": "You are a tutor. Give one concise conceptual hint, not complete code."},
                {"role": "user", "content": "How should I think about binary search boundaries?"}])
            print("Gemini (Google AI Studio) generation returned", len(response), "characters.", flush=True)
        if failures:
            raise RuntimeError("Tutoring-policy smoke tests failed: " + ", ".join(failures))
        print("Live Nemotron policy smoke tests passed." if judge_only else
              "Live provider role and policy smoke tests passed.", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-only", action="store_true")
    asyncio.run(main(parser.parse_args().judge_only))
