"""Distribution, original scoring parity, and real canary API continuation."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from guardrail_framework_demo.backend.chat_service import ChatService
from guardrail_framework_demo.backend.contracts import GuardrailRequest
from guardrail_framework_demo.backend.guardrails.canary import check_canary_leakage, is_canary_tokens_leaked, prepare_canary, protect_system_prompt
from guardrail_framework_demo.backend.main import create_app
from guardrail_framework_demo.backend.session_store import SessionStore
from guardrail_framework_demo.backend.stack_config import ExecutionStep, LayerConfig, StackConfig, load_stack_config
from guardrail_framework_demo.backend.stack_executor import GuardrailStack
from .helpers import FakeCanaryTokenizer, FakeResources, make_stack

TOKEN = "a1b2c3d4e5f60718293a4b5c6d7e8f90"


class CanaryTests(unittest.TestCase):
    def test_common_characters_and_scattered_fragments_do_not_trigger(self):
        responses = [
            "Explain a binary search: compare the middle element and update the bounds. " * 100,
            "0123456789 abcdef " * 100,
            "|".join(TOKEN[index:index + 4] for index in range(0, len(TOKEN), 4)),
        ]
        for response in responses:
            request = GuardrailRequest("prompt", "assignment", response=response, metadata={"canary": TOKEN})
            with self.subTest(response=response[:50]):
                self.assertEqual(check_canary_leakage(request, {}).status, "passed")

    def test_middle_suffix_and_case_insensitive_fragments(self):
        for fragment in (TOKEN[8:24], TOKEN[16:], TOKEN[:16].upper()):
            request = GuardrailRequest("prompt", "assignment", response="Marker: [" + fragment + "]", metadata={"canary": TOKEN})
            result = check_canary_leakage(request, {})
            self.assertEqual(result.status, "triggered")
            self.assertEqual(result.score, .5)

    def test_low_threshold_still_requires_eight_contiguous_characters(self):
        request = GuardrailRequest("prompt", "assignment", response=TOKEN[:7], metadata={"canary": TOKEN})
        self.assertEqual(check_canary_leakage(request, {"partial_threshold": .01}).status, "passed")
        request.response = TOKEN[:8]
        self.assertEqual(check_canary_leakage(request, {"partial_threshold": .01}).status, "triggered")

    def test_partial_threshold_boundary_and_full_matching(self):
        request = GuardrailRequest("prompt", "assignment", metadata={"canary": TOKEN})
        for response, expected, score in [(TOKEN[:12], "passed", 3 / 8),
                                          (TOKEN[:16], "triggered", .5),
                                          (TOKEN[:20], "triggered", 5 / 8),
                                          (TOKEN.upper(), "triggered", 1)]:
            request.response = response
            result = check_canary_leakage(request, {"partial_threshold": .5}, FakeCanaryTokenizer())
            self.assertEqual(result.status, expected)
            self.assertEqual(result.score, score)
            if expected == "triggered" and score != 1:
                self.assertEqual(result.label, "partial_leak")
            self.assertNotIn(TOKEN, result.model_dump_json())
            self.assertNotIn(TOKEN[:4], result.model_dump_json())

    def test_partial_pieces_separated_in_output_do_not_trigger(self):
        response = "|".join(TOKEN[index:index + 4] for index in range(0, 16, 4))
        request = GuardrailRequest("prompt", "assignment", response=response, metadata={"canary": TOKEN})
        result = check_canary_leakage(request, {}, FakeCanaryTokenizer())
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.score, .125)
        self.assertFalse(result.details["full_match"])

    def test_empty_and_whitespace_only_token_pieces(self):
        class EmptyTokenizer:
            def encode(self, text, add_special_tokens=False): return [0]
            def decode(self, ids, skip_special_tokens=True): return " "
        self.assertLess(is_canary_tokens_leaked("anything", TOKEN, EmptyTokenizer()), .5)
        self.assertEqual(is_canary_tokens_leaked("anything", "", EmptyTokenizer()), 0)
        with self.assertRaises(TypeError):
            is_canary_tokens_leaked(None, TOKEN, EmptyTokenizer())

    def test_distribution_covers_start_middle_end_and_preserves_original(self):
        request = GuardrailRequest("unchanged prompt", "unchanged assignment", metadata={"canary": TOKEN})
        marker = f"[{TOKEN}]"
        for length in [0, 1, 49, 50, 51, 100, 101, 1000]:
            original = ("Source text. λ\n" * 100)[:length]
            protected = protect_system_prompt(original, request, {"spacing_chars": 50})
            self.assertTrue(protected.startswith(marker))
            self.assertTrue(protected.endswith(marker))
            self.assertEqual(protected.replace(marker, ""), original)
            self.assertTrue(all(len(chunk) <= 50 for chunk in protected.split(marker)))
        self.assertEqual(request.prompt, "unchanged prompt")
        self.assertEqual(request.assignment_context, "unchanged assignment")

    def test_prepare_reuses_per_turn_marker(self):
        request = GuardrailRequest("prompt", "assignment")
        prepare_canary(request, {})
        original = request.metadata["canary"]
        prepare_canary(request, {})
        self.assertEqual(request.metadata["canary"], original)
        self.assertEqual(len(original), 32)

    def test_invalid_distribution_and_yaml_settings(self):
        request = GuardrailRequest("prompt", "assignment", metadata={"canary": TOKEN})
        with self.assertRaises(ValueError):
            protect_system_prompt("text", GuardrailRequest("prompt", "assignment"), {})
        for spacing in (0, -1, True, "50"):
            with self.assertRaises(ValueError):
                protect_system_prompt("text", request, {"spacing_chars": spacing})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.yaml"
            for settings in ("spacing_chars: 0", "spacing_chars: true", "partial_threshold: 0", "partial_threshold: 1.5", "tokenizer_model: ''"):
                path.write_text("layers:\n  canary:\n    type: canary\n    settings: {" + settings + "}\nexecution:\n  - mode: serial\n    layers: [canary]\n")
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    load_stack_config(path, "output")


class CanaryApiTests(unittest.TestCase):
    def service(self, leak_mode, fail_once=False):
        class Resources(FakeResources):
            async def evaluate(self, layer, request):
                self.calls.append(layer.layer_id)
                if layer.guardrail_type == "canary":
                    return check_canary_leakage(request, layer.settings, FakeCanaryTokenizer())
                from guardrail_framework_demo.backend.contracts import GuardrailResult
                return GuardrailResult()
        class Model:
            def __init__(self): self.calls = 0; self.messages = []
            async def generate_chat_response(self, messages):
                from guardrail_framework_demo.backend.model_client import ModelError
                self.calls += 1; self.messages.append(messages)
                if fail_once and self.calls == 1:
                    raise ModelError("Temporary provider timeout")
                token = re.search(r"\[([0-9a-f]{32})\]", messages[0]["content"])[1]
                return token if leak_mode == "full" else token[:16] if leak_mode == "partial" else "Maintain the search interval."
        resources, model = Resources(), Model()
        layers = {"canary": LayerConfig("canary", "canary", True, "output", {"spacing_chars": 50, "partial_threshold": .5}),
                  "later": LayerConfig("later", "heuristics", True, "output", {"stage": "output"})}
        output = GuardrailStack(StackConfig(layers, (ExecutionStep("serial", ("canary", "later")),)), resources)
        return ChatService(make_stack("input", resources, ["input"]), output, model, SessionStore()), model

    def test_full_and_partial_output_leaks_block_then_override_reuses_candidate(self):
        for mode in ("full", "partial"):
            service, model = self.service(mode)
            with patch("guardrail_framework_demo.backend.guardrails.canary.secrets.token_hex", return_value=TOKEN), TestClient(create_app(service)) as client:
                session = client.post("/api/sessions", json={"assignment_context": "Binary search assignment"}).json()["session_id"]
                result = client.post("/api/chat", json={"session_id": session, "prompt": "Explain the invariant"}).json()
                self.assertEqual(result["status"], "blocked_output")
                self.assertIsNone(result["assistant_response"])
                self.assertEqual(result["output_trace"][0]["label"], "leaked" if mode == "full" else "partial_leak")
                self.assertEqual(result["output_trace"][1]["status"], "not_run")
                self.assertNotIn(TOKEN[:16], str(result))
                payload = {"session_id": session, **result["override"]}
                accepted = client.post(f"/api/turns/{result['turn_id']}/override", json=payload).json()
                duplicate = client.post(f"/api/turns/{result['turn_id']}/override", json=payload).json()
                self.assertEqual(accepted["status"], "completed")
                self.assertEqual(accepted, duplicate)
                self.assertEqual(model.calls, 1)
                self.assertEqual(accepted["output_trace"][1]["status"], "passed")
                messages = model.messages[0]
                self.assertEqual(messages[-1]["content"], "Explain the invariant")
                self.assertGreater(messages[0]["content"].count(f"[{TOKEN}]"), 2)
                plain_system = messages[0]["content"].replace(f"[{TOKEN}]", "")
                self.assertIn("Binary search assignment", plain_system)
                self.assertTrue(all(len(chunk) <= 50 for chunk in messages[0]["content"].split(f"[{TOKEN}]")))

    def test_retry_preserves_distribution_and_marker(self):
        service, model = self.service("clean", fail_once=True)
        with patch("guardrail_framework_demo.backend.guardrails.canary.secrets.token_hex", return_value=TOKEN), TestClient(create_app(service)) as client:
            session = client.post("/api/sessions", json={"assignment_context": "Search"}).json()["session_id"]
            result = client.post("/api/chat", json={"session_id": session, "prompt": "help"}).json()
            self.assertEqual(result["status"], "error")
            retry = client.post(f"/api/turns/{result['turn_id']}/retry", params={"session_id": session}).json()
            self.assertEqual(retry["status"], "completed")
            self.assertEqual(model.messages[0], model.messages[1])
