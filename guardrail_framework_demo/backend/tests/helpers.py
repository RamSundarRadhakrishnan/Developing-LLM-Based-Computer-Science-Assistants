from dataclasses import replace

from guardrail_framework_demo.backend.chat_service import ChatService
from guardrail_framework_demo.backend.contracts import GuardrailResult
from guardrail_framework_demo.backend.session_store import SessionStore
from guardrail_framework_demo.backend.stack_config import ExecutionStep, LayerConfig, StackConfig
from guardrail_framework_demo.backend.stack_executor import GuardrailStack


class FakeCanaryTokenizer:
    """Deterministic 4-character pieces for network-free inference tests."""
    def encode(self, text, add_special_tokens=False):
        self.pieces = [text[index:index + 4] for index in range(0, len(text), 4)]
        return list(range(len(self.pieces)))

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.pieces[index] for index in token_ids)


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.fail = False

    async def generate_chat_response(self, messages):
        from guardrail_framework_demo.backend.model_client import ModelError
        self.calls += 1
        if self.fail:
            raise ModelError("OpenRouter timed out after 90 seconds. Retry this turn.")
        return "Consider the invariant before choosing your next step."


class FakeResources:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.calls = []

    async def evaluate(self, layer, request):
        self.calls.append(layer.layer_id)
        outcome = self.outcomes.get(layer.layer_id, "passed")
        if isinstance(outcome, Exception):
            raise outcome
        return GuardrailResult(status=outcome, label="policy", reason=f"{layer.layer_id} result")


def make_stack(stage, resources, names, mode="serial", fail_policy="fail_closed"):
    layers = {name: LayerConfig(name, "heuristics", True, stage, {"stage": stage}) for name in names}
    return GuardrailStack(StackConfig(layers, (ExecutionStep(mode, tuple(names)),), fail_policy), resources)


def make_service(outcomes=None, mode="serial"):
    resources = FakeResources(outcomes)
    model = FakeModel()
    service = ChatService(make_stack("input", resources, ["i1", "i2"], mode),
                          make_stack("output", resources, ["o1", "o2"], mode), model, SessionStore())
    return service, resources, model
