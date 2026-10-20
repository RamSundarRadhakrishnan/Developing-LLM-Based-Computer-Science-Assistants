"""Guardrail wiring and cached model resources; detection lives in modules."""
import asyncio

from .canary import check_canary_leakage
from .context_drift import ContextDriftDetector, check_context_drift
from .encoder_classifier import EncoderClassifier, check_encoder_classification
from .heuristics import check_heuristics
from .llm_guardrail import check_llm_guardrail


class GuardrailResources:
    def __init__(self, model_client):
        self.model_client = model_client
        self.encoders = {}
        self.detectors = {}

    def initialize(self, configs):
        for config in configs:
            for layer in config.layers.values():
                if not layer.enabled:
                    continue
                if layer.guardrail_type == "encoder_classifier":
                    key = layer.settings["checkpoint_dir"]
                    if key not in self.encoders:
                        self.encoders[key] = EncoderClassifier(key)
                elif layer.guardrail_type == "drift":
                    self.detectors[layer.layer_id] = ContextDriftDetector(layer.settings)

    async def evaluate(self, layer, request):
        kind, settings = layer.guardrail_type, layer.settings
        if kind == "llm_as_judge":
            return await check_llm_guardrail(request, settings, self.model_client)
        if kind == "heuristics":
            return await asyncio.to_thread(check_heuristics, request, settings)
        if kind == "canary":
            return await asyncio.to_thread(check_canary_leakage, request, settings)
        if kind == "encoder_classifier":
            return await asyncio.to_thread(check_encoder_classification, request, settings,
                                           self.encoders[settings["checkpoint_dir"]])
        if kind == "drift":
            return await asyncio.to_thread(check_context_drift, request, settings, self.detectors[layer.layer_id])
        raise ValueError("Unregistered guardrail type.")
