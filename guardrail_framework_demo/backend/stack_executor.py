"""Resumable serial/parallel execution with preserved detection evidence."""
import asyncio
import time

from .contracts import GuardrailResult
from .guardrails.canary import prepare_canary, protect_system_prompt
from .model_client import ModelError


class GuardrailStack:
    def __init__(self, config, resources):
        self.config = config
        self.resources = resources

    def prepare(self, request):
        for step in self.config.execution:
            for name in step.layers:
                layer = self.config.layers[name]
                if layer.enabled and layer.guardrail_type == "canary":
                    prepare_canary(request, layer.settings)

    def protect_system_prompt(self, system_prompt, request):
        """Apply enabled canary layers to the completed system message only."""
        for step in self.config.execution:
            for name in step.layers:
                layer = self.config.layers[name]
                if layer.enabled and layer.guardrail_type == "canary":
                    system_prompt = protect_system_prompt(system_prompt, request, layer.settings)
        return system_prompt

    def trace(self, results):
        ordered = [name for step in self.config.execution for name in step.layers]
        ordered.extend(name for name in self.config.layers if name not in ordered)
        return [results.get(name) or GuardrailResult(layer_id=name,
                guardrail_type=self.config.layers[name].guardrail_type, stage=self.config.layers[name].stage,
                status="not_run" if self.config.layers[name].enabled else "disabled", label="pending",
                reason="Not evaluated." if self.config.layers[name].enabled else "Disabled by YAML configuration.")
                for name in ordered]

    async def _evaluate(self, name, request):
        layer = self.config.layers[name]
        start = time.perf_counter()
        try:
            result = await self.resources.evaluate(layer, request)
        except Exception as error:
            # Do not echo arbitrary exception text: it could contain evaluated output.
            reason = str(error) if isinstance(error, ModelError) else "Guardrail evaluation failed. Retry or inspect backend setup."
            result = GuardrailResult(status="error", label="evaluation_error", reason=reason)
        result.layer_id = name
        result.guardrail_type = layer.guardrail_type
        result.stage = layer.stage
        result.duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return result

    def _blocking(self, result):
        return result.status == "triggered" or (result.status == "error" and self.config.fail_policy == "fail_closed")

    async def run(self, request, results):
        """Cached results are not rerun; return the first outstanding blocker."""
        for step in self.config.execution:
            active = [name for name in step.layers if self.config.layers[name].enabled]
            if step.mode == "parallel":
                missing = [name for name in active if name not in results]
                evaluated = await asyncio.gather(*(self._evaluate(name, request) for name in missing))
                results.update(zip(missing, evaluated))
                for name in active:
                    if self._blocking(results[name]):
                        return results[name]
            else:
                for name in active:
                    if name not in results:
                        results[name] = await self._evaluate(name, request)
                    if self._blocking(results[name]):
                        return results[name]
        return None
