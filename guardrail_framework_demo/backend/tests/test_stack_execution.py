import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from guardrail_framework_demo.backend.contracts import GuardrailRequest
from guardrail_framework_demo.backend.stack_config import load_stack_config
from .helpers import FakeResources, make_stack


class ConfigTests(unittest.TestCase):
    def test_invalid_configurations(self):
        examples = [
            "layers: {}\nexecution: []",
            "layers: {a: {type: unknown}}\nexecution: [{mode: serial, layers: [a]}]",
            "layers: {a: {type: heuristics, stage: output}}\nexecution: [{mode: serial, layers: [a]}]",
            "layers: {a: {type: heuristics}}\nexecution: [{mode: serial, layers: [a, a]}]",
            "layers: {a: {type: heuristics}}\nexecution: [{mode: parallel, layers: [missing]}]",
            "layers: {a: {type: heuristics, settings: {threshold: 2}}}\nexecution: [{mode: serial, layers: [a]}]",
            "layers: {a: {type: llm_as_judge, settings: {model: wrong}}}\nexecution: [{mode: serial, layers: [a]}]",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.yaml"
            for text in examples:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_stack_config(path, "input")

    def test_real_yaml_paths_and_enabled_layers(self):
        from guardrail_framework_demo.backend.input_stack import assemble_input_stack
        from guardrail_framework_demo.backend.output_stack import assemble_output_stack
        self.assertEqual(len(assemble_input_stack().config.layers), 3)
        output = assemble_output_stack().config
        self.assertEqual(len(output.layers), 5)
        self.assertTrue(Path(output.layers["context_drift"].settings["dataset_path"]).is_file())


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_serial_short_circuit_resume(self):
        resources = FakeResources({"a": "triggered"})
        stack = make_stack("input", resources, ["a", "b"])
        results = {}
        request = GuardrailRequest("prompt", "assignment")
        self.assertEqual((await stack.run(request, results)).layer_id, "a")
        self.assertEqual(resources.calls, ["a"])
        self.assertEqual(stack.trace(results)[1].status, "not_run")
        results["a"].status = "overridden"
        self.assertIsNone(await stack.run(request, results))
        self.assertEqual(resources.calls, ["a", "b"])

    async def test_parallel_waits_and_preserves_order(self):
        resources = FakeResources({"a": "triggered", "b": "triggered"})
        stack = make_stack("output", resources, ["a", "b"], mode="parallel")
        results = {}
        request = GuardrailRequest("prompt", "assignment")
        self.assertEqual((await stack.run(request, results)).layer_id, "a")
        self.assertEqual(resources.calls, ["a", "b"])
        results["a"].status = "overridden"
        self.assertEqual((await stack.run(request, results)).layer_id, "b")
        self.assertEqual(resources.calls, ["a", "b"])

    async def test_fail_open_only_applies_to_errors(self):
        resources = FakeResources({"a": ValueError("private evaluated content"), "b": "triggered"})
        stack = make_stack("input", resources, ["a", "b"], fail_policy="fail_open")
        results = {}
        self.assertEqual((await stack.run(GuardrailRequest("prompt", "assignment"), results)).layer_id, "b")
        self.assertNotIn("private", results["a"].reason)
        closed = make_stack("input", resources, ["a", "b"])
        self.assertEqual((await closed.run(GuardrailRequest("prompt", "assignment"), {})).status, "error")

    async def test_disabled_layers_are_not_run(self):
        resources = FakeResources()
        stack = make_stack("input", resources, ["a", "b"])
        stack.config.layers["a"] = replace(stack.config.layers["a"], enabled=False)
        results = {}
        await stack.run(GuardrailRequest("prompt", "assignment"), results)
        self.assertEqual(resources.calls, ["b"])
        self.assertEqual(stack.trace(results)[0].status, "disabled")

    async def test_canary_preparation_does_not_evaluate_detection(self):
        from guardrail_framework_demo.backend.output_stack import assemble_output_stack
        stack = assemble_output_stack()
        request = GuardrailRequest("prompt", "assignment")
        stack.prepare(request)
        self.assertIn("canary", request.metadata)
        self.assertEqual(next(row for row in stack.trace({}) if row.layer_id == "canary_detector").status, "not_run")
