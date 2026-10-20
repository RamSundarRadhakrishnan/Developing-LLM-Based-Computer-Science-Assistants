"""Opt-in real checkpoint inference and drift calibration verification."""
from .contracts import GuardrailRequest
from .guardrails.registry import GuardrailResources
from .input_stack import assemble_input_stack
from .model_client import ModelClient
from .output_stack import assemble_output_stack


def main():
    resources = GuardrailResources(None)
    input_stack = assemble_input_stack(resources=resources)
    output_stack = assemble_output_stack(resources=resources)
    print("Loading trained encoders and calibrating drift...", flush=True)
    resources.initialize([input_stack.config, output_stack.config])
    request = GuardrailRequest("How do I maintain the search interval?", "Implement binary search on a sorted array.",
                               response="Track a lower and upper bound, then reason about which half can contain the target.")
    from .guardrails.encoder_classifier import check_encoder_classification
    from .guardrails.context_drift import check_context_drift
    for config in (input_stack.config, output_stack.config):
        for layer in config.layers.values():
            if layer.guardrail_type == "encoder_classifier":
                result = check_encoder_classification(request, layer.settings, resources.encoders[layer.settings["checkpoint_dir"]])
                print(layer.layer_id, result.status, round(result.score, 4), result.details, flush=True)
            elif layer.guardrail_type == "drift":
                result = check_context_drift(request, layer.settings, resources.detectors[layer.layer_id])
                print(layer.layer_id, result.status, round(result.score, 4), "threshold", result.details["anchor_threshold"], flush=True)
    print("Real resource loading and inference succeeded.", flush=True)


if __name__ == "__main__":
    main()
