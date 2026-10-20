"""Strict YAML loading using the reference layers/execution structure."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .settings import JUDGE_MODEL

TYPES = {"heuristics", "encoder_classifier", "llm_as_judge", "canary", "drift"}


@dataclass(frozen=True)
class LayerConfig:
    layer_id: str
    guardrail_type: str
    enabled: bool
    stage: str
    settings: dict


@dataclass(frozen=True)
class ExecutionStep:
    mode: str
    layers: tuple[str, ...]


@dataclass(frozen=True)
class StackConfig:
    layers: dict[str, LayerConfig]
    execution: tuple[ExecutionStep, ...]
    fail_policy: str = "fail_closed"


def load_stack_config(path: Path, stage: str) -> StackConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("layers"), dict):
        raise ValueError("Stack configuration must contain a layers mapping.")
    layers = {}
    for layer_id, data in raw["layers"].items():
        if not isinstance(layer_id, str) or not layer_id.strip() or not isinstance(data, dict):
            raise ValueError("Each layer requires a name and mapping.")
        kind = data.get("type")
        if kind not in TYPES or data.get("stage", stage) != stage:
            raise ValueError(f"Invalid type or stage for {layer_id}.")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool) or not isinstance(data.get("settings", {}), dict):
            raise ValueError(f"Invalid enabled/settings values for {layer_id}.")
        settings = dict(data.get("settings", {}), stage=stage)
        if "threshold" in settings and (isinstance(settings["threshold"], bool) or
                not isinstance(settings["threshold"], (int, float)) or not 0 <= settings["threshold"] <= 1):
            raise ValueError(f"Invalid threshold for {layer_id}.")
        if kind in {"canary", "drift"} and stage != "output":
            raise ValueError(f"{kind} belongs in the output stack.")
        if kind == "canary":
            length = settings.get("length", 32)
            if not isinstance(length, int) or isinstance(length, bool) or length < 2 or length % 2:
                raise ValueError("Canary length must be an even integer >= 2.")
            spacing = settings.get("spacing_chars", 50)
            if not isinstance(spacing, int) or isinstance(spacing, bool) or spacing < 1:
                raise ValueError("Canary spacing_chars must be a positive integer.")
            threshold = settings.get("partial_threshold", 0.5)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
                raise ValueError("Canary partial_threshold must be greater than 0 and at most 1.")
            model_name = settings.get("tokenizer_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("Canary tokenizer_model must be a non-empty string.")
        if kind == "heuristics":
            from .guardrails.heuristics import COMPILED
            rules = settings.get("rules", list(COMPILED[stage]))
            if not isinstance(rules, list) or any(rule not in COMPILED[stage] for rule in rules):
                raise ValueError(f"Unknown heuristic rule for {layer_id}.")
        if kind == "llm_as_judge":
            if settings.get("model", JUDGE_MODEL) != JUDGE_MODEL or settings.get("mode", "judge") != "judge":
                raise ValueError("LLM judge layers must use Nemotron in judge mode.")
            if not isinstance(settings.get("max_tokens", 512), int) or settings.get("max_tokens", 512) < 16:
                raise ValueError("Judge max_tokens must be an integer >= 16.")
            if not isinstance(settings.get("temperature", 0), (int, float)) or not 0 <= settings.get("temperature", 0) <= 2:
                raise ValueError("Judge temperature must be between 0 and 2.")
        for name in ("anchor_threshold", "adjacent_threshold"):
            if name in settings and (not isinstance(settings[name], (int, float)) or not -1 <= settings[name] <= 1):
                raise ValueError(f"{name} must be between -1 and 1.")
        for key in ("checkpoint_dir", "dataset_path"):
            if key in settings:
                resource_path = (path.parent / settings[key]).resolve()
                if enabled and not resource_path.exists():
                    raise ValueError(f"Missing {key} for {layer_id}: {resource_path}")
                settings[key] = str(resource_path)
        if enabled and kind == "encoder_classifier":
            if "checkpoint_dir" not in settings or settings.get("label") not in {"irrelevant", "direct_solution"}:
                raise ValueError("Encoder requires checkpoint_dir and a recognized risky label.")
            for filename in ("best_model.pt", "config.json", "tokenizer.json"):
                if not (Path(settings["checkpoint_dir"]) / filename).exists():
                    raise ValueError(f"Missing encoder resource: {filename}")
        if enabled and kind == "drift" and "anchor_threshold" not in settings and "dataset_path" not in settings:
            raise ValueError("Drift requires a calibration dataset or anchor_threshold.")
        layers[layer_id] = LayerConfig(layer_id, kind, enabled, stage, settings)
    execution, seen = [], set()
    steps = raw.get("execution")
    if not isinstance(steps, list) or not steps:
        raise ValueError("execution must be a non-empty list.")
    for step in steps:
        if not isinstance(step, dict) or step.get("mode") not in {"serial", "parallel"}:
            raise ValueError("Execution steps must use serial or parallel mode.")
        names = step.get("layers")
        if not isinstance(names, list) or not names:
            raise ValueError("Execution steps require a non-empty layers list.")
        for name in names:
            if not isinstance(name, str) or name not in layers or name in seen:
                raise ValueError("Execution references an unknown or repeated layer.")
            seen.add(name)
        execution.append(ExecutionStep(step["mode"], tuple(names)))
    if any(layer.enabled and name not in seen for name, layer in layers.items()):
        raise ValueError("Every enabled layer must be scheduled.")
    policy = raw.get("fail_policy", "fail_closed")
    if policy not in {"fail_closed", "fail_open"}:
        raise ValueError("Invalid fail_policy.")
    return StackConfig(layers, tuple(execution), policy)
