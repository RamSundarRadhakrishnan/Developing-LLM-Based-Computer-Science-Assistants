"""Assemble the output architecture from output_stack.yaml."""
from pathlib import Path

from .settings import BACKEND_DIR
from .stack_config import load_stack_config
from .stack_executor import GuardrailStack


def assemble_output_stack(config_path=None, resources=None) -> GuardrailStack:
    config = load_stack_config(Path(config_path) if config_path else BACKEND_DIR / "output_stack.yaml", "output")
    return GuardrailStack(config, resources)
