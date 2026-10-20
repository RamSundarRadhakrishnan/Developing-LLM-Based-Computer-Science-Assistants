"""Shared, explicit guardrail and API contracts."""
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Stage = Literal["input", "output"]
ResultStatus = Literal["passed", "triggered", "error", "overridden", "disabled", "not_run"]


@dataclass
class AcceptedTurn:
    prompt: str
    response: str
    overridden: bool = False
    private_marker: str | None = None


@dataclass
class GuardrailRequest:
    prompt: str
    assignment_context: str
    conversation: list[AcceptedTurn] = field(default_factory=list)
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GuardrailResult(BaseModel):
    layer_id: str = ""
    guardrail_type: str = ""
    stage: Stage = "input"
    status: ResultStatus = "passed"
    label: str = "clear"
    score: float | None = None
    reason: str = ""
    duration_ms: float = 0
    details: dict[str, Any] = Field(default_factory=dict)


class SessionCreate(BaseModel):
    assignment_context: str = Field(min_length=1, max_length=16000)

    @field_validator("assignment_context")
    @classmethod
    def nonempty_context(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Assignment context is required")
        return value.strip()


class ChatInput(BaseModel):
    session_id: str
    prompt: str = Field(min_length=1, max_length=16000)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A prompt is required")
        return value.strip()


class OverrideInput(BaseModel):
    session_id: str
    stage: Stage
    layer_id: str
    token: str


class OverrideDescriptor(BaseModel):
    stage: Stage
    layer_id: str
    token: str


class ChatOutput(BaseModel):
    session_id: str
    turn_id: str
    status: Literal["completed", "blocked_input", "blocked_output", "error", "abandoned"]
    message: str
    assistant_response: str | None = None
    input_trace: list[GuardrailResult] = Field(default_factory=list)
    output_trace: list[GuardrailResult] = Field(default_factory=list)
    override: OverrideDescriptor | None = None
    retryable: bool = False
