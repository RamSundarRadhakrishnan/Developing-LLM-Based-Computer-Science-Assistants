"""Process-local sessions and private pending candidates with inactivity expiry."""
import asyncio
import time
import uuid
from dataclasses import dataclass, field

from fastapi import HTTPException

from .contracts import AcceptedTurn, ChatOutput, GuardrailRequest, GuardrailResult, OverrideDescriptor
from .settings import SESSION_TTL_SECONDS


@dataclass
class PendingTurn:
    turn_id: str
    request: GuardrailRequest
    input_results: dict[str, GuardrailResult] = field(default_factory=dict)
    output_results: dict[str, GuardrailResult] = field(default_factory=dict)
    override: OverrideDescriptor | None = None
    result: ChatOutput | None = None
    used_tokens: dict[str, tuple[str, str, ChatOutput]] = field(default_factory=dict)
    overridden: bool = False


@dataclass
class Session:
    session_id: str
    assignment_context: str
    history: list[AcceptedTurn] = field(default_factory=list)
    turns: dict[str, PendingTurn] = field(default_factory=dict)
    pending_id: str | None = None
    touched: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self, ttl=SESSION_TTL_SECONDS):
        self.sessions: dict[str, Session] = {}
        self.ttl = ttl

    def purge(self):
        now = time.monotonic()
        for key, session in list(self.sessions.items()):
            if not session.lock.locked() and now - session.touched > self.ttl:
                del self.sessions[key]

    def create(self, context: str) -> Session:
        self.purge()
        if len(self.sessions) >= 100:
            raise HTTPException(503, "Local session capacity reached. Clear an old session.")
        session = Session(str(uuid.uuid4()), context)
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        self.purge()
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(404, "Session expired or was cleared. Start a new chat.")
        session.touched = time.monotonic()
        return session

    def delete(self, session_id: str):
        session = self.get(session_id)
        if session.lock.locked():
            raise HTTPException(409, "This session is processing a turn.")
        del self.sessions[session_id]
