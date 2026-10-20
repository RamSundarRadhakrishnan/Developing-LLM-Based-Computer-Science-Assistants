"""FastAPI entry point: initialize stacks and expose the guarded chat website."""
import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .chat_service import ChatService
from .contracts import ChatInput, ChatOutput, OverrideInput, SessionCreate
from .guardrails.registry import GuardrailResources
from .input_stack import assemble_input_stack
from .model_client import ModelClient
from .output_stack import assemble_output_stack
from .session_store import SessionStore
from .settings import CHAT_MODEL, DEMO_DIR, JUDGE_MODEL, gemini_key, openrouter_key

logger = logging.getLogger(__name__)


def create_app(service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        client = None
        task = None
        if service is not None:
            app.state.service = service
            app.state.ready = True
        else:
            client = ModelClient()
            resources = GuardrailResources(client)
            try:
                input_stack = assemble_input_stack(resources=resources)
                output_stack = assemble_output_stack(resources=resources)
                app.state.service = ChatService(input_stack, output_stack, client, SessionStore())
            except Exception as error:
                app.state.initialization_error = f"Stack configuration error: {error}"
            else:
                async def initialize():
                    try:
                        if not gemini_key():
                            raise ValueError("GEMINI_API_KEY is missing from the root .env.")
                        if not openrouter_key():
                            raise ValueError("OPENROUTER_KEY is missing from the root .env.")
                        await asyncio.to_thread(resources.initialize, [input_stack.config, output_stack.config])
                        app.state.ready = True
                    except Exception as error:
                        logger.error("Guardrail initialization failed: %s", error)
                        app.state.initialization_error = "Model initialization failed. Check backend logs, checkpoint files, and model-download access."
                task = asyncio.create_task(initialize())
        try:
            yield
        finally:
            if task:
                await task
            if client:
                await client.close()

    app = FastAPI(title="Guardrail Framework Demo", lifespan=lifespan)
    app.state.ready = service is not None
    app.state.initialization_error = None
    if service:
        app.state.service = service

    def ready_service():
        if not app.state.ready:
            raise HTTPException(503, app.state.initialization_error or "Guardrail models are loading and calibrating. Please wait.")
        return app.state.service

    @app.get("/api/health")
    async def health():
        return {"status": "ready" if app.state.ready else "error" if app.state.initialization_error else "initializing",
                "message": app.state.initialization_error or ("All guardrail resources loaded." if app.state.ready else "Loading encoders and calibrating drift detector."),
                "chat_model": CHAT_MODEL, "judge_model": JUDGE_MODEL,
                "chat_provider": "Google AI Studio", "judge_provider": "OpenRouter"}

    @app.get("/api/guardrails")
    async def guardrails():
        current = getattr(app.state, "service", None)
        if current is None:
            raise HTTPException(503, app.state.initialization_error or "Initializing stacks.")
        def describe(stack):
            return {"fail_policy": stack.config.fail_policy,
                    "execution": [asdict(step) for step in stack.config.execution],
                    "layers": [{"layer_id": layer.layer_id, "type": layer.guardrail_type,
                                "stage": layer.stage, "enabled": layer.enabled,
                                "settings": {key: value for key, value in layer.settings.items()
                                             if key in {"threshold", "model", "mode", "temperature", "max_tokens", "anchor_threshold", "adjacent_threshold", "spacing_chars", "partial_threshold", "tokenizer_model"}}}
                               for layer in stack.config.layers.values()]}
        return {"input": describe(current.input_stack), "output": describe(current.output_stack)}

    @app.post("/api/sessions")
    async def create_session(data: SessionCreate):
        session = ready_service().store.create(data.assignment_context)
        return {"session_id": session.session_id}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        ready_service().store.delete(session_id)
        return {"cleared": True}

    @app.post("/api/chat", response_model=ChatOutput)
    async def chat(data: ChatInput):
        return await ready_service().start_turn(data)

    @app.post("/api/turns/{turn_id}/override", response_model=ChatOutput)
    async def override(turn_id: str, data: OverrideInput):
        return await ready_service().override_turn(turn_id, data)

    @app.post("/api/turns/{turn_id}/retry", response_model=ChatOutput)
    async def retry(turn_id: str, session_id: str = Query(...)):
        return await ready_service().retry_turn(session_id, turn_id)

    @app.post("/api/turns/{turn_id}/abandon", response_model=ChatOutput)
    async def abandon(turn_id: str, session_id: str = Query(...)):
        return await ready_service().abandon_turn(session_id, turn_id)

    dist = DEMO_DIR / "frontend" / "dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    async def index():
        if not (dist / "index.html").exists():
            raise HTTPException(404, "Frontend has not been built. Run npm run build in frontend, or use the Vite dev server.")
        return FileResponse(dist / "index.html")

    return app


app = create_app()
