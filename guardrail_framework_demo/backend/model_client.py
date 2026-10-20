"""Separate generation and moderation calls with no cross-model fallback."""
import httpx

from .settings import CHAT_MODEL, GEMINI_URL, JUDGE_MODEL, OPENROUTER_URL, gemini_key, openrouter_key


class ModelError(RuntimeError):
    """An actionable, sanitized provider error."""


class ModelClient:
    def __init__(self, client: httpx.AsyncClient | None = None, key: str | None = None,
                 gemini_api_key: str | None = None):
        self.openrouter_key = openrouter_key() if key is None else key
        self.gemini_key = gemini_key() if gemini_api_key is None else gemini_api_key
        self.client = client or httpx.AsyncClient(timeout=90)

    async def close(self):
        await self.client.aclose()

    async def _complete(self, model: str, messages: list[dict], temperature: float, max_tokens: int, prompt: str | None = None) -> str:
        if not self.openrouter_key:
            raise ModelError("OPENROUTER_KEY is missing from the repository root .env.")
        try:
            response = await self.client.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {self.openrouter_key}", "X-Title": "Guardrail Framework Demo"},
                json={"model": model, **({"prompt": prompt} if prompt is not None else {"messages": messages}),
                      "temperature": temperature, "max_tokens": max_tokens, "stream": False},
            )
        except httpx.TimeoutException as error:
            raise ModelError("OpenRouter timed out after 90 seconds. Retry this turn.") from error
        except httpx.RequestError as error:
            raise ModelError("OpenRouter could not be reached. Check your network and retry.") from error
        if response.status_code >= 400:
            messages_by_status = {
                401: "OpenRouter authentication failed. Check OPENROUTER_KEY.",
                402: "OpenRouter rejected this request due to account limits.",
                403: "OpenRouter denied access to the requested model.",
                404: f"The requested model ({model}) is unavailable.",
                429: "OpenRouter's Nemotron judge is rate limited. Wait and retry.",
            }
            raise ModelError(messages_by_status.get(response.status_code,
                f"OpenRouter model request failed (HTTP {response.status_code}). Retry later."))
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice.get("message", {}).get("content") or choice.get("text")
            if choice.get("finish_reason") == "length":
                raise ModelError("The model response was truncated. Increase max_tokens and restart.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
            return text.strip()
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError("OpenRouter returned an empty or malformed model response.") from error

    async def generate_chat_response(self, messages: list[dict]) -> str:
        """Generate through Google AI Studio, preserving system and conversation roles."""
        if not self.gemini_key:
            raise ModelError("GEMINI_API_KEY is missing from the repository root .env.")
        contents, instructions = [], []
        for message in messages:
            role, text = message.get("role"), message.get("content")
            if not isinstance(text, str) or role not in {"system", "user", "assistant"}:
                raise ModelError("Gemini chat messages must contain supported roles and text content.")
            if role == "system":
                instructions.append({"text": text})
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]})
        if not contents:
            raise ModelError("Gemini requires at least one conversation message.")
        payload = {"contents": contents, "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048}}
        if instructions:
            payload["systemInstruction"] = {"parts": instructions}
        try:
            response = await self.client.post(GEMINI_URL, headers={"x-goog-api-key": self.gemini_key}, json=payload)
        except httpx.TimeoutException as error:
            raise ModelError("Google AI Studio timed out after 90 seconds. Retry this turn.") from error
        except httpx.RequestError as error:
            raise ModelError("Google AI Studio could not be reached. Check your network and retry.") from error
        if response.status_code >= 400:
            messages_by_status = {
                400: "Google AI Studio rejected the request. Check GEMINI_API_KEY and the model configuration.",
                401: "Google AI Studio authentication failed. Check GEMINI_API_KEY.",
                403: "Google AI Studio denied access. Check GEMINI_API_KEY and project permissions.",
                404: f"Google AI Studio model {CHAT_MODEL} is unavailable for this project.",
                429: "Google AI Studio quota or rate limit reached. Wait and retry, or check your project's quota.",
            }
            raise ModelError(messages_by_status.get(response.status_code,
                f"Google AI Studio request failed (HTTP {response.status_code}). Retry later."))
        try:
            body = response.json()
            if body.get("promptFeedback", {}).get("blockReason"):
                raise ModelError("Google AI Studio blocked this prompt under its provider safety policy.")
            candidate = body["candidates"][0]
            finish = candidate.get("finishReason")
            if finish == "MAX_TOKENS":
                raise ModelError("Gemini's response was truncated at its output token limit. Shorten the request and retry.")
            if finish != "STOP":
                raise ModelError("Google AI Studio did not return a complete response; it may have been blocked by provider safety checks.")
            parts = candidate["content"]["parts"]
            text = "".join(part["text"] for part in parts if not part.get("thought") and isinstance(part.get("text"), str))
            if not text.strip():
                raise ValueError("Empty response")
            return text.strip()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
            raise ModelError("Google AI Studio returned an empty or malformed model response.") from error

    async def evaluate_guardrail_policy(self, messages: list[dict], settings: dict) -> str:
        if settings.get("model", JUDGE_MODEL) != JUDGE_MODEL:
            raise ModelError("Judge layers must use the configured Nemotron content-safety model.")
        # NVIDIA's native template ignores the incoming system instruction in standard
        # moderation mode. Render our custom-policy prompt explicitly instead.
        from .guardrails.llm_guardrail import render_nemotron_prompt
        return await self._complete(JUDGE_MODEL, messages, float(settings.get("temperature", 0)),
                                    int(settings.get("max_tokens", 512)), prompt=render_nemotron_prompt(messages))
