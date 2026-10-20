# Guardrail Framework Demo

A local assignment-focused chat website with React/TypeScript, FastAPI, and five modular guardrail types. The backend uses the repository's existing trained checkpoints and calibration dataset without modifying them.

```text
Prompt → input heuristics → irrelevancy encoder → Nemotron input judge
       → Gemini 3.5 Flash-Lite chat generation (Google AI Studio)
       → output heuristics → solution-proximity encoder → Nemotron output judge
       → canary detector → context drift → displayed response
```

## Models and secrets

| Purpose | Provider | Model |
| --- | --- | --- |
| Chat generation | Google AI Studio | `gemini-3.5-flash-lite` |
| Input/output judging | OpenRouter | `nvidia/nemotron-3.5-content-safety:free` |

The backend reads `GEMINI_API_KEY` for chat and `OPENROUTER_KEY` for judging from the **existing repository-root `.env`** using an absolute path. No key belongs in frontend code. Gemini uses Google's native `generateContent` endpoint with the key in the `x-goog-api-key` header, never in its URL. Chat history uses `user`/`model` roles and the tutoring policy/canary uses `systemInstruction`. Provider reasoning is excluded from returned text. There is no model fallback. Google AI Studio quota and OpenRouter's free Nemotron limits are independent.

Nemotron is called using OpenRouter's raw `prompt` input with an explicit native custom-policy prompt. Its standard moderation template does not reliably apply system-message tutoring instructions. The custom policy names assignment irrelevance, complete submissions, prompt injection, and internal-instruction disclosure. The parser reads only final native `User Safety`/`Response Safety` labels outside reasoning blocks. Output judging uses the response verdict independently. Current and historical private canaries are redacted before judging; raw reasoning is never sent to the browser.

## Setup on Windows

Use Python 3.11+ and Node 20.19+ or 22.12+. Run these commands from the repository root:

```powershell
python -m venv guardrail_framework_demo/.venv
guardrail_framework_demo/.venv/Scripts/Activate.ps1
python -m pip install -r guardrail_framework_demo/backend/requirements.txt
cd guardrail_framework_demo/frontend
npm install
npm run build
cd ../..
python -m uvicorn guardrail_framework_demo.backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Initial startup loads two trained encoders and an embedding model, then calibrates drift if there is no matching cache. The website reports initialization status and disables chat until resources are ready.

For CUDA, install a PyTorch build appropriate for your GPU before installing the remaining requirements. CPU inference is supported. Do not use multiple Uvicorn workers: each worker would have separate sessions and model copies.

For frontend development, start the Python server as above and, in a second terminal:

```powershell
cd guardrail_framework_demo/frontend
npm run dev
```

Open the Vite URL printed in the terminal (normally http://127.0.0.1:5173). Its `/api` proxy targets port 8000.

## Existing resources required

Relative to the repository root:

- `irrelevancy_modernbert/`: `best_model.pt`, custom `config.json`, and tokenizer files.
- `solution_proximality_modernbert/`: the equivalent ModernBERT resources.
- `encoder_classifier/drift_synthetic_dataset_raahul.csv`: labeled conversations for calibration.

The classifiers reconstruct the exact custom head, load the complete checkpoint strictly, and use the notebooks' tagged input format and CLS pooling. The input model uses 512 tokens; the output model uses 1024. Truncation is reported in the trace. Backbone architecture configurations and MiniLM weights may download from Hugging Face on first use; subsequent runs can use its local cache. No encoder is replaced by an untrained model.

Drift uses normalized attention-mask-weighted mean pooling, assistant-turn similarity to the assignment anchor, and a dataset-calibrated threshold. Its cache lives in `backend/.runtime/` and includes a dataset/model/settings fingerprint. Adjacent-turn similarity is diagnostic unless configured otherwise. The synthetic-data threshold can flag concise, abstract but legitimate guidance: inspect the scores and use an override or set a validated `anchor_threshold` in YAML. These checks are a demo of detection, not a guarantee of perfect classifications.

## Modular backend

`backend/guardrails/` contains one file for each type:

| Module | Public invocation |
| --- | --- |
| `heuristics.py` | `check_heuristics(request, settings)` |
| `encoder_classifier.py` | `check_encoder_classification(request, settings, classifier)` |
| `llm_guardrail.py` | `check_llm_guardrail(request, settings, model_client)` |
| `canary.py` | `prepare_canary(request, settings)`, `protect_system_prompt(system_prompt, request, settings)`, and `check_canary_leakage(request, settings)` |
| `context_drift.py` | `check_context_drift(request, settings, detector)` |

`input_stack.py` and `output_stack.py` expose `assemble_input_stack()` and `assemble_output_stack()`. They read the corresponding YAML files. `stack_config.py` validates the architecture; `stack_executor.py` handles evaluation and continuation. `chat_service.py` expresses the input → model → output sequence. `main.py` initializes the application and exposes routes.

## Change the YAML architecture

Each stack has `layers`, `execution`, and `fail_policy`, matching the reference framework's structure. Paths resolve relative to the YAML file. Restart the backend after edits; pending turns keep their existing architecture.

```yaml
fail_policy: fail_closed
layers:
  prompt_heuristics:
    type: heuristics
    stage: input
    enabled: true
    settings:
      threshold: 0.75
execution:
  - mode: serial
    layers: [prompt_heuristics]
```

Default YAML enables all three input and five output layers. Reorder layer names in `execution` to change order. Set `enabled: false` to disable a layer. Every enabled layer must appear exactly once.

To evaluate a group concurrently, split execution into steps:

```yaml
execution:
  - mode: serial
    layers: [prompt_heuristics]
  - mode: parallel
    layers: [irrelevancy_encoder, llm_input_judge]
```

Parallel groups finish all their checks before continuing, and their traces retain YAML order. Each triggered member requires its own explicit override. Detection triggers always block. `fail_open` allows continuation only after runtime evaluation errors, keeping those errors visible; the default `fail_closed` pauses for retry.

Parameters include heuristic `rules` and `threshold`, encoder `checkpoint_dir`, `label`, and `threshold`, judge `policy`, `temperature`, and `max_tokens`, canary `length`, `spacing_chars`, and `partial_threshold`, and drift `model_name`, `dataset_path`, `anchor_threshold`, and `adjacent_threshold`. Judge layers require the exact Nemotron model and `mode: judge`. Canary preparation happens before generation even though leakage detection appears later in the output stack.

### Distributed canaries and partial detection

The existing 32-character random canary is repeated as `[canary]` at the beginning and end of the entire system prompt, and after every 50 original characters. The completed tutoring instructions and assignment context are protected before being sent as Gemini's `systemInstruction`. Only the model-facing system prompt receives insertions: user prompts, conversation history, and the original assignment context used by classifiers stay unchanged. Retries reuse the same canary and reconstruct the same marked prompt.

Full-token detection is case-insensitive and always triggers. Partial detection measures the longest contiguous, case-insensitive substring shared by the canary and response, divided by the canary length. At the default **0.5** threshold, a 32-character canary requires a matching run of at least 16 characters. A minimum of eight contiguous characters (or the entire token if shorter) applies even when the threshold is lowered. Scattered letters, digits, and separated tokenizer pieces do not accumulate evidence. This score is a matched-length fraction, not a probability.

Canary detection requires no tokenizer or model download. The old `tokenizer_model` setting is accepted for compatibility but has no effect. Traces expose the score, threshold, full-match flag, and verdict without exposing the secret. Existing judge redaction of full current/historical canaries remains in place. Partial detection catches verbatim excerpts; fragments shorter than the required length, separated fragments, and paraphrases can escape detection.

## Chat and overrides

Enter the assignment context and start a chat. Assignment edits take effect when starting a new chat, which creates a fresh session.

- Serial execution stops at the first detection trigger; later checks show `not_run`.
- **Override & continue** accepts only that triggered layer for this turn. Its original label, reason, score, and original trigger status remain in the trace. Remaining layers still run.
- An output override reuses the privately retained response. It never generates a new response merely to bypass a trigger.
- **Retry turn** retries failed work while retaining successful checks and an existing candidate response.
- **Abandon turn** clears the pending turn without adding it to model history.
- Accepted turns with any explicit override stay in conversation history but are excluded from later historical drift minima.

Candidates are buffered on the server until checks finish. Blocked responses do not appear in API replies. Sessions are process-local and expire after 30 minutes of inactivity; restarting the backend clears them. Limits are 100 sessions and 100 turns per session. Session IDs are random local-demo capabilities; this application is intended for localhost, with no account system or public deployment configuration.

## API

| Method and route | Purpose |
| --- | --- |
| `GET /api/health` | Resource readiness and model names |
| `GET /api/guardrails` | Sanitized active architecture |
| `POST /api/sessions` | `{ "assignment_context": "..." }` → session ID |
| `DELETE /api/sessions/{session_id}` | Clear a session |
| `POST /api/chat` | `{ "session_id": "...", "prompt": "..." }` |
| `POST /api/turns/{turn_id}/override` | Session ID plus returned `stage`, `layer_id`, and `token` |
| `POST /api/turns/{turn_id}/retry?session_id=...` | Resume an errored turn |
| `POST /api/turns/{turn_id}/abandon?session_id=...` | Abandon a pending turn |

Chat statuses are `completed`, `blocked_input`, `blocked_output`, `error`, and `abandoned`. Only `completed` includes assistant text. Results include both traces and, for a trigger, a single-use override descriptor. Repeating the same consumed override request returns its cached result without repeating inference; mismatched tokens and session ownership are rejected.

Interactive API documentation: http://127.0.0.1:8000/docs.

## Verification

From the repository root:

```powershell
python -m unittest discover -s guardrail_framework_demo/backend/tests -t . -v
# Optional: pytest supports the same suite.
python -m pytest guardrail_framework_demo/backend/tests
python -m guardrail_framework_demo.backend.verify_resources
python -m guardrail_framework_demo.backend.verify_canary
python -m guardrail_framework_demo.backend.verify_gemini
python -m guardrail_framework_demo.backend.verify_openrouter --judge-only
python -m guardrail_framework_demo.backend.verify_openrouter
# With the server already running:
python -m guardrail_framework_demo.backend.verify_api
```

From `frontend`, run `npm run typecheck` and `npm run build`. The unit/integration suite uses mocked inference and needs no API requests. Resource verification loads real checkpoints and calibrates drift. `verify_gemini` tests Google AI Studio chat directly. OpenRouter verification checks five live Nemotron tutoring-policy examples and optionally also tests Gemini generation. Smoke scripts never print secrets or model reasoning.

## Troubleshooting

- **Initializing:** wait for model loading/calibration. First-run downloads require internet access.
- **Initialization failed:** inspect backend logs and confirm checkpoint/tokenizer files exist. Restart after fixing resources.
- **401/403:** for chat, check `GEMINI_API_KEY` and Google AI Studio project/model access; for judging, check `OPENROUTER_KEY`.
- **429:** the error names the provider. Check Google AI Studio quota for chat or OpenRouter's free Nemotron limits for judging. Wait and use Retry turn; models are never changed automatically.
- **Judge error:** missing, conflicting, invalid, or truncated final verdicts fail closed. Retry; increase judge `max_tokens` if needed.
- **Frontend 404:** run `npm run build` before starting the single-server application, or use Vite for development.
- **Session expired:** start a new chat; pending turns cannot survive expiration or a restart.

## Implementation verification

The implementation was checked with 42 passing backend tests, a passing TypeScript/production build, strict loading and CUDA inference of both real trained checkpoints, and drift calibration over 663 labeled examples. All five live Nemotron tutoring-policy examples passed after switching to the explicit custom-policy prompt. Direct Google AI Studio generation using `gemini-3.5-flash-lite` also passed. Canary tests additionally cover contiguous fragment scoring and benign-output false-positive regressions, full/partial output blocking, override continuation without regeneration, distribution boundaries, and identical prompts on retry. Tokenizer-independent canary verification is available through `verify_canary`. The local API verifies input blocking, override continuation, and abandoned-turn isolation.

Earlier live validation found that a benign guidance response scored 0.529 with the original tokenizer-piece heuristic. That scorer counted common letters and digits scattered throughout ordinary output. The detector now requires a contiguous secret fragment instead; it no longer preserves the original scorer's false-positive behavior.

Chat generation now uses Google AI Studio rather than the previous OpenRouter chat endpoint. Provider-specific request mapping, credential separation, response parsing, safety blocks, truncation, and errors are covered by mocked tests. Browser screenshot and narrow-viewport checks could not run because no browser surface was available to the session; the responsive CSS and production build are present.
