export type LayerStatus = 'passed' | 'triggered' | 'error' | 'overridden' | 'disabled' | 'not_run';
export interface GuardrailResult {
  layer_id: string; guardrail_type: string; stage: 'input' | 'output'; status: LayerStatus;
  label: string; score: number | null; reason: string; duration_ms: number; details: Record<string, unknown>;
}
export interface TurnResult {
  session_id: string; turn_id: string;
  status: 'completed' | 'blocked_input' | 'blocked_output' | 'error' | 'abandoned';
  message: string; assistant_response: string | null;
  input_trace: GuardrailResult[]; output_trace: GuardrailResult[];
  override: { stage: 'input' | 'output'; layer_id: string; token: string } | null;
  retryable: boolean;
}
export interface DisplayTurn { prompt: string; result?: TurnResult }
export interface Health { status: 'ready' | 'initializing' | 'error'; message: string; chat_model: string; judge_model: string }
