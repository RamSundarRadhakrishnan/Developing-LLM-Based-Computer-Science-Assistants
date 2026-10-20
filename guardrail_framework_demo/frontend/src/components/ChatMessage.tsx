import type { DisplayTurn, TurnResult } from '../types';
import { GuardrailTrace } from './GuardrailTrace';

interface Props { turn: DisplayTurn; pending: boolean; busy: boolean; onAction: (action: 'override' | 'retry' | 'abandon', result: TurnResult) => void }
export function ChatMessage({ turn, pending, busy, onAction }: Props) {
  const result = turn.result;
  return <article className="turn">
    <div className="user-message"><div className="message-label">YOU</div><p>{turn.prompt}</p></div>
    {result?.assistant_response && <div className="assistant-message"><div className="message-label">GEMINI <span>GUARDED RESPONSE</span></div><p>{result.assistant_response}</p></div>}
    {result && result.status !== 'completed' && <div className={`trigger-card ${result.status === 'error' ? 'error' : ''}`} role="status">
      <div className="message-label">{result.status === 'abandoned' ? 'TURN ABANDONED' : result.status === 'error' ? 'REQUEST NEEDS ATTENTION' : `${result.status === 'blocked_input' ? 'INPUT' : 'OUTPUT'} GUARDRAIL TRIGGERED`}</div>
      <p>{result.message}</p>
      {pending && <>
        <p className="tiny">{result.override ? `Continue past ${result.override.layer_id.replaceAll('_', ' ')} for this turn. Remaining guardrails still run.` : 'Retry resumes this turn. Any retained response is reused.'}</p>
        <div className="trigger-actions">
          {result.override && <button className="button override-button" disabled={busy} onClick={() => onAction('override', result)}>Override & continue →</button>}
          {result.retryable && <button className="button secondary" disabled={busy} onClick={() => onAction('retry', result)}>Retry turn</button>}
          <button className="text-button" disabled={busy} onClick={() => onAction('abandon', result)}>Abandon turn</button>
        </div>
      </>}
    </div>}
    {result && <GuardrailTrace input={result.input_trace} output={result.output_trace} />}
  </article>;
}
