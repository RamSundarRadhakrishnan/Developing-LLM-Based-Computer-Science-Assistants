import type { GuardrailResult } from '../types';

const labels: Record<string, string> = {
  prompt_heuristics: 'Prompt heuristics', irrelevancy_encoder: 'Relevance encoder', llm_input_judge: 'Nemotron input judge',
  response_heuristics: 'Response heuristics', solution_proximity_encoder: 'Solution proximity', llm_output_judge: 'Nemotron output judge',
  canary_detector: 'Canary detector', context_drift: 'Context drift',
};
export function GuardrailTrace({ input, output }: { input: GuardrailResult[]; output: GuardrailResult[] }) {
  const results = [...input, ...output];
  const checked = results.filter(item => ['passed', 'triggered', 'overridden', 'error'].includes(item.status)).length;
  return <details className="trace"><summary>Guardrail trace <span>{checked} / {results.filter(item => item.status !== 'disabled').length} evaluated</span></summary>
    {[['INPUT', input], ['OUTPUT', output]].map(([stage, rows]) => <div className="trace-stage" key={stage as string}>
      <div className="eyebrow">{stage as string}</div>
      {(rows as GuardrailResult[]).map(result => <div className="trace-row" key={result.layer_id}>
        <div><span className={`status-dot ${result.status}`} /><strong>{labels[result.layer_id] || result.layer_id}</strong><span className={`status-label ${result.status}`}>{result.status.replace('_', ' ')}</span></div>
        <p>{result.reason}</p>
        {result.score !== null && <span className="tiny">{result.guardrail_type === 'drift' ? 'Anchor similarity' : 'Score'} {result.score.toFixed(3)}{typeof result.details.threshold === 'number' ? ` · threshold ${result.details.threshold}` : typeof result.details.anchor_threshold === 'number' ? ` · threshold ${result.details.anchor_threshold.toFixed(3)}` : ''}</span>}
        {result.duration_ms > 0 && <span className="tiny"> · {Math.round(result.duration_ms)} ms</span>}
        {result.details.truncated === true && <p className="tiny">Input was truncated to the encoder’s token limit.</p>}
      </div>)}
    </div>)}
  </details>;
}
