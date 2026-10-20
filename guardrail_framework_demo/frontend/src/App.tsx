import { useEffect, useRef, useState } from 'react';
import { api } from './api';
import { AssignmentContext } from './components/AssignmentContext';
import { ChatMessage } from './components/ChatMessage';
import { PromptComposer } from './components/PromptComposer';
import type { DisplayTurn, Health, TurnResult } from './types';

export default function App() {
  const [assignment, setAssignment] = useState('');
  const [session, setSession] = useState<string | null>(null);
  const [turns, setTurns] = useState<DisplayTurn[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const end = useRef<HTMLDivElement>(null);
  const operation = useRef(false);
  const latest = turns.at(-1)?.result;
  const pending = latest && ['blocked_input', 'blocked_output', 'error'].includes(latest.status) ? latest : null;

  useEffect(() => {
    let active = true;
    const poll = () => api.health().then(value => { if (active) setHealth(value); })
      .catch(() => { if (active) setHealth({ status: 'error', message: 'Backend unavailable. Start the Python server.', chat_model: '', judge_model: '' }); });
    void poll();
    const timer = window.setInterval(() => void poll(), 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [turns, busy]);

  async function run(work: () => Promise<void>) {
    if (operation.current) return;
    operation.current = true; setBusy(true); setError('');
    try { await work(); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Request failed.'); }
    finally { operation.current = false; setBusy(false); }
  }
  async function start() {
    await run(async () => {
      const created = await api.createSession(assignment);
      if (session) {
        try { await api.deleteSession(session); } catch { /* Old sessions also expire automatically. */ }
      }
      setSession(created.session_id); setTurns([]);
    });
  }
  async function send(prompt: string) {
    if (!session) return;
    await run(async () => {
      setTurns(previous => [...previous, { prompt }]);
      try {
        const result = await api.chat(session, prompt);
        setTurns(previous => [...previous.slice(0, -1), { prompt, result }]);
      } catch (cause) {
        setTurns(previous => previous.slice(0, -1));
        throw cause;
      }
    });
  }
  async function action(kind: 'override' | 'retry' | 'abandon', result: TurnResult) {
    await run(async () => {
      const updated = await api[kind](result);
      setTurns(previous => previous.map(turn => turn.result?.turn_id === result.turn_id ? { ...turn, result: updated } : turn));
    });
  }
  return <div className="app-shell">
    <header className="header"><a className="brand" href="/" aria-label="Guardrail Framework home">Guardrail Framework</a>
      <span className={`health-pill ${health?.status}`}><span className="status-dot" />{health?.status === 'ready' ? 'All systems ready' : health?.status === 'initializing' ? 'Loading guardrails' : health?.status === 'error' ? 'Backend needs attention' : 'Connecting'}</span></header>
    <main>
      <div className="model-strip"><span><b>CHAT</b> Gemini 3.5 Flash-Lite · AI Studio</span><span><b>JUDGE</b> Nemotron 3.5 Content Safety</span></div>
      {health && health.status !== 'ready' && <div className="notice" role="status">{health.message}</div>}
      <AssignmentContext value={assignment} onChange={setAssignment} active={!!session}
        disabled={busy || health?.status !== 'ready'} onStart={() => void start()} />
      <section className="chat-panel" aria-label="Guarded conversation">
        <div className="chat-heading"><span className="eyebrow">CONVERSATION</span><span className="tiny">Input → model → output</span></div>
        <div className="messages" aria-live="polite">
          {turns.length === 0 && <div className="empty-state"><span className="empty-symbol">↳</span><p>{session ? 'Ask a question about your assignment. We’ll work through it together.' : 'Add your assignment above, then start a conversation.'}</p></div>}
          {turns.map((turn, index) => <ChatMessage key={turn.result?.turn_id || index} turn={turn} pending={pending?.turn_id === turn.result?.turn_id}
            busy={busy} onAction={(kind, result) => void action(kind, result)} />)}
          {busy && <div className="loading" role="status"><span className="loading-dot" />Working through the guardrails…</div>}
          <div ref={end} />
        </div>
        {error && <div className="notice error" role="alert">{error}</div>}
        <PromptComposer disabled={!session || busy || !!pending || health?.status !== 'ready'} onSend={send} />
      </section>
    </main>
  </div>;
}
