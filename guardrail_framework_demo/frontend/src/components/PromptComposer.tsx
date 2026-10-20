import { useState } from 'react';

export function PromptComposer({ disabled, onSend }: { disabled: boolean; onSend: (prompt: string) => Promise<void> }) {
  const [prompt, setPrompt] = useState('');
  async function submit() {
    if (!prompt.trim() || disabled) return;
    const text = prompt.trim();
    setPrompt('');
    await onSend(text);
  }
  return <form className="composer" onSubmit={event => { event.preventDefault(); void submit(); }}>
    <label htmlFor="prompt" className="sr-only">Your prompt</label>
    <textarea id="prompt" placeholder="Ask for a hint, explore a concept, or debug an idea…" value={prompt}
      onChange={event => setPrompt(event.target.value)} maxLength={16000} rows={2} disabled={disabled}
      onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submit(); } }} />
    <div className="composer-footer">
      <button className="button primary" type="submit" disabled={disabled || !prompt.trim()}>Send <span aria-hidden="true">↑</span></button></div>
  </form>;
}
