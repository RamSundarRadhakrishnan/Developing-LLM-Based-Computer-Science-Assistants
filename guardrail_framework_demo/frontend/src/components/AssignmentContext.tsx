interface Props { value: string; onChange: (value: string) => void; active: boolean; disabled: boolean; onStart: () => void }

export function AssignmentContext({ value, onChange, active, disabled, onStart }: Props) {
  return <section className="assignment-panel">
    <div className="section-heading"><span className="eyebrow">THE ASSIGNMENT</span><span className="tiny">Your conversation’s anchor</span></div>
    <label className="sr-only" htmlFor="assignment">Assignment context</label>
    <textarea id="assignment" value={value} onChange={event => onChange(event.target.value)}
      placeholder="Describe the assignment, its requirements, and what you’re working on…" maxLength={16000} rows={3} disabled={disabled} />
    <div className="assignment-footer"><p>Hints, concepts, and debugging. You do the solving.</p>
      <button className={active ? 'button secondary' : 'button primary'} disabled={disabled || !value.trim()} onClick={onStart}>
        {active ? 'Start new chat' : 'Start chat'} <span aria-hidden="true">↗</span>
      </button></div>
  </section>;
}
