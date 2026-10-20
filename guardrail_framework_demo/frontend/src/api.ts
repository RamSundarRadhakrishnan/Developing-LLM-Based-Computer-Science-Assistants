import type { Health, TurnResult } from './types';

async function request<T>(path: string, body?: unknown, method = 'POST'): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed. Check your input.');
  }
  return data as T;
}

export const api = {
  health: () => request<Health>('/health', undefined, 'GET'),
  createSession: (assignment_context: string) => request<{ session_id: string }>('/sessions', { assignment_context }),
  deleteSession: (id: string) => request(`/sessions/${encodeURIComponent(id)}`, undefined, 'DELETE'),
  chat: (session_id: string, prompt: string) => request<TurnResult>('/chat', { session_id, prompt }),
  override: (result: TurnResult) => request<TurnResult>(`/turns/${result.turn_id}/override`, { session_id: result.session_id, ...result.override }),
  retry: (result: TurnResult) => request<TurnResult>(`/turns/${result.turn_id}/retry?session_id=${encodeURIComponent(result.session_id)}`),
  abandon: (result: TurnResult) => request<TurnResult>(`/turns/${result.turn_id}/abandon?session_id=${encodeURIComponent(result.session_id)}`),
};
