export interface N8nChatResponse {
  version: 2;
  requestId: string;
  status: 'pending' | 'completed' | 'error';
  answer?: string;
  error?: string;
}

export function parseN8nChatResponse(
  value: unknown,
  requestId: string,
): N8nChatResponse | null {
  if (typeof value !== 'object' || value === null) return null;
  const response = value as Partial<N8nChatResponse>;
  if (
    response.version !== 2 ||
    response.requestId !== requestId ||
    !['pending', 'completed', 'error'].includes(response.status ?? '')
  ) {
    return null;
  }
  if (
    response.status === 'completed' &&
    (typeof response.answer !== 'string' || !response.answer.trim())
  ) {
    return null;
  }
  if (
    response.status === 'error' &&
    (typeof response.error !== 'string' || !response.error.trim())
  ) {
    return null;
  }
  return response as N8nChatResponse;
}
