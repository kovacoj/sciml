import { serializeChatRequest, type ChatRequest } from './chat-protocol';
import { parseN8nChatResponse } from './n8n-chat-response';

export type ChatProgress = 'queued' | 'running' | 'completed';

const timeoutMs = Number(
  process.env.NEXT_PUBLIC_N8N_CHAT_TIMEOUT_MS ?? 180_000,
);
const retryIntervalMs = Number(
  process.env.NEXT_PUBLIC_N8N_CHAT_RETRY_INTERVAL_MS ?? 3_000,
);

function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(timer);
        reject(new DOMException('The request was aborted.', 'AbortError'));
      },
      { once: true },
    );
  });
}

export async function submitN8nChatRequest(
  request: ChatRequest,
  signal: AbortSignal,
  onProgress: (progress: ChatProgress) => void,
): Promise<string> {
  const endpoint = process.env.NEXT_PUBLIC_N8N_CHAT_WEBHOOK_URL;
  if (!endpoint) throw new Error('The n8n documentation chat is not configured.');

  const deadline = Date.now() + timeoutMs;
  let hasQueued = false;
  while (Date.now() < deadline) {
    const response = await fetch(endpoint, {
      method: 'POST',
      mode: 'cors',
      credentials: 'omit',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      body: serializeChatRequest(request),
      signal,
    });
    if (!response.ok) {
      const details = (await response.text()).slice(0, 500);
      throw new Error(`n8n chat request failed with HTTP ${response.status}. ${details}`);
    }

    const result = parseN8nChatResponse(await response.json(), request.requestId);
    if (!result) throw new Error('The n8n chat returned an invalid response.');
    if (result.status === 'completed') {
      onProgress('completed');
      return result.answer as string;
    }
    if (result.status === 'error') throw new Error(result.error);

    onProgress(hasQueued ? 'running' : 'queued');
    hasQueued = true;
    await wait(retryIntervalMs, signal);
  }

  throw new Error('The documentation assistant did not respond before the timeout.');
}
