export const REQUEST_MARKER_PREFIX = '<!-- sciml-chat-request:';
export const RESPONSE_MARKER_PREFIX = '<!-- sciml-chat-response:';
const NAVIGATION_PATTERN =
  /<!--\s*sciml-navigate:(\/[A-Za-z0-9/_-]*docs[A-Za-z0-9/_-]*\/?)[ ]*-->/;

export interface QueueConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface QueueRequest {
  version: 1;
  requestId: string;
  question: string;
  currentPageUrl: string;
  conversation: QueueConversationMessage[];
}

export interface QueueResponse {
  version: 1;
  requestId: string;
  status: 'completed' | 'error';
  answer?: string;
  error?: string;
}

export interface IssueComment {
  id: number;
  body?: string | null;
}

interface PollOptions {
  requestId: string;
  timeoutMs: number;
  pollIntervalMs: number;
  signal: AbortSignal;
  onProgress: (progress: 'running' | 'completed') => void;
  fetchComments: () => Promise<IssueComment[]>;
  waitForNext: (milliseconds: number, signal: AbortSignal) => Promise<void>;
  now?: () => number;
}

function extractJson(body: string): unknown {
  const match = body.match(/```json\s*([\s\S]*?)\s*```/);
  if (!match) return null;

  try {
    return JSON.parse(match[1]);
  } catch {
    return null;
  }
}

export function encodeQueueRequest(request: QueueRequest): string {
  return [
    `${REQUEST_MARKER_PREFIX}${request.requestId} -->`,
    '',
    '```json',
    JSON.stringify(request),
    '```',
  ].join('\n');
}

export function parseQueueResponse(
  body: string,
  requestId: string,
): QueueResponse | null {
  if (!body.includes(`${RESPONSE_MARKER_PREFIX}${requestId} -->`)) {
    return null;
  }

  const value = extractJson(body);
  if (typeof value !== 'object' || value === null) return null;

  const candidate = value as Partial<QueueResponse>;
  if (
    candidate.version !== 1 ||
    candidate.requestId !== requestId ||
    (candidate.status !== 'completed' && candidate.status !== 'error')
  ) {
    return null;
  }

  if (
    candidate.status === 'completed' &&
    (typeof candidate.answer !== 'string' || candidate.answer.trim() === '')
  ) {
    return null;
  }

  if (
    candidate.status === 'error' &&
    (typeof candidate.error !== 'string' || candidate.error.trim() === '')
  ) {
    return null;
  }

  return candidate as QueueResponse;
}

export function findQueueResponse(
  comments: IssueComment[],
  requestId: string,
): QueueResponse | null {
  for (let index = comments.length - 1; index >= 0; index -= 1) {
    const body = comments[index].body;
    if (typeof body !== 'string') continue;

    const response = parseQueueResponse(body, requestId);
    if (response) return response;
  }

  return null;
}

export function extractNavigationTarget(
  answer: string,
  basePath: string,
): string | null {
  const match = answer.match(NAVIGATION_PATTERN);
  if (!match) return null;

  const candidate = match[1].replace(/\/$/, '');
  const normalizedBasePath = basePath.replace(/\/$/, '');
  if (
    normalizedBasePath &&
    (candidate === `${normalizedBasePath}/docs` ||
      candidate.startsWith(`${normalizedBasePath}/docs/`))
  ) {
    return candidate;
  }

  if (candidate === '/docs' || candidate.startsWith('/docs/')) {
    return `${normalizedBasePath}${candidate}`;
  }

  return null;
}

export function stripNavigationAction(answer: string): string {
  return answer.replace(NAVIGATION_PATTERN, '').trim();
}

export async function pollForQueueResponse({
  requestId,
  timeoutMs,
  pollIntervalMs,
  signal,
  onProgress,
  fetchComments,
  waitForNext,
  now = Date.now,
}: PollOptions): Promise<string> {
  const deadline = now() + timeoutMs;
  let hasPolled = false;

  while (now() < deadline) {
    await waitForNext(pollIntervalMs, signal);
    if (!hasPolled) {
      onProgress('running');
      hasPolled = true;
    }

    const queueResponse = findQueueResponse(
      await fetchComments(),
      requestId,
    );
    if (!queueResponse) continue;
    if (queueResponse.status === 'error') {
      throw new Error(queueResponse.error);
    }

    onProgress('completed');
    return queueResponse.answer as string;
  }

  throw new Error(
    'The documentation assistant did not respond before the queue timeout.',
  );
}
