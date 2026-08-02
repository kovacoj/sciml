export const MAX_REQUEST_CHARACTERS = 24_000;
export const MAX_QUESTION_CHARACTERS = 5_000;
export const MAX_CONTEXT_MESSAGES = 8;
export const MAX_CONTEXT_CHARACTERS = 20_000;
const NAVIGATION_PATTERN =
  /<!--\s*sciml-navigate:(\/[A-Za-z0-9/_-]*docs[A-Za-z0-9/_-]*\/?)[ ]*-->/;

export interface ChatContextMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  version: 2;
  requestId: string;
  conversationId: string;
  question: string;
  currentPageUrl: string;
  context: {
    recentMessages: ChatContextMessage[];
  };
}

export function serializeChatRequest(request: ChatRequest): string {
  const contextCharacters = request.context.recentMessages.reduce(
    (total, message) => total + message.content.length,
    0,
  );
  if (
    request.question.length > MAX_QUESTION_CHARACTERS ||
    request.context.recentMessages.length > MAX_CONTEXT_MESSAGES ||
    contextCharacters > MAX_CONTEXT_CHARACTERS
  ) {
    throw new Error('The documentation chat request is too large.');
  }
  const encoded = JSON.stringify(request);
  if (encoded.length > MAX_REQUEST_CHARACTERS) {
    throw new Error('The documentation chat request is too large.');
  }
  return encoded;
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

export function isExplicitNavigationRequest(question: string): boolean {
  return (
    /\b(?:take|bring)\s+me\b/i.test(question) ||
    /\bnavigate\b.*\b(?:to|there)\b/i.test(question) ||
    /\bgo\s+(?:to|there)\b/i.test(question) ||
    /\bopen\s+(?:it|the|that|this)\b/i.test(question) ||
    /\bsend\s+me\s+there\b/i.test(question)
  );
}

export function extractDocumentationLinkTarget(
  answer: string,
  basePath: string,
): string | null {
  const match = answer.match(
    /\/(?:[A-Za-z0-9_-]+\/)*docs(?:\/[A-Za-z0-9/_-]+)?\/?/,
  );
  if (!match) return null;

  const candidate = match[0].replace(/\/$/, '');
  const docsIndex = candidate.indexOf('/docs');
  if (docsIndex < 0) return null;

  const normalizedBasePath = basePath.replace(/\/$/, '');
  return `${normalizedBasePath}${candidate.slice(docsIndex)}`;
}
