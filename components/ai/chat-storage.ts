import type { ChatContextMessage } from './chat-protocol';

export const CHAT_STORAGE_KEY = 'sciml-ai-chat-v3';
export const MAX_CONTEXT_MESSAGES = 8;
export const MAX_CONTEXT_CHARACTERS = 20_000;

export interface StoredMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
  status: 'pending' | 'completed' | 'failed';
  requestId?: string;
  submittedAt?: string;
  error?: string;
}

export interface StoredChatState {
  version: 3;
  conversationId: string;
  open: boolean;
  messages: StoredMessage[];
}

function isStoredMessage(value: unknown): value is StoredMessage {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as Partial<StoredMessage>;
  return (
    typeof message.id === 'string' &&
    (message.role === 'user' || message.role === 'assistant') &&
    typeof message.content === 'string' &&
    typeof message.createdAt === 'number' &&
    (message.status === 'pending' ||
      message.status === 'completed' ||
      message.status === 'failed') &&
    (message.requestId === undefined || typeof message.requestId === 'string') &&
    (message.submittedAt === undefined ||
      typeof message.submittedAt === 'string') &&
    (message.error === undefined || typeof message.error === 'string')
  );
}

export function createChatState(): StoredChatState {
  return {
    version: 3,
    conversationId: crypto.randomUUID(),
    open: false,
    messages: [],
  };
}

export function parseChatState(value: string | null): StoredChatState | null {
  if (!value) return null;
  try {
    const state = JSON.parse(value) as Partial<StoredChatState>;
    if (
      state.version !== 3 ||
      typeof state.conversationId !== 'string' ||
      typeof state.open !== 'boolean' ||
      !Array.isArray(state.messages) ||
      !state.messages.every(isStoredMessage)
    ) {
      return null;
    }
    return {
      ...state,
      messages: state.messages.filter((message) => message.content.trim()),
    } as StoredChatState;
  } catch {
    return null;
  }
}

export function buildRecentContext(
  messages: StoredMessage[],
): ChatContextMessage[] {
  const completed = messages.filter(
    (message) => message.status === 'completed' && message.content.trim(),
  );
  const context: ChatContextMessage[] = [];
  let remaining = MAX_CONTEXT_CHARACTERS;

  for (
    let index = completed.length - 1;
    index >= 0 && context.length < MAX_CONTEXT_MESSAGES && remaining > 0;
    index -= 1
  ) {
    const message = completed[index];
    const content = message.content.slice(-remaining);
    context.unshift({ role: message.role, content });
    remaining -= content.length;
  }

  return context;
}

export function findPendingMessage(
  messages: StoredMessage[],
): StoredMessage | undefined {
  return messages.find(
    (message) =>
      message.role === 'user' &&
      message.status === 'pending' &&
      typeof message.requestId === 'string',
  );
}
