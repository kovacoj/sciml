import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildRecentContext,
  findPendingMessage,
  MAX_CONTEXT_CHARACTERS,
  parseChatState,
  type StoredMessage,
} from './chat-storage.ts';

const message = (
  index: number,
  overrides: Partial<StoredMessage> = {},
): StoredMessage => ({
  id: String(index),
  role: index % 2 ? 'user' : 'assistant',
  content: `message ${index}`,
  createdAt: index,
  status: 'completed',
  ...overrides,
});

test('restores rich pending-request metadata', () => {
  const pending = message(1, {
    status: 'pending',
    requestId: 'c9474d2a-a2d6-4cf5-b525-f1466886e87e',
    submittedAt: '2026-08-02T12:00:00.000Z',
  });
  const restored = parseChatState(
    JSON.stringify({
      version: 3,
      conversationId: 'conversation',
      open: true,
      messages: [pending],
    }),
  );

  assert.deepEqual(restored?.messages, [pending]);
  assert.equal(restored?.open, true);
  assert.deepEqual(findPendingMessage(restored?.messages ?? []), pending);
});

test('rejects malformed stored state', () => {
  assert.equal(parseChatState('{"version":3,"messages":[]}'), null);
  assert.equal(parseChatState('not json'), null);
});

test('keeps only the eight newest completed context messages', () => {
  const context = buildRecentContext([
    ...Array.from({ length: 10 }, (_, index) => message(index)),
    message(10, { status: 'failed' }),
  ]);

  assert.equal(context.length, 8);
  assert.equal(context[0].content, 'message 2');
  assert.equal(context[7].content, 'message 9');
});

test('caps aggregate context at 20000 characters from the newest messages', () => {
  const context = buildRecentContext([
    message(1, { content: 'a'.repeat(15_000) }),
    message(2, { content: 'b'.repeat(15_000) }),
  ]);

  assert.equal(
    context.reduce((total, item) => total + item.content.length, 0),
    MAX_CONTEXT_CHARACTERS,
  );
  assert.equal(context[1].content, 'b'.repeat(15_000));
});
