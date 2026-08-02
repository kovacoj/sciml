import assert from 'node:assert/strict';
import test from 'node:test';
import { parseN8nChatResponse } from './n8n-chat-response.ts';

const requestId = 'c9474d2a-a2d6-4cf5-b525-f1466886e87e';

test('parses matching completed and pending n8n responses', () => {
  assert.equal(
    parseN8nChatResponse(
      { version: 2, requestId, status: 'completed', answer: 'response' },
      requestId,
    )?.answer,
    'response',
  );
  assert.equal(
    parseN8nChatResponse({ version: 2, requestId, status: 'pending' }, requestId)
      ?.status,
    'pending',
  );
});

test('rejects mismatched and malformed n8n responses', () => {
  assert.equal(
    parseN8nChatResponse(
      { version: 2, requestId, status: 'completed', answer: '' },
      requestId,
    ),
    null,
  );
  assert.equal(
    parseN8nChatResponse(
      { version: 2, requestId: crypto.randomUUID(), status: 'pending' },
      requestId,
    ),
    null,
  );
});
