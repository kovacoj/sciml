import assert from 'node:assert/strict';
import test from 'node:test';
import { pollForQueueResponse } from './queue-protocol.ts';

const requestId = 'c9474d2a-a2d6-4cf5-b525-f1466886e87e';
const responseBody = `<!-- sciml-chat-response:${requestId} -->\n\n\`\`\`json\n${JSON.stringify({ version: 1, requestId, status: 'completed', answer: 'response' })}\n\`\`\``;

test('polls until a matching response is available', async () => {
  const progress: string[] = [];
  let pollCount = 0;
  const answer = await pollForQueueResponse({
    requestId,
    timeoutMs: 100,
    pollIntervalMs: 1,
    signal: new AbortController().signal,
    onProgress: (value) => progress.push(value),
    fetchComments: async () => {
      pollCount += 1;
      return pollCount === 1 ? [] : [{ id: 2, body: responseBody }];
    },
    waitForNext: async () => undefined,
    now: () => 0,
  });

  assert.equal(answer, 'response');
  assert.deepEqual(progress, ['running', 'completed']);
});

test('reports a deterministic timeout', async () => {
  let now = 0;
  await assert.rejects(
    pollForQueueResponse({
      requestId,
      timeoutMs: 5,
      pollIntervalMs: 1,
      signal: new AbortController().signal,
      onProgress: () => undefined,
      fetchComments: async () => [],
      waitForNext: async () => {
        now += 5;
      },
      now: () => now,
    }),
    /queue timeout/,
  );
});
