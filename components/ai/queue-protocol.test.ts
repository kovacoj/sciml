import assert from 'node:assert/strict';
import test from 'node:test';
import {
  encodeQueueRequest,
  extractNavigationTarget,
  findQueueResponse,
  isExplicitNavigationRequest,
  parseQueueResponse,
  stripNavigationAction,
  type QueueRequest,
} from './queue-protocol.ts';

const request: QueueRequest = {
  version: 1,
  requestId: 'c9474d2a-a2d6-4cf5-b525-f1466886e87e',
  question: 'What is the equilibrium covariance?',
  currentPageUrl: 'https://kovacoj.github.io/sciml/docs/',
  conversation: [],
};

test('encodes a request marker and JSON payload', () => {
  const encoded = encodeQueueRequest(request);
  assert.match(encoded, /sciml-chat-request:c9474d2a/);
  assert.match(encoded, /```json/);
  assert.match(encoded, /equilibrium covariance/);
});

test('parses only a matching response', () => {
  const body = [
    `<!-- sciml-chat-response:${request.requestId} -->`,
    '```json',
    JSON.stringify({
      version: 1,
      requestId: request.requestId,
      status: 'completed',
      answer: 'beta^-1 A^-1',
    }),
    '```',
  ].join('\n');

  assert.equal(parseQueueResponse(body, request.requestId)?.answer, 'beta^-1 A^-1');
  assert.equal(parseQueueResponse(body, crypto.randomUUID()), null);
});

test('ignores malformed responses', () => {
  assert.equal(
    parseQueueResponse(
      `<!-- sciml-chat-response:${request.requestId} -->\n\n\`\`\`json\n{}\n\`\`\``,
      request.requestId,
    ),
    null,
  );
});

test('deduplicates responses by using the newest valid comment', () => {
  const responseBody = (answer: string) =>
    `<!-- sciml-chat-response:${request.requestId} -->\n\n\`\`\`json\n${JSON.stringify({ version: 1, requestId: request.requestId, status: 'completed', answer })}\n\`\`\``;

  const response = findQueueResponse(
    [
      { id: 1, body: responseBody('old') },
      { id: 2, body: responseBody('new') },
    ],
    request.requestId,
  );

  assert.equal(response?.answer, 'new');
});

test('extracts only documentation navigation targets', () => {
  assert.equal(
    extractNavigationTarget(
      'Opening it.\n<!-- sciml-navigate:/docs/experiments/marimo-demo -->',
      '/sciml',
    ),
    '/sciml/docs/experiments/marimo-demo',
  );
  assert.equal(
    extractNavigationTarget(
      '<!-- sciml-navigate:https://example.com/docs -->',
      '/sciml',
    ),
    null,
  );
  assert.equal(
    extractNavigationTarget('<!-- sciml-navigate:/admin -->', '/sciml'),
    null,
  );
  assert.equal(
    stripNavigationAction(
      'Opening the page.\n<!-- sciml-navigate:/docs/experiments/marimo-demo -->',
    ),
    'Opening the page.',
  );
});

test('distinguishes navigation commands from link requests', () => {
  assert.equal(isExplicitNavigationRequest('Take me to that page'), true);
  assert.equal(isExplicitNavigationRequest('Navigate to the experiment'), true);
  assert.equal(isExplicitNavigationRequest('Can you open it?'), true);
  assert.equal(isExplicitNavigationRequest('Give me a link to that page'), false);
  assert.equal(isExplicitNavigationRequest('What is the exact URL?'), false);
});
