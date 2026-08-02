import assert from 'node:assert/strict';
import test from 'node:test';
import {
  serializeChatRequest,
  extractDocumentationLinkTarget,
  extractNavigationTarget,
  isExplicitNavigationRequest,
  stripNavigationAction,
  type ChatRequest,
} from './chat-protocol.ts';

const request: ChatRequest = {
  version: 2,
  requestId: 'c9474d2a-a2d6-4cf5-b525-f1466886e87e',
  conversationId: '9d153721-b308-4f23-9f04-f659e3c343f1',
  question: 'What is the equilibrium covariance?',
  currentPageUrl: 'https://kovacoj.github.io/sciml/docs/',
  context: { recentMessages: [] },
};

test('serializes a bounded request payload', () => {
  const encoded = serializeChatRequest(request);
  assert.match(encoded, /equilibrium covariance/);
  assert.doesNotMatch(encoded, /"conversation"/);
});

test('rejects an oversized chat payload', () => {
  assert.throws(
    () =>
      serializeChatRequest({
        ...request,
        context: {
          recentMessages: [{ role: 'user', content: 'x'.repeat(24_000) }],
        },
      }),
    /too large/,
  );
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
  assert.equal(isExplicitNavigationRequest('Navigate please to lingebra'), true);
  assert.equal(isExplicitNavigationRequest('You navigate me there'), true);
  assert.equal(isExplicitNavigationRequest('Can you open it?'), true);
  assert.equal(isExplicitNavigationRequest('Give me a link to that page'), false);
  assert.equal(isExplicitNavigationRequest('What is the exact URL?'), false);
});

test('extracts a same-site documentation path from a normal answer', () => {
  assert.equal(
    extractDocumentationLinkTarget(
      'See https://kovacoj.github.io/sciml/docs/experiments/thermodynamic-linear-algebra for details.',
      '/sciml',
    ),
    '/sciml/docs/experiments/thermodynamic-linear-algebra',
  );
  assert.equal(
    extractDocumentationLinkTarget('There is no relevant page.', '/sciml'),
    null,
  );
});
