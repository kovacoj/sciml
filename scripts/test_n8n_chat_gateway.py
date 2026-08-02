import tempfile
import unittest
import uuid

from scripts.n8n_chat_gateway import (
    MAX_NEW_REQUESTS_PER_MINUTE,
    RateLimiter,
    ResponseStore,
    build_messages,
    validate_payload,
)


def valid_payload():
    return {
        "version": 2,
        "requestId": str(uuid.uuid4()),
        "conversationId": str(uuid.uuid4()),
        "question": "What is the covariance?",
        "currentPageUrl": "https://kovacoj.github.io/sciml/docs",
        "context": {
            "recentMessages": [
                {"role": "assistant", "content": "A prior answer."}
            ]
        },
    }


class PayloadTests(unittest.TestCase):
    def test_validates_bounded_context(self):
        payload = validate_payload(valid_payload())
        self.assertEqual(payload["question"], "What is the covariance?")

    def test_rejects_too_many_messages(self):
        payload = valid_payload()
        payload["context"]["recentMessages"] = [
            {"role": "user", "content": "x"}
        ] * 9
        with self.assertRaisesRegex(ValueError, "exceeds 8"):
            validate_payload(payload)

    def test_builds_prompt_with_question_only_once(self):
        payload = validate_payload(valid_payload())
        messages = build_messages(payload, "documentation")
        self.assertEqual(messages[-1], {"role": "user", "content": payload["question"]})
        self.assertEqual(len(messages), 3)


class ResponseStoreTests(unittest.TestCase):
    def test_returns_completed_response_for_duplicate_request(self):
        with tempfile.NamedTemporaryFile() as database:
            store = ResponseStore(database.name)
            request_id = str(uuid.uuid4())
            self.assertEqual(store.claim(request_id), (True, None))
            self.assertEqual(store.claim(request_id), (False, None))
            response = {"status": "completed", "answer": "result"}
            store.complete(request_id, response)
            self.assertEqual(store.claim(request_id), (False, response))

    def test_rate_limits_new_requests(self):
        limiter = RateLimiter()
        for _ in range(MAX_NEW_REQUESTS_PER_MINUTE):
            self.assertTrue(limiter.allow())
        self.assertFalse(limiter.allow())


if __name__ == "__main__":
    unittest.main()
