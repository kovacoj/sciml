import json
import unittest

from scripts.docs_chat import bound_legacy_context, parse_request, validate_context


class ParseRequestTests(unittest.TestCase):
    def test_parses_matching_request(self):
        request_id = "c9474d2a-a2d6-4cf5-b525-f1466886e87e"
        payload = {
            "version": 1,
            "requestId": request_id,
            "question": "What is the covariance?",
        }
        comment = (
            f"<!-- sciml-chat-request:{request_id} -->\n\n"
            f"```json\n{json.dumps(payload)}\n```"
        )

        parsed_id, parsed_payload = parse_request(comment)
        self.assertEqual(parsed_id, request_id)
        self.assertEqual(parsed_payload["question"], payload["question"])

    def test_rejects_mismatched_payload(self):
        request_id = "c9474d2a-a2d6-4cf5-b525-f1466886e87e"
        payload = {
            "version": 1,
            "requestId": "ab3bebdb-1221-4398-af92-3e58cffd7f23",
            "question": "What is the covariance?",
        }
        comment = (
            f"<!-- sciml-chat-request:{request_id} -->\n\n"
            f"```json\n{json.dumps(payload)}\n```"
        )

        with self.assertRaisesRegex(ValueError, "do not match"):
            parse_request(comment)

    def test_parses_bounded_version_two_context(self):
        request_id = "c9474d2a-a2d6-4cf5-b525-f1466886e87e"
        payload = {
            "version": 2,
            "requestId": request_id,
            "conversationId": "browser-conversation",
            "question": "What is the covariance?",
            "currentPageUrl": "https://example.test/docs",
            "context": {
                "recentMessages": [
                    {"role": "assistant", "content": "Prior answer"}
                ]
            },
        }
        comment = (
            f"<!-- sciml-chat-request:{request_id} -->\n\n"
            f"```json\n{json.dumps(payload)}\n```"
        )

        _, parsed_payload = parse_request(comment)
        self.assertEqual(parsed_payload["version"], 2)

    def test_rejects_too_many_context_messages(self):
        messages = [{"role": "user", "content": "x"}] * 9
        with self.assertRaisesRegex(ValueError, "exceeds 8 messages"):
            validate_context(messages)

    def test_rejects_oversized_context(self):
        messages = [{"role": "user", "content": "x" * 20001}]
        with self.assertRaisesRegex(ValueError, "exceeds 20000 characters"):
            validate_context(messages)

    def test_bounds_legacy_context_from_the_newest_messages(self):
        messages = [
            {"role": "user", "content": f"message {index}"}
            for index in range(10)
        ]
        bounded = bound_legacy_context(messages)
        self.assertEqual(len(bounded), 8)
        self.assertEqual(bounded[0]["content"], "message 2")
        self.assertEqual(bounded[-1]["content"], "message 9")


if __name__ == "__main__":
    unittest.main()
