import json
import unittest

from scripts.docs_chat import parse_request


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


if __name__ == "__main__":
    unittest.main()
