#!/usr/bin/env python3
"""Internal idempotent chat gateway for the public n8n webhook."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_REQUEST_BYTES = 24_000
MAX_QUESTION_CHARACTERS = 5_000
MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARACTERS = 20_000
PENDING_STALE_SECONDS = 300
RESPONSE_RETENTION_SECONDS = 24 * 60 * 60
MAX_NEW_REQUESTS_PER_MINUTE = 20


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def validate_payload(payload: object) -> dict:
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise ValueError("Unsupported request format")

    request_id = payload.get("requestId")
    try:
        uuid.UUID(request_id)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Request ID is invalid") from error

    conversation_id = payload.get("conversationId")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("Conversation ID is empty")
    if len(conversation_id) > 100:
        raise ValueError("Conversation ID is too long")

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question is empty")
    if len(question) > MAX_QUESTION_CHARACTERS:
        raise ValueError("Question exceeds 5000 characters")

    current_page_url = payload.get("currentPageUrl")
    if not isinstance(current_page_url, str) or len(current_page_url) > 2000:
        raise ValueError("Current page URL is invalid")

    context = payload.get("context")
    if not isinstance(context, dict):
        raise ValueError("Request context is invalid")
    recent_messages = context.get("recentMessages")
    if not isinstance(recent_messages, list):
        raise ValueError("Recent messages must be a list")
    if len(recent_messages) > MAX_CONTEXT_MESSAGES:
        raise ValueError("Recent context exceeds 8 messages")

    total_characters = 0
    validated_messages = []
    for message in recent_messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Recent context contains an invalid message")
        total_characters += len(message["content"])
        if total_characters > MAX_CONTEXT_CHARACTERS:
            raise ValueError("Recent context exceeds 20000 characters")
        validated_messages.append(
            {"role": message["role"], "content": message["content"]}
        )

    return {
        "version": 2,
        "requestId": request_id,
        "conversationId": conversation_id,
        "question": question.strip(),
        "currentPageUrl": current_page_url,
        "context": {"recentMessages": validated_messages},
    }


def request_json(url: str, *, headers: dict[str, str], data: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"Upstream HTTP {error.code}: {details}") from error


def load_documentation() -> str:
    url = os.environ.get(
        "DOCUMENTATION_URL",
        "https://kovacoj.github.io/sciml/llms-full.txt",
    )
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def build_messages(payload: dict, documentation: str) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You are an AI assistant for a mathematical research documentation website.",
            "Use the supplied documentation as the primary reference.",
            "Distinguish general background from claims made in the documentation.",
            "Do not claim an experiment was performed unless the documentation says so.",
            "All mathematical notation must be valid KaTeX.",
            "Use $...$ for inline mathematics and $$...$$ for display mathematics.",
            "Do not use custom LaTeX macros unless they are standard KaTeX commands.",
            "Write differentials as \\mathrm{d}x, \\mathrm{d}t, and \\mathrm{d}W_t; never use \\d.",
            "When the user explicitly asks you to take, navigate, open, or go to a documented page,",
            "end the answer with exactly <!-- sciml-navigate:/docs/PATH --> using that page's",
            "relative /docs route. Never emit this marker for an external URL or unless the",
            "user explicitly requests navigation. A request for a link or URL is not a navigation",
            "request: answer those requests with a normal Markdown link and no navigation marker.",
            f"Current page: {payload['currentPageUrl']}",
            "",
            "DOCUMENTATION:",
            documentation,
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        *payload["context"]["recentMessages"],
        {"role": "user", "content": payload["question"]},
    ]


def call_siemens(payload: dict) -> str:
    api_key = required_environment("SIEMENS_LLM_API_KEY")
    base_url = os.environ.get(
        "SIEMENS_BASE_URL", "https://api.siemens.com/llm/v1"
    ).rstrip("/")
    response = request_json(
        f"{base_url}/chat/completions",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data={
            "model": os.environ.get("SIEMENS_MODEL", "qwen-3.6-27b"),
            "messages": build_messages(payload, load_documentation()),
            "max_tokens": 1000,
            "temperature": 0.2,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The Siemens LLM returned an empty response")
    return content.strip()


class ResponseStore:
    def __init__(self, path: str):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                request_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                response_json TEXT,
                updated_at INTEGER NOT NULL
            )
            """
        )
        self.connection.execute(
            "DELETE FROM responses WHERE updated_at < ?",
            (int(time.time()) - RESPONSE_RETENTION_SECONDS,),
        )
        self.connection.commit()

    def claim(self, request_id: str) -> tuple[bool, dict | None]:
        now = int(time.time())
        with self.lock:
            row = self.connection.execute(
                "SELECT state, response_json, updated_at FROM responses WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row and row[0] == "completed":
                return False, json.loads(row[1])
            if row and now - row[2] < PENDING_STALE_SECONDS:
                return False, None
            self.connection.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, 'pending', NULL, ?)",
                (request_id, now),
            )
            self.connection.commit()
        return True, None

    def complete(self, request_id: str, response: dict) -> None:
        with self.lock:
            self.connection.execute(
                "UPDATE responses SET state = 'completed', response_json = ?, updated_at = ? WHERE request_id = ?",
                (json.dumps(response), int(time.time()), request_id),
            )
            self.connection.commit()

    def get(self, request_id: str) -> dict | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT state, response_json FROM responses WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if not row:
            return None
        if row[0] == "completed":
            return json.loads(row[1])
        return {"version": 2, "requestId": request_id, "status": "pending"}


class RateLimiter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[float] = []

    def allow(self) -> bool:
        cutoff = time.monotonic() - 60
        with self.lock:
            self.requests = [timestamp for timestamp in self.requests if timestamp >= cutoff]
            if len(self.requests) >= MAX_NEW_REQUESTS_PER_MINUTE:
                return False
            self.requests.append(time.monotonic())
            return True


class ChatHandler(BaseHTTPRequestHandler):
    store: ResponseStore
    rate_limiter = RateLimiter()

    def send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        if self.path != "/chat":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is too large or empty")
            payload = validate_payload(
                json.loads(self.rfile.read(content_length).decode())
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.send_json(400, {"status": "error", "error": str(error)})
            return

        request_id = payload["requestId"]
        claimed, cached = self.store.claim(request_id)
        if cached:
            self.send_json(200, cached)
            return
        if not claimed:
            self.send_json(202, {"version": 2, "requestId": request_id, "status": "pending"})
            return
        if not self.rate_limiter.allow():
            response = {
                "version": 2,
                "requestId": request_id,
                "status": "error",
                "error": "The documentation assistant is receiving too many requests.",
            }
            self.store.complete(request_id, response)
            self.send_json(200, response)
            return

        try:
            response = {
                "version": 2,
                "requestId": request_id,
                "status": "completed",
                "answer": call_siemens(payload),
            }
        except Exception as error:
            response = {
                "version": 2,
                "requestId": request_id,
                "status": "error",
                "error": str(error)[:1000],
            }
        self.store.complete(request_id, response)
        self.send_json(200, response)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_json(200, {"status": "ok"})
            return
        if self.path.startswith("/chat/"):
            request_id = self.path.removeprefix("/chat/")
            try:
                uuid.UUID(request_id)
            except ValueError:
                self.send_json(400, {"error": "Request ID is invalid"})
                return
            response = self.store.get(request_id)
            if response is None:
                self.send_json(404, {"error": "Request not found"})
                return
            self.send_json(200, response)
            return
        self.send_json(404, {"error": "Not found"})

    def log_message(self, message_format: str, *args: object) -> None:
        print(message_format % args, file=sys.stderr)


def main() -> int:
    database_path = os.environ.get("CHAT_DATABASE", "/data/chat-responses.sqlite3")
    ChatHandler.store = ResponseStore(database_path)
    server = ThreadingHTTPServer(("0.0.0.0", 8001), ChatHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
