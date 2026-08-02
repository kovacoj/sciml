#!/usr/bin/env python3
"""Process one documentation chat request from a GitHub issue comment."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

REQUEST_PATTERN = re.compile(
    r"<!-- sciml-chat-request:([0-9a-fA-F-]{36}) -->\s*"
    r"```json\s*([\s\S]*?)\s*```"
)
RESPONSE_MARKER = "<!-- sciml-chat-response:{request_id} -->"
MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARACTERS = 20_000
MAX_REQUEST_CHARACTERS = 24_000


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def request_json(url: str, *, headers: dict[str, str], data: dict | None = None) -> dict:
    encoded = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=encoded, headers=headers)
    if encoded is not None:
        request.method = "POST"

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"HTTP {error.code} from {url}: {details}") from error


def parse_request(comment: str) -> tuple[str, dict]:
    match = REQUEST_PATTERN.fullmatch(comment.strip())
    if not match:
        raise ValueError("Comment does not match the documentation chat request format")

    request_id, payload_text = match.groups()
    payload = json.loads(payload_text)
    if payload.get("version") not in {1, 2} or payload.get("requestId") != request_id:
        raise ValueError("Request marker and payload do not match")
    if not isinstance(payload.get("question"), str) or not payload["question"].strip():
        raise ValueError("Question is empty")
    if len(payload["question"]) > 5000:
        raise ValueError("Question exceeds 5000 characters")
    if payload["version"] == 2:
        if len(comment) > MAX_REQUEST_CHARACTERS:
            raise ValueError("Request exceeds 24000 characters")
        if not isinstance(payload.get("conversationId"), str) or not payload[
            "conversationId"
        ].strip():
            raise ValueError("Conversation ID is empty")
        context = payload.get("context")
        if not isinstance(context, dict):
            raise ValueError("Request context is invalid")
        validate_context(context.get("recentMessages"))

    return request_id, payload


def validate_context(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("Recent messages must be a list")
    if len(messages) > MAX_CONTEXT_MESSAGES:
        raise ValueError("Recent context exceeds 8 messages")

    validated = []
    total_characters = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Recent context contains an invalid message")
        total_characters += len(message["content"])
        if total_characters > MAX_CONTEXT_CHARACTERS:
            raise ValueError("Recent context exceeds 20000 characters")
        validated.append(
            {"role": message["role"], "content": message["content"]}
        )
    return validated


def bound_legacy_context(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    bounded = []
    remaining = MAX_CONTEXT_CHARACTERS
    for message in reversed(messages):
        if len(bounded) >= MAX_CONTEXT_MESSAGES or remaining <= 0:
            break
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            continue
        content = message["content"][-remaining:]
        bounded.append({"role": message["role"], "content": content})
        remaining -= len(content)
    return list(reversed(bounded))


def load_documentation(url: str) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def call_siemens(payload: dict, documentation: str) -> str:
    api_key = required_environment("SIEMENS_LLM_API_KEY")
    base_url = os.environ.get("SIEMENS_BASE_URL", "https://api.siemens.com/llm/v1").rstrip("/")
    model = os.environ.get("SIEMENS_MODEL", "qwen-3.6-27b")
    system_prompt = "\n".join(
        [
            "You are an AI assistant for a mathematical research documentation website.",
            "Use the supplied documentation as the primary reference.",
            "Distinguish general background from claims made in the documentation.",
            "Do not claim an experiment was performed unless the documentation says so.",
            "When the user explicitly asks you to take, navigate, open, or go to a documented page,",
            "end the answer with exactly <!-- sciml-navigate:/docs/PATH --> using that page's",
            "relative /docs route. Never emit this marker for an external URL or unless the",
            "user explicitly requests navigation. A request for a link or URL is not a navigation",
            "request: answer those requests with a normal Markdown link and no navigation marker.",
            "Treat natural variants such as 'navigate please to', 'navigate me there', and minor",
            "misspellings as explicit navigation when the intended documented page is clear.",
            f"Current page: {payload.get('currentPageUrl', '')}",
            "",
            "DOCUMENTATION:",
            documentation,
        ]
    )
    if payload.get("version") == 2:
        conversation = validate_context(payload["context"]["recentMessages"])
    else:
        conversation = bound_legacy_context(payload.get("conversation", []))
    messages = [{"role": "system", "content": system_prompt}]
    for message in conversation:
        if (
            isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ):
            messages.append(
                {"role": message["role"], "content": message["content"][:10000]}
            )

    if not messages or messages[-1].get("content") != payload["question"]:
        messages.append({"role": "user", "content": payload["question"]})

    response = request_json(
        f"{base_url}/chat/completions",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data={
            "model": model,
            "messages": messages,
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


def post_response(request_id: str, response_payload: dict) -> None:
    repository = required_environment("GITHUB_REPOSITORY")
    issue = required_environment("QUEUE_ISSUE")
    token = required_environment("GH_TOKEN")
    body = "\n\n".join(
        [
            RESPONSE_MARKER.format(request_id=request_id),
            f"```json\n{json.dumps(response_payload, ensure_ascii=False)}\n```",
        ]
    )
    request_json(
        f"https://api.github.com/repos/{repository}/issues/{issue}/comments",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        data={"body": body},
    )


def main() -> int:
    comment = required_environment("REQUEST_COMMENT")
    request_id, payload = parse_request(comment)
    try:
        documentation = load_documentation(required_environment("DOCUMENTATION_URL"))
        answer = call_siemens(payload, documentation)
        response_payload = {
            "version": 1,
            "requestId": request_id,
            "status": "completed",
            "answer": answer,
        }
    except Exception as error:
        response_payload = {
            "version": 1,
            "requestId": request_id,
            "status": "error",
            "error": str(error)[:1000],
        }
    post_response(request_id, response_payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
