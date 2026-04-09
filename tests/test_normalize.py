import os
import json
import tempfile
from mempalace.normalize import normalize, _try_copilot_cli_jsonl, _try_factory_jsonl


def test_plain_text():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("Hello world\nSecond line\n")
    f.close()
    result = normalize(f.name)
    assert "Hello world" in result
    os.unlink(f.name)


def test_claude_json():
    data = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    assert "Hi" in result
    os.unlink(f.name)


def test_empty():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.close()
    result = normalize(f.name)
    assert result.strip() == ""
    os.unlink(f.name)


def _make_copilot_jsonl(events: list) -> str:
    """Helper: build a Copilot CLI events.jsonl string."""
    return "\n".join(json.dumps(e) for e in events)


def test_copilot_cli_basic():
    """Happy path: session.start + user.message + assistant.message."""
    raw = _make_copilot_jsonl([
        {"type": "session.start", "data": {"sessionId": "abc", "version": 1, "producer": "copilot-agent"}},
        {"type": "user.message", "data": {"content": "Implement AIM-P009", "transformedContent": "<injected>"}},
        {"type": "assistant.message", "data": {"content": "Phase 0 complete. All artifacts validated.", "messageId": "x"}},
    ])
    result = _try_copilot_cli_jsonl(raw)
    assert result is not None
    assert "Implement AIM-P009" in result
    assert "Phase 0 complete" in result
    # transformedContent must not leak into the transcript
    assert "<injected>" not in result


def test_copilot_cli_filters_short_assistant_messages():
    """Assistant messages under 30 chars are noise — skip them."""
    raw = _make_copilot_jsonl([
        {"type": "session.start", "data": {"sessionId": "abc", "version": 1, "producer": "copilot-agent"}},
        {"type": "user.message", "data": {"content": "Run the pipeline"}},
        {"type": "assistant.message", "data": {"content": "ok"}},
        {"type": "assistant.message", "data": {"content": "Phase 1 reviewers running in parallel. Waiting for results."}},
    ])
    result = _try_copilot_cli_jsonl(raw)
    assert result is not None
    assert "ok" not in result
    assert "Phase 1 reviewers running" in result


def test_copilot_cli_filters_pure_system_notifications():
    """Short system_notification wrappers should be dropped."""
    raw = _make_copilot_jsonl([
        {"type": "session.start", "data": {"sessionId": "abc", "version": 1, "producer": "copilot-agent"}},
        {"type": "user.message", "data": {"content": "Check status"}},
        {"type": "assistant.message", "data": {"content": "<system_notification>Agent done.</system_notification>"}},
        {"type": "assistant.message", "data": {"content": "Both reviewers approved. Phase 1 passed — moving to architecture."}},
    ])
    result = _try_copilot_cli_jsonl(raw)
    assert result is not None
    assert "<system_notification>" not in result
    assert "Phase 1 passed" in result


def test_copilot_cli_requires_session_start():
    """Without session.start fingerprint, should not match."""
    raw = _make_copilot_jsonl([
        {"type": "user.message", "data": {"content": "Hello"}},
        {"type": "assistant.message", "data": {"content": "Hi there, ready to help with anything you need."}},
    ])
    result = _try_copilot_cli_jsonl(raw)
    assert result is None


def test_copilot_cli_requires_two_messages():
    """Single message is not enough to form a transcript."""
    raw = _make_copilot_jsonl([
        {"type": "session.start", "data": {"sessionId": "abc", "version": 1, "producer": "copilot-agent"}},
        {"type": "user.message", "data": {"content": "Just one message"}},
    ])
    result = _try_copilot_cli_jsonl(raw)
    assert result is None


def test_copilot_cli_via_normalize_file():
    """End-to-end: normalize() dispatches to Copilot normalizer for .jsonl files."""
    events = [
        {"type": "session.start", "data": {"sessionId": "test-session", "version": 1, "producer": "copilot-agent"}},
        {"type": "user.message", "data": {"content": "Implement petition P-001"}},
        {"type": "tool.execution_start", "data": {"toolName": "read_file"}},
        {"type": "tool.execution_complete", "data": {"toolName": "read_file"}},
        {"type": "assistant.message", "data": {"content": "Read P-001.md successfully. Starting phase review now."}},
        {"type": "subagent.started", "data": {"agentName": "petition-translator-reviewer"}},
        {"type": "assistant.message", "data": {"content": "Petition translator reviewer approved with 0 blocking issues."}},
        {"type": "session.shutdown", "data": {"shutdownType": "routine"}},
    ]
    raw = "\n".join(json.dumps(e) for e in events)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write(raw)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "Implement petition P-001" in result
    assert "Starting phase review now" in result
    assert "approved with 0 blocking issues" in result
    # Non-message events must not appear
    assert "tool.execution_start" not in result
    assert "subagent.started" not in result


# ---------------------------------------------------------------------------
# Factory.ai / Droid normalizer tests
# ---------------------------------------------------------------------------

def _make_factory_jsonl(events: list) -> str:
    """Helper: build a Factory.ai session JSONL string."""
    return "\n".join(json.dumps(e) for e in events)


def _factory_msg(role: str, texts: list, msg_id: str = "id1", parent: str = None) -> dict:
    """Build a Factory.ai message event with content blocks."""
    content = [{"type": "text", "text": t} for t in texts]
    event = {
        "type": "message",
        "id": msg_id,
        "timestamp": "2025-11-18T12:00:00.000Z",
        "message": {"role": role, "content": content},
    }
    if parent:
        event["parentId"] = parent
    return event


def test_factory_basic():
    """Happy path: session_start + user message + assistant message."""
    raw = _make_factory_jsonl([
        {"type": "session_start", "id": "abc", "title": "test session", "owner": "user", "version": 2},
        _factory_msg("user", ["How do I implement the debt service?"]),
        _factory_msg("assistant", ["To implement the debt service, start by creating the Spring Boot application."]),
    ])
    result = _try_factory_jsonl(raw)
    assert result is not None
    assert "How do I implement the debt service?" in result
    assert "start by creating the Spring Boot application" in result


def test_factory_filters_system_reminder():
    """<system-reminder> injections in user content must be excluded."""
    raw = _make_factory_jsonl([
        {"type": "session_start", "id": "abc", "title": "t", "owner": "u", "version": 2},
        _factory_msg("user", [
            "<system-reminder>\nImportant: Never call a file editing tool in parallel.\n</system-reminder>",
            "Add a new endpoint to the creditor service.",
        ]),
        _factory_msg("assistant", ["I will add the endpoint to the creditor service controller now."]),
    ])
    result = _try_factory_jsonl(raw)
    assert result is not None
    assert "<system-reminder>" not in result
    assert "Add a new endpoint" in result


def test_factory_filters_tool_use_blocks():
    """Assistant tool_use blocks must not appear in the transcript."""
    raw = _make_factory_jsonl([
        {"type": "session_start", "id": "abc", "title": "t", "owner": "u", "version": 2},
        _factory_msg("user", ["Search for usages of DebtService."]),
        {
            "type": "message",
            "id": "asst1",
            "timestamp": "2025-11-18T12:00:01.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "Grep", "input": {"pattern": "DebtService"}},
                    {"type": "text", "text": "Found 12 usages of DebtService across 5 files."},
                ],
            },
        },
    ])
    result = _try_factory_jsonl(raw)
    assert result is not None
    assert "Found 12 usages" in result
    assert "tool_use" not in result
    assert "Grep" not in result


def test_factory_filters_thinking_blocks():
    """Extended thinking blocks must not appear in the transcript."""
    raw = _make_factory_jsonl([
        {"type": "session_start", "id": "abc", "title": "t", "owner": "u", "version": 2},
        _factory_msg("user", ["Does Droid support agents?"]),
        {
            "type": "message",
            "id": "asst2",
            "timestamp": "2025-11-18T12:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "signature": "sig==", "thinking": "The user asks about agent support."},
                    {"type": "text", "text": "Yes, Factory.ai supports droids which act as autonomous agents."},
                ],
            },
        },
    ])
    result = _try_factory_jsonl(raw)
    assert result is not None
    assert "Yes, Factory.ai supports droids" in result
    assert "thinking" not in result
    assert "The user asks about agent support" not in result


def test_factory_requires_session_start():
    """Without session_start fingerprint, must not match."""
    raw = _make_factory_jsonl([
        _factory_msg("user", ["Hello"]),
        _factory_msg("assistant", ["Hello! I am ready to help you with your coding tasks today."]),
    ])
    result = _try_factory_jsonl(raw)
    assert result is None


def test_factory_requires_two_messages():
    """Single message is not enough to form a transcript."""
    raw = _make_factory_jsonl([
        {"type": "session_start", "id": "abc", "title": "t", "owner": "u", "version": 2},
        _factory_msg("user", ["Just one message here"]),
    ])
    result = _try_factory_jsonl(raw)
    assert result is None


def test_factory_via_normalize_file():
    """End-to-end: normalize() dispatches to Factory normalizer for .jsonl files."""
    events = [
        {"type": "session_start", "id": "s1", "title": "opendebt session", "owner": "u", "version": 2},
        _factory_msg("user", [
            "<system-reminder>Do not call editing tools in parallel.</system-reminder>",
            "Implement the payment reconciliation endpoint.",
        ], msg_id="u1"),
        {
            "type": "message", "id": "a1",
            "timestamp": "2025-11-18T12:00:01.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "ReadFile", "input": {}},
                    {"type": "text", "text": "I have read the service file. Adding the reconciliation endpoint now."},
                ],
            },
            "parentId": "u1",
        },
    ]
    raw = "\n".join(json.dumps(e) for e in events)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write(raw)
    f.close()
    result = normalize(f.name)
    os.unlink(f.name)
    assert "Implement the payment reconciliation endpoint" in result
    assert "I have read the service file" in result
    assert "<system-reminder>" not in result
    assert "ReadFile" not in result
