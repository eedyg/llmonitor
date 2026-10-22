"""Non-streamed clients still get one body - rebuilt from the engine's stream."""
import json

from monitor.assemble import ChatAssembler


def _events(*objs):
    asm = ChatAssembler()
    for obj in objs:
        asm.feed_event(obj)
    return asm


def _chunk(**over):
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1792500000,
            "model": "qwen3.8-27b", "choices": []}
    base.update(over)
    return base


def test_thinking_and_content_are_reassembled_in_order():
    asm = _events(
        _chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}}]),
        _chunk(choices=[{"index": 0, "delta": {"reasoning_content": "先想一"}}]),
        _chunk(choices=[{"index": 0, "delta": {"reasoning_content": "下。"}}]),
        _chunk(choices=[{"index": 0, "delta": {"content": "答案"}}]),
        _chunk(choices=[{"index": 0, "delta": {"content": "是 42"}, "finish_reason": "stop"}]),
        _chunk(choices=[], usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}),
    )
    body = asm.result()
    assert body["object"] == "chat.completion"
    assert body["id"] == "chatcmpl-1"
    assert body["model"] == "qwen3.8-27b"
    choice = body["choices"][0]
    assert choice["message"]["content"] == "答案是 42"
    assert choice["message"]["reasoning_content"] == "先想一下。"
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 7


def test_tool_call_arguments_are_concatenated_by_index():
    asm = _events(
        _chunk(choices=[{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "type": "function",
             "function": {"name": "get_weather", "arguments": ""}}]}}]),
        _chunk(choices=[{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"city"'}}]}}]),
        _chunk(choices=[{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ': "成都"}'}}]}}]),
        _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    )
    message = asm.result()["choices"][0]["message"]
    assert message["content"] == ""
    call = message["tool_calls"][0]
    assert call["id"] == "call_a"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "成都"}
    assert asm.result()["choices"][0]["finish_reason"] == "tool_calls"


def test_two_choices_stay_separate():
    asm = _events(
        _chunk(choices=[{"index": 0, "delta": {"content": "A"}}]),
        _chunk(choices=[{"index": 1, "delta": {"content": "B"}}]),
    )
    body = asm.result()
    assert [c["message"]["content"] for c in body["choices"]] == ["A", "B"]


def test_usage_only_chunk_does_not_create_a_choice():
    asm = _events(
        _chunk(choices=[{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]),
        _chunk(choices=[], usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
    )
    body = asm.result()
    assert len(body["choices"]) == 1
    assert body["usage"]["prompt_tokens"] == 3


def test_error_event_is_kept_for_the_caller():
    asm = ChatAssembler()
    asm.feed_event({"error": {"message": "engine exploded", "type": "boom"}})
    assert asm.error and "exploded" in asm.error["message"]
