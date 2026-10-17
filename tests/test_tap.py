import json
import time

from monitor.tap import ANTHROPIC, CHAT, RESPONSES, Record, StreamTap, request_meta


def sse(obj) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def feed(tap: StreamTap, chunks: list[bytes], split: int = 7) -> None:
    """Feed chunks with awkward boundaries to exercise the incremental parser."""
    blob = b"".join(chunks)
    for i in range(0, len(blob), split):
        tap.feed(blob[i : i + split])


def test_chat_thinking_and_usage():
    record = Record(protocol=CHAT, stream=True)
    tap = StreamTap(record)
    feed(
        tap,
        [
            sse({"choices": [{"delta": {"reasoning": "We need"}}]}),
            sse({"choices": [{"delta": {"reasoning": " to add."}}]}),
            sse({"choices": [{"delta": {"content": "4"}}]}),
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            sse({"choices": [], "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 8},
            }}),
            b"data: [DONE]\n\n",
        ],
    )
    assert record.thinking_chunks == 2
    assert record.content_chunks == 1
    assert record.thinking_text == "We need to add."
    assert record.content_text == "4"
    assert record.finish_reason == "stop"
    assert record.prompt_tokens == 12
    assert record.prompt_cached == 4
    assert record.thinking_tokens == 8
    assert record.content_tokens == 1
    assert record.phase == "decode"
    assert record.ttft_ms is not None


def test_chat_legacy_reasoning_content():
    record = Record(protocol=CHAT, stream=True)
    tap = StreamTap(record)
    feed(tap, [sse({"choices": [{"delta": {"reasoning_content": "hmm"}}]})], split=3)
    assert record.thinking_text == "hmm"


def test_anthropic_stream():
    record = Record(protocol=ANTHROPIC, stream=True)
    tap = StreamTap(record)
    feed(
        tap,
        [
            b"event: message_start\n" + sse({"type": "message_start", "message": {"usage": {"input_tokens": 20}}}),
            b"event: content_block_delta\n" + sse({"type": "content_block_delta",
                                                  "delta": {"type": "thinking_delta", "thinking": "think"}}),
            b"event: content_block_delta\n" + sse({"type": "content_block_delta",
                                                  "delta": {"type": "text_delta", "text": "answer"}}),
            b"event: message_delta\n" + sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                             "usage": {"output_tokens": 7, "cache_read_input_tokens": 5}}),
        ],
    )
    assert record.thinking_text == "think"
    assert record.content_text == "answer"
    assert record.prompt_tokens == 20
    assert record.prompt_cached == 5
    assert record.finish_reason == "end_turn"
    assert record.thinking_chunks == 1 and record.content_chunks == 1


def test_responses_stream():
    record = Record(protocol=RESPONSES, stream=True)
    tap = StreamTap(record)
    feed(
        tap,
        [
            b"event: response.reasoning_summary_text.delta\n"
            + sse({"type": "response.reasoning_summary_text.delta", "delta": "why"}),
            b"event: response.output_text.delta\n"
            + sse({"type": "response.output_text.delta", "delta": "hello"}),
            b"event: response.completed\n"
            + sse({"type": "response.completed", "response": {"status": "completed", "usage": {
                "input_tokens": 6, "output_tokens": 2,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 1}}}}),
        ],
    )
    assert record.thinking_text == "why"
    assert record.content_text == "hello"
    assert record.prompt_tokens == 6
    assert record.prompt_cached == 2
    assert record.thinking_tokens == 1
    assert record.content_tokens == 1


def test_non_stream_body():
    record = Record(protocol=CHAT, stream=False)
    tap = StreamTap(record)
    body = json.dumps({
        "choices": [{"message": {"content": "hi", "reasoning_content": "r"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }).encode()
    tap.feed_body(body)
    assert record.content_text == "hi"
    assert record.thinking_text == "r"
    assert record.prompt_tokens == 3


def test_request_meta_chat():
    record = Record(protocol=CHAT)
    request_meta(record, {
        "model": "qwen3.8-27b",
        "stream": True,
        "max_tokens": 128,
        "temperature": 0.7,
        "reasoning_effort": "none",
        "messages": [
            {"role": "user", "content": "hello 世界"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}]},
        ],
    })
    assert record.model == "qwen3.8-27b"
    assert record.stream is True
    assert record.max_tokens == 128
    assert record.prompt_messages == 2
    assert record.has_media is True
    assert record.thinking_requested is False
    assert record.prompt_tokens_est > 0


def test_live_decode_tps():
    record = Record(protocol=CHAT, stream=True)
    tap = StreamTap(record)
    record.first_token_at = time.time() - 2.0
    record.content_chunks = 41
    data = record.live()
    assert data["content_tokens"] == 41
    assert 15 < data["live_decode_tps"] < 25
