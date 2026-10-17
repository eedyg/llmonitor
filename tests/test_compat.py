from monitor.compat import CompletionStream, chat_to_completion, to_chat_request


def test_to_chat_request():
    chat, prompt = to_chat_request({
        "model": "qwen3.8-27b",
        "prompt": ["a", "b"],
        "max_tokens": 32,
        "temperature": 0.2,
        "stream": True,
    })
    assert prompt == "a\nb"
    assert chat["messages"] == [{"role": "user", "content": "a\nb"}]
    assert chat["reasoning_effort"] == "none"
    assert chat["max_tokens"] == 32
    assert chat["temperature"] == 0.2
    assert chat["stream"] is True


def test_chat_to_completion_echo():
    out = chat_to_completion({"id": "x", "model": "m", "choices": [{"message": {"content": " world"}, "finish_reason": "stop"}]}, "hello", True)
    assert out["object"] == "text_completion"
    assert out["choices"][0]["text"] == "hello world"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_completion_stream_translation():
    state = CompletionStream("", False, "m")
    chunk = (b'data: {"id":"c1","choices":[{"delta":{"reasoning":"skip"},"finish_reason":null}]}\n\n'
             b'data: {"id":"c1","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
             b'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
    out = state.feed(chunk)
    text = out.decode()
    assert '"text": "ok"' in text
    assert '"finish_reason": "stop"' in text
    assert "reasoning" not in text
    assert state.end() == b"data: [DONE]\n\n"
