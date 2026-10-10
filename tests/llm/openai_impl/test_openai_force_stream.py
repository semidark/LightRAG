"""Tests for the ``force_stream`` option of ``openai_complete_if_cache``.

``force_stream=True`` sends a call that did not ask for a stream with
``stream=True`` and aggregates the chunks back into the non-streaming return
value. The contract pinned here: for every response shape, the forced-stream
path returns exactly what the plain non-streaming path returns (same value,
same ``TruncatedResponse`` flag, same exception type, same token usage), while
the request on the wire carries ``stream`` + ``stream_options.include_usage``
and never the ``force_stream`` key itself.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tenacity import RetryError, wait_none

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm.binding_options import OpenAILLMOptions
from lightrag.llm.openai import (
    InvalidResponseError,
    azure_openai_complete_if_cache,
    openai_complete_if_cache,
)
from lightrag.utils import is_truncated_response

USAGE = (10, 20)


def _usage(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _make_completion(content, reasoning_content="", finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,
                    parsed=None,
                ),
            )
        ],
        usage=_usage(*USAGE),
    )


def _chunk(content=None, reasoning_content=None, finish_reason=None, index=0):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                index=index,
                delta=SimpleNamespace(
                    content=content, reasoning_content=reasoning_content
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _stream_chunks(content, reasoning_content="", finish_reason="stop"):
    """Split a completion into chunks the way OpenAI-compatible servers do."""
    chunks = []
    if reasoning_content:
        for i in range(0, len(reasoning_content), 3):
            chunks.append(_chunk(reasoning_content=reasoning_content[i : i + 3]))
    if content:
        for i in range(0, len(content), 4):
            chunks.append(_chunk(content=content[i : i + 4]))
    chunks.append(_chunk(finish_reason=finish_reason))
    # Trailing usage chunk (stream_options.include_usage): empty choices.
    chunks.append(SimpleNamespace(usage=_usage(*USAGE), choices=[]))
    return chunks


class _FakeAsyncStream:
    def __init__(self, chunks, fail_after=None):
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise ConnectionResetError("connection reset by gateway")
            yield chunk

    async def close(self):
        self.closed = True


class _Tracker:
    def __init__(self):
        self.usage = []

    def add_usage(self, usage):
        self.usage.append(usage)


def _make_client(case, fail_after=None):
    """Fake AsyncOpenAI client answering both streaming and plain requests."""
    streams = []

    async def create(**kwargs):
        if kwargs.get("stream"):
            stream = _FakeAsyncStream(_stream_chunks(**case), fail_after=fail_after)
            streams.append(stream)
            return stream
        return _make_completion(**case)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=create))
        ),
        close=AsyncMock(),
    )
    client.streams = streams
    return client


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())


async def _call(case, *, force_stream, enable_cot=False, **kwargs):
    """Run one completion; return (outcome, client, tracker)."""
    client = _make_client(case)
    tracker = _Tracker()
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=client):
        try:
            result = await openai_complete_if_cache(
                model="test-model",
                prompt="Extract entities",
                enable_cot=enable_cot,
                token_tracker=tracker,
                force_stream=force_stream,
                **kwargs,
            )
            outcome = ("ok", str(result), is_truncated_response(result))
        except RetryError as e:
            inner = e.last_attempt.exception()
            outcome = ("retry_error", type(inner).__name__)
        except Exception as e:
            outcome = ("error", type(e).__name__)
    return outcome, client, tracker


CASES = {
    "normal": dict(content='{"entities": [1, 2, 3]} \u00e4\u00f6\u00fc'),
    "truncated": dict(content='{"entities": [1, 2', finish_reason="length"),
    "reasoning_only": dict(content=None, reasoning_content="thinking hard"),
    "content_and_reasoning": dict(content="answer", reasoning_content="thinking"),
    "empty": dict(content=None),
    "empty_truncated": dict(content=None, finish_reason="length"),
}


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("enable_cot", [False, True])
@pytest.mark.parametrize("case_name", list(CASES))
async def test_force_stream_matches_non_streaming(case_name, enable_cot, no_retry_wait):
    """Forced streaming is indistinguishable from the non-streaming path."""
    case = CASES[case_name]

    plain, plain_client, plain_tracker = await _call(
        case, force_stream=False, enable_cot=enable_cot
    )
    forced, forced_client, forced_tracker = await _call(
        case, force_stream=True, enable_cot=enable_cot
    )

    assert forced == plain
    assert forced_tracker.usage == plain_tracker.usage
    assert forced_tracker.usage  # usage from the trailing chunk was counted

    request = forced_client.chat.completions.create.await_args.kwargs
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert "force_stream" not in request
    forced_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_force_stream_empty_truncated_fails_fast(no_retry_wait):
    """Deterministic empty token-limit responses keep their no-retry type."""
    outcome, client, _ = await _call(CASES["empty_truncated"], force_stream=True)

    assert outcome == ("error", EmptyTruncatedResponseError.__name__)
    assert client.chat.completions.create.await_count == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_force_stream_mid_stream_failure_is_retryable(no_retry_wait):
    """A dropped stream becomes the retryable InvalidResponseError."""
    client = _make_client(CASES["normal"], fail_after=2)
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=client):
        with pytest.raises(RetryError) as exc_info:
            await openai_complete_if_cache(
                model="test-model", prompt="Extract entities", force_stream=True
            )

    assert isinstance(exc_info.value.last_attempt.exception(), InvalidResponseError)
    # Retried like any other transient failure, and every stream was released.
    assert client.chat.completions.create.await_count == 3
    assert client.streams and all(stream.closed for stream in client.streams)
    assert client.close.await_count >= 3


@pytest.mark.offline
@pytest.mark.asyncio
async def test_force_stream_keeps_caller_stream_options():
    """Caller-provided stream_options are kept; include_usage is only defaulted."""
    _, client, _ = await _call(
        CASES["normal"],
        force_stream=True,
        stream_options={"include_usage": False, "include_obfuscation": False},
    )

    request = client.chat.completions.create.await_args.kwargs
    assert request["stream_options"] == {
        "include_usage": False,
        "include_obfuscation": False,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_force_stream_ignores_other_choice_indexes():
    """Only choice 0 is aggregated, mirroring choices[0] on the plain path."""
    chunks = [
        _chunk(content="kept"),
        _chunk(content="IGNORED", index=1),
        _chunk(finish_reason="stop"),
        SimpleNamespace(usage=_usage(*USAGE), choices=[]),
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_FakeAsyncStream(chunks))
            )
        ),
        close=AsyncMock(),
    )
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=client):
        result = await openai_complete_if_cache(
            model="test-model", prompt="p", force_stream=True
        )

    assert result == "kept"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_caller_requested_stream_is_untouched():
    """stream=True callers (queries) still get a real async iterator."""
    client = _make_client(dict(content="streamed answer"))
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=client):
        result = await openai_complete_if_cache(
            model="test-model", prompt="p", stream=True, force_stream=True
        )
        assert hasattr(result, "__aiter__")
        text = "".join([piece async for piece in result])

    assert text == "streamed answer"
    request = client.chat.completions.create.await_args.kwargs
    assert request["stream"] is True
    assert "stream_options" not in request
    assert "force_stream" not in request


@pytest.mark.offline
@pytest.mark.asyncio
async def test_force_stream_off_leaves_request_unchanged():
    """Without the option the request carries no stream keys at all."""
    _, client, _ = await _call(CASES["normal"], force_stream=False)

    request = client.chat.completions.create.await_args.kwargs
    assert "stream" not in request
    assert "stream_options" not in request
    assert "force_stream" not in request


@pytest.mark.offline
@pytest.mark.asyncio
async def test_azure_delegation_honours_force_stream():
    """Azure shares the unified completion path, so the option applies too."""
    client = _make_client(CASES["normal"])
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=client):
        result = await azure_openai_complete_if_cache(
            model="test-deployment", prompt="p", force_stream=True
        )

    assert result == CASES["normal"]["content"]
    assert client.chat.completions.create.await_args.kwargs["stream"] is True


@pytest.mark.offline
def test_force_stream_binding_option_parsing(monkeypatch):
    """OPENAI_LLM_FORCE_STREAM is a registered bool option (base and role)."""
    import argparse

    monkeypatch.setenv("OPENAI_LLM_FORCE_STREAM", "true")
    parser = argparse.ArgumentParser()
    OpenAILLMOptions.add_args(parser)
    args = parser.parse_args([])
    assert OpenAILLMOptions.options_dict(args)["force_stream"] is True

    monkeypatch.setenv("EXTRACT_OPENAI_LLM_FORCE_STREAM", "false")
    role_options = OpenAILLMOptions.options_dict_for_role(args, "extract")
    assert role_options["force_stream"] is False

    monkeypatch.delenv("OPENAI_LLM_FORCE_STREAM")
    monkeypatch.delenv("EXTRACT_OPENAI_LLM_FORCE_STREAM")
    parser = argparse.ArgumentParser()
    OpenAILLMOptions.add_args(parser)
    # Unset means "not sent" (argparse.SUPPRESS), like every other option.
    assert "force_stream" not in OpenAILLMOptions.options_dict(parser.parse_args([]))
