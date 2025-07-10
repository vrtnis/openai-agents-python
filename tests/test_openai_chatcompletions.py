from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import (
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from agents import (
    Agent,
    ModelResponse,
    ModelSettings,
    ModelSettings as AgentModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    OpenAIProvider,
    RunConfig,
    Runner,
    function_tool,
    generation_span,
)
from agents.models.chatcmpl_helpers import ChatCmplHelpers
from agents.models.fake_id import FAKE_RESPONSES_ID


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_text_message(monkeypatch) -> None:
    """
    When the model returns a ChatCompletionMessage with plain text content,
    `get_response` should produce a single `ResponseOutputMessage` containing
    a `ResponseOutputText` with that content, and a `Usage` populated from
    the completion's usage.
    """
    msg = ChatCompletionMessage(role="assistant", content="Hello")
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=CompletionUsage(
            completion_tokens=5,
            prompt_tokens=7,
            total_tokens=12,
            # completion_tokens_details left blank to test default
            prompt_tokens_details=PromptTokensDetails(cached_tokens=3),
        ),
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        prompt=None,
    )
    # Should have produced exactly one output message with one text part
    assert isinstance(resp, ModelResponse)
    assert len(resp.output) == 1
    assert isinstance(resp.output[0], ResponseOutputMessage)
    msg_item = resp.output[0]
    assert len(msg_item.content) == 1
    assert isinstance(msg_item.content[0], ResponseOutputText)
    assert msg_item.content[0].text == "Hello"
    # Usage should be preserved from underlying ChatCompletion.usage
    assert resp.usage.input_tokens == 7
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 12
    assert resp.usage.input_tokens_details.cached_tokens == 3
    assert resp.usage.output_tokens_details.reasoning_tokens == 0
    assert resp.response_id is None


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_refusal(monkeypatch) -> None:
    """
    When the model returns a ChatCompletionMessage with a `refusal` instead
    of normal `content`, `get_response` should produce a single
    `ResponseOutputMessage` containing a `ResponseOutputRefusal` part.
    """
    msg = ChatCompletionMessage(role="assistant", refusal="No thanks")
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=None,
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        prompt=None,
    )
    assert len(resp.output) == 1
    assert isinstance(resp.output[0], ResponseOutputMessage)
    refusal_part = resp.output[0].content[0]
    assert isinstance(refusal_part, ResponseOutputRefusal)
    assert refusal_part.refusal == "No thanks"
    # With no usage from the completion, usage defaults to zeros.
    assert resp.usage.requests == 0
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0
    assert resp.usage.input_tokens_details.cached_tokens == 0
    assert resp.usage.output_tokens_details.reasoning_tokens == 0


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_tool_call(monkeypatch) -> None:
    """
    If the ChatCompletionMessage includes one or more tool_calls, `get_response`
    should append corresponding `ResponseFunctionToolCall` items after the
    assistant message item with matching name/arguments.
    """
    tool_call = ChatCompletionMessageToolCall(
        id="call-id",
        type="function",
        function=Function(name="do_thing", arguments="{'x':1}"),
    )
    msg = ChatCompletionMessage(role="assistant", content="Hi", tool_calls=[tool_call])
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=None,
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        prompt=None,
    )
    # Expect a message item followed by a function tool call item.
    assert len(resp.output) == 2
    assert isinstance(resp.output[0], ResponseOutputMessage)
    fn_call_item = resp.output[1]
    assert isinstance(fn_call_item, ResponseFunctionToolCall)
    assert fn_call_item.call_id == "call-id"
    assert fn_call_item.name == "do_thing"
    assert fn_call_item.arguments == "{'x':1}"


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_get_response_with_no_message(monkeypatch) -> None:
    """If the model returns no message, get_response should return an empty output."""
    msg = ChatCompletionMessage(role="assistant", content="ignored")
    choice = Choice(index=0, finish_reason="content_filter", message=msg)
    choice.message = None  # type: ignore[assignment]
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
        usage=None,
    )

    async def patched_fetch_response(self, *args, **kwargs):
        return chat

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", patched_fetch_response)
    model = OpenAIProvider(use_responses=False).get_model("gpt-4")
    resp: ModelResponse = await model.get_response(
        system_instructions=None,
        input="",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        prompt=None,
    )
    assert resp.output == []


@pytest.mark.asyncio
async def test_fetch_response_non_stream(monkeypatch) -> None:
    """
    Verify that `_fetch_response` builds the correct OpenAI API call when not
    streaming and returns the ChatCompletion object directly. We supply a
    dummy ChatCompletion through a stubbed OpenAI client and inspect the
    captured kwargs.
    """

    # Dummy completions to record kwargs
    class DummyCompletions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return chat

    class DummyClient:
        def __init__(self, completions: DummyCompletions) -> None:
            self.chat = type("_Chat", (), {"completions": completions})()
            self.base_url = httpx.URL("http://fake")

    msg = ChatCompletionMessage(role="assistant", content="ignored")
    choice = Choice(index=0, finish_reason="stop", message=msg)
    chat = ChatCompletion(
        id="resp-id",
        created=0,
        model="fake",
        object="chat.completion",
        choices=[choice],
    )
    completions = DummyCompletions()
    dummy_client = DummyClient(completions)
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=dummy_client)  # type: ignore
    # Execute the private fetch with a system instruction and simple string input.
    with generation_span(disabled=True) as span:
        result = await model._fetch_response(
            system_instructions="sys",
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            span=span,
            tracing=ModelTracing.DISABLED,
            stream=False,
        )
    assert result is chat
    # Ensure expected args were passed through to OpenAI client.
    kwargs = completions.kwargs
    assert kwargs["stream"] is False
    assert kwargs["store"] is NOT_GIVEN
    assert kwargs["model"] == "gpt-4"
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][0]["content"] == "sys"
    assert kwargs["messages"][1]["role"] == "user"
    # Defaults for optional fields become the NOT_GIVEN sentinel
    assert kwargs["tools"] is NOT_GIVEN
    assert kwargs["tool_choice"] is NOT_GIVEN
    assert kwargs["response_format"] is NOT_GIVEN
    assert kwargs["stream_options"] is NOT_GIVEN


@pytest.mark.asyncio
async def test_fetch_response_stream(monkeypatch) -> None:
    """
    When `stream=True`, `_fetch_response` should return a bare `Response`
    object along with the underlying async stream. The OpenAI client call
    should include `stream_options` to request usage-delimited chunks.
    """

    async def event_stream() -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield  # pragma: no cover

    class DummyCompletions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return event_stream()

    class DummyClient:
        def __init__(self, completions: DummyCompletions) -> None:
            self.chat = type("_Chat", (), {"completions": completions})()
            self.base_url = httpx.URL("http://fake")

    completions = DummyCompletions()
    dummy_client = DummyClient(completions)
    model = OpenAIChatCompletionsModel(model="gpt-4", openai_client=dummy_client)  # type: ignore
    with generation_span(disabled=True) as span:
        response, stream = await model._fetch_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            span=span,
            tracing=ModelTracing.DISABLED,
            stream=True,
        )
    # Check OpenAI client was called for streaming
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["store"] is NOT_GIVEN
    assert completions.kwargs["stream_options"] is NOT_GIVEN
    # Response is a proper openai Response
    assert isinstance(response, Response)
    assert response.id == FAKE_RESPONSES_ID
    assert response.model == "gpt-4"
    assert response.object == "response"
    assert response.output == []
    # We returned the async iterator produced by our dummy.
    assert hasattr(stream, "__aiter__")


def test_store_param():
    """Should default to True for OpenAI API calls, and False otherwise."""

    model_settings = ModelSettings()
    client = AsyncOpenAI()
    assert ChatCmplHelpers.get_store_param(client, model_settings) is True, (
        "Should default to True for OpenAI API calls"
    )

    model_settings = ModelSettings(store=False)
    assert ChatCmplHelpers.get_store_param(client, model_settings) is False, (
        "Should respect explicitly set store=False"
    )

    model_settings = ModelSettings(store=True)
    assert ChatCmplHelpers.get_store_param(client, model_settings) is True, (
        "Should respect explicitly set store=True"
    )

    client = AsyncOpenAI(base_url="http://www.notopenai.com")
    model_settings = ModelSettings()
    assert ChatCmplHelpers.get_store_param(client, model_settings) is None, (
        "Should default to None for non-OpenAI API calls"
    )

    model_settings = ModelSettings(store=False)
    assert ChatCmplHelpers.get_store_param(client, model_settings) is False, (
        "Should respect explicitly set store=False"
    )

    model_settings = ModelSettings(store=True)
    assert ChatCmplHelpers.get_store_param(client, model_settings) is True, (
        "Should respect explicitly set store=True"
    )


@function_tool
async def grab(x: int) -> int:
    return x * 2


async def collect_tool_events(run):
    seq = []
    async for ev in run.stream_events():
        if hasattr(ev, "name"):
            name = ev.name
            item_name = getattr(getattr(ev, "item", None), "name", None)
            seq.append((name, item_name))
    return seq


@pytest.mark.asyncio
async def test_as_tool_streams_nested_tool_calls(monkeypatch):
    """Verify nested tool events surface in order for a single wrapped tool."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    events = [
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})}),
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})}),
    ]

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in events:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="5")
    names = await collect_tool_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("tool_called", "grab"),
        ("tool_output", "grab"),
        ("tool_output", "grab_tool"),
    ]


@pytest.mark.asyncio
async def test_as_tool_parallel_streams(monkeypatch):
    """Verify nested events surface for multiple tools in parallel."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    seq = []
    for t in ("A", "B"):
        seq.extend(
            [
                type("E", (), {"name": "tool_called", "item": type("I", (), {"name": t})}),
                type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})}),
                type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab"})}),
                type("E", (), {"name": "tool_output", "item": type("I", (), {"name": t})}),
            ]
        )

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in seq:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    a = sub.as_tool("A", "A", stream_inner_events=True)
    b = sub.as_tool("B", "B", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[a, b])
    run = Runner.run_streamed(
        main,
        input="",
        run_config=RunConfig(model_settings=AgentModelSettings(parallel_tool_calls=True)),
    )
    names = await collect_tool_events(run)
    assert names.count(("tool_called", "grab")) == 2
    assert names.count(("tool_output", "grab")) == 2
    assert ("tool_called", "A") in names and ("tool_called", "B") in names


@pytest.mark.asyncio
async def test_as_tool_error_propagation(monkeypatch):
    """Errors from the inner agent surface via the outer tool output."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    events = [
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})}),
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})}),
    ]

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in events:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="boom")
    names = await collect_tool_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("tool_called", "grab"),
        ("tool_output", "grab_tool"),
    ]


@pytest.mark.asyncio
async def test_as_tool_empty_inner_run(monkeypatch):
    """Even if the inner agent does nothing, outer events still emit."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    events = [
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})}),
    ]

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in events:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="")
    names = await collect_tool_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("tool_output", "grab_tool"),
    ]


@pytest.mark.asyncio
async def test_as_tool_mixed_reasoning_and_tools(monkeypatch):
    """Interleaved reasoning and tool events maintain relative order."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    events = [
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})}),
        type("E", (), {"name": "reasoning_item_created", "item": type("I", (), {"name": "r1"})}),
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab"})}),
        type("E", (), {"name": "reasoning_item_created", "item": type("I", (), {"name": "r2"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})}),
    ]

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in events:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="mix")
    names = await collect_tool_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("reasoning_item_created", "r1"),
        ("tool_called", "grab"),
        ("tool_output", "grab"),
        ("reasoning_item_created", "r2"),
        ("tool_output", "grab_tool"),
    ]


@pytest.mark.asyncio
async def test_as_tool_multiple_inner_tools(monkeypatch):
    """Inner agent calling two tools produces events for each."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    events = [
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})}),
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab1"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab1"})}),
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab2"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab2"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})}),
    ]

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in events:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="multi")
    names = await collect_tool_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("tool_called", "grab1"),
        ("tool_output", "grab1"),
        ("tool_called", "grab2"),
        ("tool_output", "grab2"),
        ("tool_output", "grab_tool"),
    ]


@pytest.mark.asyncio
async def test_as_tool_heavy_concurrency_ordering(monkeypatch):
    """Parallel runs with many tools preserve inner ordering."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    seq = []
    for t in ("A", "B", "C"):
        seq.extend(
            [
                type("E", (), {"name": "tool_called", "item": type("I", (), {"name": t})}),
                type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})}),
                type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab"})}),
                type("E", (), {"name": "tool_output", "item": type("I", (), {"name": t})}),
            ]
        )

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in seq:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tools = [sub.as_tool(t, t, stream_inner_events=True) for t in ("A", "B", "C")]
    main = Agent(name="main", instructions="", tools=tools)
    run = Runner.run_streamed(
        main,
        input="",
        run_config=RunConfig(model_settings=AgentModelSettings(parallel_tool_calls=True)),
    )
    names = await collect_tool_events(run)
    assert names.count(("tool_called", "grab")) == 3
    assert names.count(("tool_output", "grab")) == 3
    for t in ("A", "B", "C"):
        assert names.count(("tool_called", t)) == 1
        assert names.count(("tool_output", t)) == 1


@pytest.mark.asyncio
async def test_as_tool_backward_compatibility(monkeypatch):
    """With stream_inner_events=False inner events are suppressed."""

    async def dummy_fetch_response(*args, **kwargs):
        raise AssertionError("_fetch_response should not be called")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "_fetch_response", dummy_fetch_response)

    events = [
        type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})}),
        type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})}),
    ]

    def fake_run_streamed(*args, **kwargs):
        class R:
            async def stream_events(self_inner):
                for e in events:
                    yield e

        return R()

    monkeypatch.setattr(Runner, "run_streamed", fake_run_streamed)

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=False)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="off")
    names = await collect_tool_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("tool_output", "grab_tool"),
    ]
