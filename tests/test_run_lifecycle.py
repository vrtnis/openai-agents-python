import asyncio
import time
import pytest
from collections.abc import AsyncIterator

from agents.run import Runner, RunConfig
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.items import TResponseInputItem
from agents.agent_output import AgentOutputSchemaBase
from agents.tool import Tool
from agents.handoffs import Handoff
from openai.types.responses import ResponseStreamEvent
from agents.stream_events import RunUpdatedStreamEvent

# Reuse the repo’s helper to build a FunctionTool correctly
from tests.test_responses import get_function_tool  # <-- existing test helper


# -------------------------
# Shared minimal fakes
# -------------------------

class MinimalAgent:
    """Just enough surface for Runner."""
    def __init__(self, model: Model, name: str = "test-agent"):
        self.name = name
        self.model = model
        self.model_settings = ModelSettings()
        self.output_type = None
        self.hooks = None
        self.handoffs = []
        self.reset_tool_choice = False
        self.input_guardrails: list = []
        self.output_guardrails: list = []

    async def get_system_prompt(self, _): return None
    async def get_prompt(self, _): return None
    async def get_all_tools(self, _): return []


# -------------------------
# Tests for cancellation & status events
# -------------------------

class FakeModelNeverCompletes(Model):
    """Never completes; yields generic events forever so we can cancel mid-stream."""
    async def get_response(self, *a, **k):
        raise NotImplementedError

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt=None,
    ) -> AsyncIterator[ResponseStreamEvent]:
        while True:
            await asyncio.sleep(0.02)
            yield object()


@pytest.mark.anyio
async def test_cancel_streamed_run_emits_cancelled_status():
    """When status events are enabled, cancel should emit run.updated(cancelled)."""
    agent = MinimalAgent(model=FakeModelNeverCompletes())
    result = Runner.run_streamed(agent, input="hello world", run_config=RunConfig(model=agent.model), max_turns=10)
    # Opt-in to status events for this test
    result._emit_status_events = True

    seen_status = None

    async def consume():
        nonlocal seen_status
        async for ev in result.stream_events():
            if isinstance(ev, RunUpdatedStreamEvent):
                seen_status = ev.status

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.08)
    result.cancel("user-requested")
    await consumer

    assert result.is_complete is True
    assert seen_status == "cancelled"


@pytest.mark.anyio
async def test_default_flag_off_emits_no_status_event():
    """By default, no run.updated events should be emitted (back-compat)."""
    agent = MinimalAgent(model=FakeModelNeverCompletes())
    result = Runner.run_streamed(agent, input="x", run_config=RunConfig(model=agent.model))
    # DO NOT set result._emit_status_events here
    statuses = []

    async def consume():
        async for ev in result.stream_events():
            if isinstance(ev, RunUpdatedStreamEvent):
                statuses.append(ev.status)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    result.cancel("user")
    await task

    assert statuses == []  # no run.updated by default


@pytest.mark.anyio
async def test_midstream_cancel_emits_cancelled_status_when_enabled():
    """Cancel while model is streaming yields cancelled when flag is on."""
    agent = MinimalAgent(model=FakeModelNeverCompletes())
    result = Runner.run_streamed(agent, input="x", run_config=RunConfig(model=agent.model))
    result._emit_status_events = True
    statuses = []

    async def consume():
        async for ev in result.stream_events():
            if isinstance(ev, RunUpdatedStreamEvent):
                statuses.append(ev.status)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.06)
    result.cancel("user")
    await task

    assert "cancelled" in statuses


# -------------------------
# Non-streamed cancel
# -------------------------

class FakeModelSlowGet(Model):
    async def get_response(self, *a, **k):
        # simulate long compute so we can cancel
        await asyncio.sleep(1.0)

    async def stream_response(self, *a, **k):
        raise NotImplementedError


@pytest.mark.anyio
async def test_non_streamed_cancel_propagates_cancelled_error_or_returns_terminal_result():
    """Runner.run cancellation should terminate cleanly (either CancelledError or terminal RunResult)."""
    agent = MinimalAgent(model=FakeModelSlowGet())

    async def run_it():
        return await Runner.run(agent, input="y", run_config=RunConfig(model=agent.model))

    task = asyncio.create_task(run_it())
    await asyncio.sleep(0.05)
    task.cancel()

    try:
        result = await task
    except asyncio.CancelledError:
        # Current contract may propagate cancel; this is acceptable.
        return

    # If your contract returns a terminal result on cancel, assert it here.
    assert getattr(result, "final_output", None) is None


# -------------------------
# Inject mid-run
# -------------------------

@pytest.mark.anyio
async def test_inject_is_consumed_on_next_turn():
    """
    Injected items should be included in a subsequent model turn input.
    We capture the inputs passed into FakeModel each turn and assert presence.
    """
    INJECT_TOKEN = {"role": "user", "content": "INJECTED"}  # match message-style items

    class FakeModelCapture(Model):
        def __init__(self):
            self.inputs = []  # list[list[dict]]

        async def get_response(self, *a, **k):  # non-stream path not used
            raise NotImplementedError

        async def stream_response(
            self, system_instructions, input, model_settings, tools,
            output_schema, handoffs, tracing, previous_response_id, prompt=None
        ) -> AsyncIterator[ResponseStreamEvent]:
            # Keep streaming so we never hit the "no final response" error.
            while True:
                # record the input for this turn
                self.inputs.append(list(input))
                # emit one event to complete a turn
                yield object()
                await asyncio.sleep(0.01)

    model = FakeModelCapture()
    agent = MinimalAgent(model=model)

    result = Runner.run_streamed(
        starting_agent=agent,
        input="hello",
        run_config=RunConfig(model=agent.model),
        max_turns=6,
    )

    async def drive_and_inject():
        # Let at least one turn record baseline input
        await asyncio.sleep(0.05)
        # Inject so a future turn sees it
        result.inject([INJECT_TOKEN])
        # Give time for a couple more turns to run
        await asyncio.sleep(0.12)
        result.cancel("done")

    consumer = asyncio.create_task(drive_and_inject())
    async for _ in result.stream_events():
        pass
    await consumer

    # We should have recorded ≥2 turns
    assert len(model.inputs) >= 2

    # Assert the injected message appears in ANY turn after injection time
    flattened_after_injection = [item for turn in model.inputs[1:] for item in turn]
    assert any(
        isinstance(item, dict)
        and item.get("role") == "user"
        and item.get("content") == "INJECTED"
        for item in flattened_after_injection
    ), f"Injected item not present after injection; captured={model.inputs}"


# -------------------------
# Tool cancellation (function tool) — lightweight path
# -------------------------

class FakeModelTriggersTool(Model):
    """
    Emits continuous events so we can cancel while a function tool is (hypothetically) running.
    Note: This is a timing smoke test. For a full tool-call path test, emit tool-call outputs.
    """
    async def get_response(self, *a, **k): raise NotImplementedError

    async def stream_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, previous_response_id, prompt=None
    ) -> AsyncIterator[ResponseStreamEvent]:
        while True:
            await asyncio.sleep(0.02)
            yield object()

class AgentWithTool(MinimalAgent):
    def __init__(self, model: Model, tool: Tool):
        super().__init__(model)
        self._tool = tool
    async def get_all_tools(self, _): return [self._tool]


@pytest.mark.anyio
async def test_function_tool_cancels_promptly():
    # Build the tool using the repo helper (it doesn't take a handler argument)
    tool = get_function_tool("long", "done")

    agent = AgentWithTool(FakeModelTriggersTool(), tool)

    result = Runner.run_streamed(agent, input="trigger tool", run_config=RunConfig(model=agent.model))
    start = time.perf_counter()
    await asyncio.sleep(0.05)  # let some activity happen
    result.cancel("user")

    # Drain stream; ensure no hang
    async for _ in result.stream_events():
        pass

    elapsed = time.perf_counter() - start
    # Expect prompt cancellation (well under 1s)
    assert elapsed < 0.4

