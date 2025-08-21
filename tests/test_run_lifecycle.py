import asyncio
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


# -------------------------
# Minimal fakes used across tests
# -------------------------

class FakeModel(Model):
    """Never completes; yields generic events forever so we can cancel mid-stream."""
    async def get_response(self, *a, **k):
        raise NotImplementedError  # non-stream path not used here

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
            # Not a ResponseCompletedEvent; runner will keep streaming
            yield object()


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
# Tests
# -------------------------

@pytest.mark.anyio
async def test_cancel_streamed_run_emits_cancelled_status():
    """When status events are enabled, cancel should emit run.updated(cancelled)."""
    agent = MinimalAgent(model=FakeModel())
    run_config = RunConfig(model=agent.model)

    result = Runner.run_streamed(
        starting_agent=agent,
        input="hello world",
        run_config=run_config,
        max_turns=10,
    )
    # Opt-in to status events for this test
    result._emit_status_events = True

    seen_status = None

    async def consume():
        nonlocal seen_status
        async for ev in result.stream_events():
            if isinstance(ev, RunUpdatedStreamEvent):
                seen_status = ev.status

    consumer = asyncio.create_task(consume())

    await asyncio.sleep(0.08)  # allow a couple of ticks
    result.cancel("user-requested")
    await consumer

    assert result.is_complete is True
    assert seen_status == "cancelled"


@pytest.mark.anyio
async def test_default_flag_off_emits_no_status_event():
    """By default, no run.updated events should be emitted (back-compat)."""
    agent = MinimalAgent(model=FakeModel())
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
    agent = MinimalAgent(model=FakeModel())
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
        input="hello",                      # first turn will contain this
        run_config=RunConfig(model=agent.model),
        max_turns=6,
    )

    async def drive_and_inject():
        # Let at least one turn record baseline input
        await asyncio.sleep(0.05)
        # Inject so a future turn sees it
        result.inject([INJECT_TOKEN])
        # Give the runner time to execute another turn (or two) that should include the injection
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
