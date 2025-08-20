import asyncio
import pytest

from agents.run import Runner, RunConfig
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing  # <-- import the interface
from agents.items import TResponseInputItem
from agents.agent_output import AgentOutputSchemaBase
from agents.tool import Tool
from agents.handoffs import Handoff
from collections.abc import AsyncIterator
from openai.types.responses import ResponseStreamEvent
from agents.stream_events import RunUpdatedStreamEvent

class FakeModel(Model):  # <-- subclass the interface
    async def get_response(
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
    ):
        raise NotImplementedError  # not used in this test

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
            await asyncio.sleep(0.05)
            # Yield any object; runner only treats ResponseCompletedEvent specially
            yield object()



class MinimalAgent:
    """
    Minimal Agent-like object with the attributes/methods Runner needs.
    """
    def __init__(self, name="test-agent", model=None):
        self.name = name
        self.model = model
        self.model_settings = ModelSettings()  # default settings
        self.output_type = None
        self.hooks = None
        self.handoffs = []
        self.reset_tool_choice = False
        self.input_guardrails = []
        self.output_guardrails = []

    async def get_system_prompt(self, context_wrapper):
        return None

    async def get_prompt(self, context_wrapper):
        return None

    async def get_all_tools(self, context_wrapper):
        return []


@pytest.mark.anyio
async def test_cancel_streamed_run_emits_cancelled_status():
    agent = MinimalAgent(model=FakeModel())
    # Force our FakeModel via RunConfig.model
    run_config = RunConfig(model=agent.model)

    # Start the streamed run
    result = Runner.run_streamed(
        starting_agent=agent,
        input="hello world",
        run_config=run_config,
        max_turns=10,
    )
    result._emit_status_events = True
    # Consume events until cancellation finishes the stream
    seen_status = None

    async def consume():
        nonlocal seen_status
        async for ev in result.stream_events():
            # Capture the high-level run status updates
            if isinstance(ev, RunUpdatedStreamEvent):
                seen_status = ev.status

    consumer = asyncio.create_task(consume())

    # Let it stream a bit, then cancel
    await asyncio.sleep(0.15)
    result.cancel("user-requested")

    # Wait for the stream to wind down
    await consumer

    # Assertions: stream completed and we saw a "cancelled" status event
    assert result.is_complete is True
    assert seen_status == "cancelled"
