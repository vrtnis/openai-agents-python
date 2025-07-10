import json

import pytest

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool

from .fake_model import FakeModel
from .test_responses import get_function_tool_call, get_text_message


@function_tool
async def grab(x: int) -> int:
    return x * 2


async def collect_events(run):
    seq = []
    async for ev in run.stream_events():
        if hasattr(ev, "item") and ev.name in {"tool_called", "tool_output"}:
            seq.append(ev.name)
    return seq


@pytest.mark.asyncio
async def test_stream_inner_events_single_agent():
    sub_model = FakeModel()
    sub_model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("grab", json.dumps({"x": 5}))],
            [get_text_message("done")],
        ]
    )
    sub = Agent(name="sub", instructions="", model=sub_model, tools=[grab])
    tool = sub.as_tool(tool_name="grab_tool", tool_description="test", stream_inner_events=True)

    main_model = FakeModel()
    main_model.add_multiple_turn_outputs(
        [
            [get_function_tool_call("grab_tool", json.dumps({"input": "5"}))],
            [get_text_message("final")],
        ]
    )
    main = Agent(name="main", instructions="", model=main_model, tools=[tool])
    run = Runner.run_streamed(main, input="start")
    names = await collect_events(run)
    assert names == ["tool_called", "tool_output", "tool_called", "tool_output"]


@pytest.mark.asyncio
async def test_parallel_stream_inner_events():
    sub_model_a = FakeModel()
    sub_model_a.add_multiple_turn_outputs(
        [
            [get_function_tool_call("grab", json.dumps({"x": 1}))],
            [get_text_message("done")],
        ]
    )
    sub_a = Agent(name="sub", instructions="", model=sub_model_a, tools=[grab])
    a = sub_a.as_tool(tool_name="A", tool_description="A", stream_inner_events=True)

    sub_model_b = FakeModel()
    sub_model_b.add_multiple_turn_outputs(
        [
            [get_function_tool_call("grab", json.dumps({"x": 1}))],
            [get_text_message("done")],
        ]
    )
    sub_b = Agent(name="sub", instructions="", model=sub_model_b, tools=[grab])
    b = sub_b.as_tool(tool_name="B", tool_description="B", stream_inner_events=True)

    main_model = FakeModel()
    main_model.add_multiple_turn_outputs(
        [
            [
                get_function_tool_call("A", json.dumps({"input": ""})),
                get_function_tool_call("B", json.dumps({"input": ""})),
            ],
            [get_text_message("done")],
        ]
    )
    main = Agent(name="main", instructions="", model=main_model, tools=[a, b])
    run = Runner.run_streamed(
        main,
        input="",
        run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=True)),
    )
    names = await collect_events(run)
    assert names.count("tool_called") == 4
    assert names.count("tool_output") == 4
