import pytest

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool


@function_tool
async def grab(x: int) -> int:
    return x * 2


async def collect_events(run):
    seq = []
    async for ev in run.stream_events():
        if ev.name in {"tool_called", "tool_output"}:
            seq.append((ev.name, ev.item.name))
    return seq


@pytest.mark.asyncio
async def test_stream_inner_events_single_agent(monkeypatch):
    """Verify we stream inner tool events for a single agent."""

    async def fake_stream(self):
        yield type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab_tool"})})
        yield type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})})
        yield type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab"})})
        yield type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab_tool"})})

    monkeypatch.setattr(
        Runner,
        "run_streamed",
        lambda *args, **kwargs: type("R", (), {"stream_events": fake_stream})(),
    )

    sub = Agent(name="sub", instructions="", tools=[grab])
    tool = sub.as_tool("grab_tool", "test", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[tool])
    run = Runner.run_streamed(main, input="5")
    names = await collect_events(run)
    assert names == [
        ("tool_called", "grab_tool"),
        ("tool_called", "grab"),
        ("tool_output", "grab"),
        ("tool_output", "grab_tool"),
    ]


@pytest.mark.asyncio
async def test_parallel_stream_inner_events(monkeypatch):
    """Verify we stream inner tool events for parallel tools."""

    async def fake_stream(self):
        for tool_name in ("A", "B"):
            yield type("E", (), {"name": "tool_called", "item": type("I", (), {"name": tool_name})})
            yield type("E", (), {"name": "tool_called", "item": type("I", (), {"name": "grab"})})
            yield type("E", (), {"name": "tool_output", "item": type("I", (), {"name": "grab"})})
            yield type("E", (), {"name": "tool_output", "item": type("I", (), {"name": tool_name})})

    monkeypatch.setattr(
        Runner,
        "run_streamed",
        lambda *args, **kwargs: type("R", (), {"stream_events": fake_stream})(),
    )

    sub = Agent(name="sub", instructions="", tools=[grab])
    a = sub.as_tool("A", "A", stream_inner_events=True)
    b = sub.as_tool("B", "B", stream_inner_events=True)
    main = Agent(name="main", instructions="", tools=[a, b])
    run = Runner.run_streamed(
        main,
        input="",
        run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=True)),
    )
    names = await collect_events(run)
    assert names.count(("tool_called", "grab")) == 2
    assert names.count(("tool_output", "grab")) == 2
    assert ("tool_called", "A") in names and ("tool_called", "B") in names
