import pytest

from agents import Agent
from agents.exceptions import UserError
from agents.mcp.server import _MCPServerWithClientSession
from agents.run_context import RunContextWrapper


class CrashingClientSessionServer(_MCPServerWithClientSession):
    def __init__(self):
        super().__init__(cache_tools_list=False, client_session_timeout_seconds=5)
        self.cleanup_called = False

    def create_streams(self):
        raise ValueError("Crash!")

    async def cleanup(self):
        self.cleanup_called = True
        await super().cleanup()

    @property
    def name(self) -> str:
        return "crashing_client_session_server"


@pytest.mark.asyncio
async def test_server_errors_cause_error_and_cleanup_called():
    server = CrashingClientSessionServer()

    with pytest.raises(ValueError):
        await server.connect()

    assert server.cleanup_called


@pytest.mark.asyncio
async def test_not_calling_connect_causes_error():
    server = CrashingClientSessionServer()

    run_context = RunContextWrapper(context=None)
    agent = Agent(name="test_agent", instructions="Test agent")

    with pytest.raises(UserError):
        await server.list_tools(run_context, agent)

    with pytest.raises(UserError):
        await server.call_tool("foo", {})
