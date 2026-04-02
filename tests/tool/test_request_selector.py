from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flocks.agent.agent import AgentInfo
import flocks.session.message as session_message
from flocks.tool.request_selector import RequestToolSelector
from flocks.tool.registry import ToolCategory, ToolInfo


def _tool(name: str, category: ToolCategory, native: bool = True) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=f"{name} description",
        category=category,
        native=native,
        enabled=True,
    )


@pytest.mark.asyncio
async def test_selector_blocks_untrusted_high_risk_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [
        _tool("read", ToolCategory.FILE),
        _tool("question", ToolCategory.SYSTEM),
        _tool("tool_search", ToolCategory.SYSTEM),
        _tool("bash", ToolCategory.CODE),
    ]
    event_callback = AsyncMock()

    monkeypatch.setattr("flocks.tool.request_selector.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr("flocks.tool.request_selector.get_discovered_tools", AsyncMock(return_value=set()))
    monkeypatch.setattr(session_message.Message, "get_text_content", AsyncMock(return_value="read the file"))
    monkeypatch.setattr(session_message.Message, "parts", AsyncMock(return_value=[]))

    selector = RequestToolSelector(
        session_id="session-1",
        step=3,
        trusted=False,
        event_publish_callback=event_callback,
    )
    result = await selector.select(AgentInfo(name="rex", mode="primary"), [
        SimpleNamespace(id="m1", role="user"),
    ])

    names = [tool.name for tool in result.selected_tool_infos]
    assert "read" in names
    assert "bash" not in names
    assert result.metadata["filteredByTrustCount"] == 1
    event_types = [call.args[0] for call in event_callback.await_args_list]
    assert "runtime.permission_gate" in event_types
    assert "runtime.tool_selection" in event_types


@pytest.mark.asyncio
async def test_selector_keeps_terminal_tools_visible_for_primary_rex(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [
        _tool("read", ToolCategory.FILE),
        _tool("question", ToolCategory.SYSTEM),
        _tool("tool_search", ToolCategory.SYSTEM),
        _tool("bash", ToolCategory.TERMINAL),
    ]

    monkeypatch.setattr("flocks.tool.request_selector.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr("flocks.tool.request_selector.get_discovered_tools", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        session_message.Message,
        "get_text_content",
        AsyncMock(return_value="run a shell command and inspect output"),
    )
    monkeypatch.setattr(session_message.Message, "parts", AsyncMock(return_value=[]))

    selector = RequestToolSelector(session_id="session-terminal", step=1, trusted=True)
    result = await selector.select(AgentInfo(name="rex", mode="primary"), [
        SimpleNamespace(id="m1", role="user"),
    ])

    names = [tool.name for tool in result.selected_tool_infos]
    assert "bash" in names


@pytest.mark.asyncio
async def test_selector_includes_discovered_deferred_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [
        _tool("read", ToolCategory.FILE),
        _tool("question", ToolCategory.SYSTEM),
        _tool("tool_search", ToolCategory.SYSTEM),
        _tool("websearch", ToolCategory.BROWSER),
    ]

    monkeypatch.setattr("flocks.tool.request_selector.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr("flocks.tool.request_selector.get_discovered_tools", AsyncMock(return_value={"websearch"}))
    monkeypatch.setattr(
        session_message.Message,
        "get_text_content",
        AsyncMock(return_value="please use web research to search docs"),
    )
    monkeypatch.setattr(session_message.Message, "parts", AsyncMock(return_value=[]))

    selector = RequestToolSelector(session_id="session-2", step=1, trusted=True)
    result = await selector.select(AgentInfo(name="rex", mode="primary"), [
        SimpleNamespace(id="m1", role="user"),
    ])

    names = [tool.name for tool in result.selected_tool_infos]
    assert "websearch" in names
    assert result.metadata["discoveredToolCount"] == 1
    assert "web" in result.metadata["matchedTags"]


@pytest.mark.asyncio
async def test_selector_hides_undiscovered_deferred_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [
        _tool("read", ToolCategory.FILE),
        _tool("question", ToolCategory.SYSTEM),
        _tool("tool_search", ToolCategory.SYSTEM),
        _tool("websearch", ToolCategory.BROWSER),
    ]

    monkeypatch.setattr("flocks.tool.request_selector.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr("flocks.tool.request_selector.get_discovered_tools", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        session_message.Message,
        "get_text_content",
        AsyncMock(return_value="please use web research to search docs"),
    )
    monkeypatch.setattr(session_message.Message, "parts", AsyncMock(return_value=[]))

    selector = RequestToolSelector(session_id="session-3", step=2, trusted=True)
    result = await selector.select(AgentInfo(name="rex", mode="primary"), [
        SimpleNamespace(id="m1", role="user"),
    ])

    names = [tool.name for tool in result.selected_tool_infos]
    assert "websearch" not in names
    assert result.metadata["hiddenDeferredToolCount"] == 1


@pytest.mark.asyncio
async def test_selector_keeps_user_plugin_tools_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [
        _tool("read", ToolCategory.FILE),
        _tool("question", ToolCategory.SYSTEM),
        _tool("tool_search", ToolCategory.SYSTEM),
        _tool("project_memory", ToolCategory.CUSTOM, native=False),
    ]

    monkeypatch.setattr("flocks.tool.request_selector.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr("flocks.tool.request_selector.get_discovered_tools", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        session_message.Message,
        "get_text_content",
        AsyncMock(return_value="project memory plugin"),
    )
    monkeypatch.setattr(session_message.Message, "parts", AsyncMock(return_value=[]))

    selector = RequestToolSelector(session_id="session-plugin", step=1, trusted=True)
    result = await selector.select(AgentInfo(name="rex", mode="primary"), [
        SimpleNamespace(id="m1", role="user"),
    ])

    names = [tool.name for tool in result.selected_tool_infos]
    assert "project_memory" in names
