from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flocks.tool.registry import ToolCategory, ToolInfo
from flocks.tool.system.tool_search import tool_search


def _tool(name: str, category: ToolCategory, native: bool = True) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=f"{name} description",
        category=category,
        native=native,
        enabled=True,
    )


@pytest.mark.asyncio
async def test_tool_search_discovers_deferred_tools_and_emits_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [
        _tool("websearch", ToolCategory.BROWSER),
        _tool("read", ToolCategory.FILE),
        _tool("plugin_only", ToolCategory.CUSTOM, native=False),
    ]
    remember = AsyncMock(return_value={"websearch"})
    event_callback = AsyncMock()

    monkeypatch.setattr("flocks.tool.system.tool_search.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr("flocks.tool.system.tool_search.remember_discovered_tools", remember)

    ctx = SimpleNamespace(session_id="session-3", event_publish_callback=event_callback)
    result = await tool_search(ctx, query="web", limit=5)

    assert result.success is True
    assert result.output["discoveredToolNames"] == ["websearch"]
    assert result.output["discoveredToolCount"] == 1
    assert result.output["matches"][0]["name"] == "websearch"
    assert result.output["matches"][0]["should_defer"] is True
    remember.assert_awaited_once_with("session-3", ["websearch"])
    event_callback.assert_awaited()


@pytest.mark.asyncio
async def test_tool_search_supports_category_and_tag_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [
        ToolInfo(
            name="websearch",
            description="Search the web for public information",
            category=ToolCategory.BROWSER,
            native=True,
            enabled=True,
            tags=["web", "research"],
        ),
        ToolInfo(
            name="read",
            description="Read local files",
            category=ToolCategory.FILE,
            native=True,
            enabled=True,
            tags=["code-reading"],
        ),
    ]

    monkeypatch.setattr("flocks.tool.system.tool_search.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr(
        "flocks.tool.system.tool_search.remember_discovered_tools",
        AsyncMock(return_value={"websearch"}),
    )

    ctx = SimpleNamespace(session_id="session-4", event_publish_callback=AsyncMock())
    result = await tool_search(ctx, query="research", category="browser", limit=5)

    assert result.success is True
    assert result.output["count"] == 1
    assert result.output["matches"][0]["name"] == "websearch"
    assert result.output["matches"][0]["matchedTags"] == ["research"]
    assert result.output["matchedTags"] == ["research"]


@pytest.mark.asyncio
async def test_tool_search_returns_user_plugin_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = [
        _tool("plugin_memory", ToolCategory.CUSTOM, native=False),
        _tool("read", ToolCategory.FILE),
    ]

    monkeypatch.setattr("flocks.tool.system.tool_search.ToolRegistry.list_tools", lambda: tools)
    monkeypatch.setattr(
        "flocks.tool.system.tool_search.remember_discovered_tools",
        AsyncMock(return_value=set()),
    )

    ctx = SimpleNamespace(session_id="session-plugin", event_publish_callback=AsyncMock())
    result = await tool_search(ctx, query="plugin_memory", limit=5)

    assert result.success is True
    assert result.output["count"] == 1
    assert result.output["matches"][0]["name"] == "plugin_memory"
    assert result.output["matches"][0]["native"] is False


def test_runtime_tool_events_are_recognized() -> None:
    from flocks.server.routes.event import is_runtime_event

    assert is_runtime_event("runtime.tool_selection") is True
    assert is_runtime_event("runtime.tool_discovery") is True
