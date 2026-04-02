from pathlib import Path
import tempfile

import pytest

from flocks.storage.storage import Storage
from flocks.tool.discovery import clear_discovered_tools, get_discovered_tools, remember_discovered_tools


@pytest.fixture
async def discovery_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_discovery.db"
        await Storage.init(db_path)
        yield
        await Storage.clear()


@pytest.mark.asyncio
async def test_discovery_persists_unique_sorted_tools(discovery_storage) -> None:
    await remember_discovered_tools("session-discovery", ["websearch", "task", "websearch"])

    result = await get_discovered_tools("session-discovery")

    assert result == {"task", "websearch"}
    stored = await Storage.get("tool_discovery:session-discovery")
    assert stored == {"tools": ["task", "websearch"]}


@pytest.mark.asyncio
async def test_discovery_clear_removes_cache_and_storage(discovery_storage) -> None:
    await remember_discovered_tools("session-discovery-clear", ["websearch"])
    await clear_discovered_tools("session-discovery-clear")

    result = await get_discovered_tools("session-discovery-clear")

    assert result == set()
    assert await Storage.get("tool_discovery:session-discovery-clear") is None
