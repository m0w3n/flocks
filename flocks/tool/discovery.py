"""
Session-scoped discovered deferred tool state.
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

from flocks.storage.storage import Storage


_DISCOVERY_PREFIX = "tool_discovery:"
_cache: Dict[str, Set[str]] = {}


async def get_discovered_tools(session_id: str) -> Set[str]:
    if session_id in _cache:
        return set(_cache[session_id])
    stored = await Storage.get(f"{_DISCOVERY_PREFIX}{session_id}")
    if isinstance(stored, dict):
        names = set(str(name) for name in stored.get("tools", []) if name)
    elif isinstance(stored, list):
        names = set(str(name) for name in stored if name)
    else:
        names = set()
    _cache[session_id] = set(names)
    return set(names)


async def remember_discovered_tools(session_id: str, tool_names: Iterable[str]) -> Set[str]:
    current = await get_discovered_tools(session_id)
    current.update(str(name) for name in tool_names if name)
    normalized = set(sorted(current))
    _cache[session_id] = normalized
    await Storage.set(
        f"{_DISCOVERY_PREFIX}{session_id}",
        {"tools": sorted(normalized)},
        "tool_discovery",
    )
    return set(normalized)


async def clear_discovered_tools(session_id: str) -> None:
    _cache.pop(session_id, None)
    await Storage.delete(f"{_DISCOVERY_PREFIX}{session_id}")
