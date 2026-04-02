"""
Tool search / discovery helper.

Lets the model discover long-tail tools on demand instead of requiring the full
tool catalog to be injected into every turn.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

from flocks.tool.discovery import remember_discovered_tools
from flocks.tool.policy import get_tool_policy
from flocks.tool.registry import (
    ParameterType,
    ToolCategory,
    ToolContext,
    ToolParameter,
    ToolRegistry,
    ToolResult,
)


DESCRIPTION = """Search available tools by task intent, keyword, or category.

Use this tool when you need to discover a tool that is not already exposed in
the current turn. Search by user goal, capability, or keyword, then choose from
the returned candidates. Matching deferred tools discovered here become
available in later turns."""


def _score_tool(query: str, category: Optional[str], tool_info) -> Tuple[int, List[str]]:
    q = (query or "").strip().lower()
    tokens = [token for token in q.split() if token]
    name = tool_info.name.lower()
    desc = (tool_info.description or "").lower()
    source = (tool_info.source or "").lower()
    tool_category = getattr(tool_info.category, "value", str(tool_info.category)).lower()
    policy = get_tool_policy(tool_info.name, tool_info)
    tags = [tag.lower() for tag in policy.tags]
    matched_tags = [tag for tag in policy.tags if q and tag.lower() in q]

    score = 0
    if not q:
        score += 10
    if q and q in name:
        score += 120
    if q and any(token in name for token in tokens):
        score += 55
    if q and q in desc:
        score += 40
    if q and any(token in desc for token in tokens):
        score += 20
    if q and q in source:
        score += 10
    if q and any(token in tag for token in tokens for tag in tags):
        score += 75
        matched_tags = list(dict.fromkeys(
            matched_tags + [tag for tag in policy.tags if any(token in tag.lower() for token in tokens)]
        ))
    if category and tool_category == category.lower():
        score += 60
    if q and q in tool_category:
        score += 20
    if policy.always_load:
        score += 5
    if getattr(tool_info, "requires_confirmation", False):
        score -= 5
    return score, matched_tags


@ToolRegistry.register_function(
    name="tool_search",
    description=DESCRIPTION,
    category=ToolCategory.SYSTEM,
    parameters=[
        ToolParameter(
            name="query",
            type=ParameterType.STRING,
            description="Search query describing the capability or task intent",
            required=False,
        ),
        ToolParameter(
            name="category",
            type=ParameterType.STRING,
            description="Optional category filter such as file, search, code, terminal, system, browser, custom",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type=ParameterType.INTEGER,
            description="Maximum number of matching tools to return",
            required=False,
            default=8,
        ),
    ],
)
async def tool_search(
    ctx: ToolContext,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 8,
) -> ToolResult:
    limit = max(1, min(limit or 8, 20))

    tools = []
    for tool_info in ToolRegistry.list_tools():
        if not getattr(tool_info, "enabled", True):
            continue
        if tool_info.name in {"invalid", "_noop"}:
            continue
        if category:
            tool_category = getattr(tool_info.category, "value", str(tool_info.category))
            if tool_category.lower() != category.lower():
                continue
        score, matched_tags = _score_tool(query or "", category, tool_info)
        if query and score <= 0:
            continue
        tools.append((score, tool_info, matched_tags))

    tools.sort(key=lambda item: (-item[0], item[1].name))
    matches = []
    discovered_candidates: List[str] = []
    matched_tag_set = set()
    for score, tool_info, matched_tags in tools[:limit]:
        policy = get_tool_policy(tool_info.name, tool_info)
        matched_tag_set.update(matched_tags)
        if policy.should_defer:
            discovered_candidates.append(tool_info.name)
        matches.append({
            "name": tool_info.name,
            "description": tool_info.description,
            "category": getattr(tool_info.category, "value", str(tool_info.category)),
            "requires_confirmation": getattr(tool_info, "requires_confirmation", False),
            "source": getattr(tool_info, "source", None),
            "native": getattr(tool_info, "native", False),
            "should_defer": policy.should_defer,
            "always_load": policy.always_load,
            "risk_level": policy.risk_level,
            "tags": policy.tags,
            "matchedTags": matched_tags,
            "score": score,
        })

    discovered_tools = await remember_discovered_tools(ctx.session_id, discovered_candidates)
    if ctx.event_publish_callback:
        await ctx.event_publish_callback("runtime.tool_discovery", {
            "sessionID": ctx.session_id,
            "query": query or "",
            "category": category,
            "returnedToolCount": len(matches),
            "discoveredToolCount": len(discovered_tools),
            "discoveredToolNames": sorted(discovered_candidates),
            "matchedTags": sorted(matched_tag_set),
        })

    return ToolResult(
        success=True,
        output={
            "query": query or "",
            "category": category,
            "count": len(matches),
            "matchedTags": sorted(matched_tag_set),
            "discoveredToolNames": sorted(discovered_candidates),
            "discoveredToolCount": len(discovered_tools),
            "matches": matches,
        },
    )
