"""
Request-time tool selection for built-in tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from flocks.agent.agent import AgentInfo
from flocks.tool.discovery import get_discovered_tools
from flocks.tool.governance import ToolGovernanceService
from flocks.tool.policy import CORE_TOOL_NAMES, get_tool_policy
from flocks.tool.registry import ToolCategory, ToolRegistry

if TYPE_CHECKING:
    from flocks.session.message import MessageInfo


MAX_OPTIONAL_TOOLS = 10
MIN_TOOL_SCORE = 35

TOOL_HINTS: Tuple[Tuple[Tuple[str, ...], Set[str]], ...] = (
    (("search", "find", "grep", "regex", "keyword"), {"grep", "glob", "list", "read"}),
    (("read", "open", "inspect", "analyze", "check"), {"read", "list", "glob", "grep"}),
    (("edit", "modify", "replace", "patch", "refactor"), {"edit", "multiedit", "apply_patch", "write"}),
    (("run", "command", "shell", "terminal", "install"), {"bash"}),
    (("web", "url", "http", "fetch", "site"), {"webfetch", "websearch"}),
    (("question", "confirm", "choose", "ask"), {"question"}),
    (("skill", "skills"), {"skill"}),
    (("tool", "tools", "capability", "available"), {"tool_search"}),
    (("delegate", "subagent", "agent"), {"delegate_task", "call_omo_agent", "task"}),
    (("workflow", "node", "pipeline"), {"run_workflow", "run_workflow_node"}),
    (("todo", "task list", "plan"), {"todowrite", "todoread", "task"}),
    (("session", "history", "conversation"), {"session_list", "session_get"}),
)


@dataclass
class RequestSelectionResult:
    selected_tool_infos: List[Any]
    metadata: Dict[str, Any]


class RequestToolSelector:
    def __init__(
        self,
        session_id: str,
        step: int,
        trusted: bool,
        event_publish_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self.session_id = session_id
        self.step = step
        self.trusted = trusted
        self.event_publish_callback = event_publish_callback

    async def _publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_publish_callback:
            return
        await self.event_publish_callback(event_type, {
            "sessionID": self.session_id,
            "step": self.step,
            **payload,
        })

    async def _collect_recent_context_signals(
        self,
        messages: List["MessageInfo"],
    ) -> Tuple[str, Set[str]]:
        from flocks.session.message import Message, MessageRole

        recent_text_chunks: List[str] = []
        recent_tool_names: Set[str] = set()

        for msg in messages[-8:]:
            text_content = (await Message.get_text_content(msg)).strip()
            if text_content:
                recent_text_chunks.append(text_content.lower())

            if msg.role != MessageRole.ASSISTANT:
                continue

            for part in await Message.parts(msg.id, self.session_id):
                if getattr(part, "type", None) != "tool":
                    continue
                tool_name = getattr(part, "tool", "")
                if tool_name:
                    recent_tool_names.add(tool_name)

        return "\n".join(recent_text_chunks)[-4000:], recent_tool_names

    def _score_tool_for_turn(
        self,
        agent: AgentInfo,
        tool_info: Any,
        recent_text: str,
        recent_tool_names: Set[str],
    ) -> Tuple[int, List[str]]:
        name = getattr(tool_info, "name", "")
        description = (getattr(tool_info, "description", "") or "").lower()
        normalized_name = name.lower().replace("_", " ").replace("-", " ")
        name_tokens = [token for token in normalized_name.split() if token]
        category = getattr(tool_info, "category", None)
        policy = get_tool_policy(name, tool_info)
        preferred_names = ToolGovernanceService.get_preferred_tool_names(agent)
        preferred_categories = ToolGovernanceService.get_preferred_categories(agent)
        matched_tags = [tag for tag in policy.tags if tag and tag.lower() in recent_text]

        score = 0
        if name in CORE_TOOL_NAMES:
            score += 100
        if policy.always_load:
            score += 120
        if name in preferred_names:
            score += 70
        if name in recent_tool_names:
            score += 80
        if name.lower() in recent_text:
            score += 65
        elif name_tokens and all(token in recent_text for token in name_tokens[:2]):
            score += 55
        elif name_tokens and any(token in recent_text for token in name_tokens):
            score += 28
        if description and any(token in recent_text for token in description.split()[:8]):
            score += 8
        if category in preferred_categories:
            score += 20
        if category in {ToolCategory.FILE, ToolCategory.SEARCH, ToolCategory.CODE}:
            score += 8
        if category == ToolCategory.TERMINAL:
            score += 4
        if matched_tags:
            score += 45 + len(matched_tags) * 5
        if policy.should_defer:
            score -= 20
        if getattr(tool_info, "requires_confirmation", False):
            score -= 5

        for keywords, hinted_tools in TOOL_HINTS:
            if name in hinted_tools and any(keyword in recent_text for keyword in keywords):
                score += 45

        return score, matched_tags

    async def select(
        self,
        agent: AgentInfo,
        messages: List["MessageInfo"],
    ) -> RequestSelectionResult:
        candidate_infos = []
        available_count = 0
        filtered_by_agent: List[str] = []
        filtered_by_trust: List[str] = []
        discovered_tools = await get_discovered_tools(self.session_id)
        deferred_count = 0
        hidden_deferred: List[str] = []

        for tool_info in ToolRegistry.list_tools():
            if tool_info.name in {"invalid", "_noop"} or not getattr(tool_info, "enabled", True):
                continue

            policy = get_tool_policy(tool_info.name, tool_info)
            available_count += 1
            decision = ToolGovernanceService.evaluate_exposure(
                agent,
                tool_info,
                trusted=self.trusted,
                discovered_tools=discovered_tools,
                tool_policy=policy,
            )
            if not decision.allowed and decision.reason != "workspace_trust" and decision.reason != "deferred_until_discovered":
                filtered_by_agent.append(tool_info.name)
                continue
            if decision.reason == "workspace_trust":
                filtered_by_trust.append(tool_info.name)
                continue
            if policy.should_defer:
                deferred_count += 1
                if decision.reason == "deferred_until_discovered":
                    hidden_deferred.append(tool_info.name)
                    continue
            candidate_infos.append(tool_info)

        if filtered_by_trust:
            await self._publish("runtime.permission_gate", {
                "trusted": self.trusted,
                "blockedToolNames": sorted(filtered_by_trust),
                "blockedCount": len(filtered_by_trust),
                "reason": "workspace_trust",
            })

        recent_text, recent_tool_names = await self._collect_recent_context_signals(messages)
        preferred_names = ToolGovernanceService.get_preferred_tool_names(agent)
        selected_names: Set[str] = set()
        selected_infos: List[Any] = []
        scored_infos: List[Tuple[int, Any, List[str]]] = []
        matched_tags: Set[str] = set()

        for tool_info in candidate_infos:
            score, tool_tags = self._score_tool_for_turn(agent, tool_info, recent_text, recent_tool_names)
            matched_tags.update(tool_tags)
            scored_infos.append((score, tool_info, tool_tags))
            policy = get_tool_policy(tool_info.name, tool_info)
            if tool_info.name in CORE_TOOL_NAMES or policy.always_load or tool_info.name in preferred_names:
                if tool_info.name not in selected_names:
                    selected_names.add(tool_info.name)
                    selected_infos.append(tool_info)

        for score, tool_info, _tool_tags in sorted(scored_infos, key=lambda item: (-item[0], item[1].name)):
            if tool_info.name in selected_names or score < MIN_TOOL_SCORE:
                continue
            selected_names.add(tool_info.name)
            selected_infos.append(tool_info)
            if len(selected_infos) >= len(CORE_TOOL_NAMES) + MAX_OPTIONAL_TOOLS:
                break

        if "question" not in selected_names:
            question_tool = next((tool for tool in candidate_infos if tool.name == "question"), None)
            if question_tool is not None:
                selected_names.add("question")
                selected_infos.append(question_tool)

        metadata = {
            "trusted": self.trusted,
            "availableToolCount": available_count,
            "candidateToolCount": len(candidate_infos),
            "selectedToolCount": len(selected_infos),
            "deferredToolCount": deferred_count,
            "discoveredToolCount": len(discovered_tools),
            "hiddenDeferredToolCount": len(hidden_deferred),
            "filteredByAgentCount": len(filtered_by_agent),
            "filteredByTrustCount": len(filtered_by_trust),
            "selectedToolNames": [tool.name for tool in selected_infos],
            "recentToolMatches": sorted(recent_tool_names),
            "matchedTags": sorted(matched_tags),
        }

        await self._publish("runtime.tool_deferred", {
            "deferredToolCount": deferred_count,
            "discoveredToolCount": len(discovered_tools),
            "hiddenDeferredToolCount": len(hidden_deferred),
            "hiddenDeferredToolNames": sorted(hidden_deferred)[:20],
        })
        await self._publish("runtime.tool_selection", metadata)

        return RequestSelectionResult(selected_tool_infos=selected_infos, metadata=metadata)
