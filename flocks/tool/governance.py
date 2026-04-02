"""
Unified tool governance service for request-time exposure and execution gating.
"""

from __future__ import annotations

from typing import Any, Optional, Set

from pydantic import BaseModel, Field

from flocks.permission.rule import PermissionLevel
from flocks.tool.policy import ToolPolicyInfo, get_tool_policy
from flocks.tool.registry import ToolCategory


class ModePolicyInfo(BaseModel):
    allowed_categories: Set[ToolCategory] = Field(default_factory=set)
    preferred_categories: Set[ToolCategory] = Field(default_factory=set)
    preferred_tools: Set[str] = Field(default_factory=set)
    block_high_risk: bool = False


class ToolExposureDecision(BaseModel):
    allowed: bool
    reason: str
    policy: ToolPolicyInfo


MODE_POLICIES = {
    "primary": ModePolicyInfo(
        allowed_categories={
            ToolCategory.FILE,
            ToolCategory.SEARCH,
            ToolCategory.CODE,
            ToolCategory.TERMINAL,
            ToolCategory.SYSTEM,
            ToolCategory.BROWSER,
            ToolCategory.CUSTOM,
        },
        preferred_categories={
            ToolCategory.FILE,
            ToolCategory.SEARCH,
            ToolCategory.CODE,
            ToolCategory.TERMINAL,
            ToolCategory.SYSTEM,
        },
        preferred_tools={"tool_search", "question"},
        block_high_risk=False,
    ),
    "subagent": ModePolicyInfo(
        allowed_categories={
            ToolCategory.FILE,
            ToolCategory.SEARCH,
            ToolCategory.CODE,
            ToolCategory.SYSTEM,
            ToolCategory.CUSTOM,
        },
        preferred_categories={
            ToolCategory.FILE,
            ToolCategory.SEARCH,
            ToolCategory.CODE,
        },
        preferred_tools={"read", "glob", "grep", "list"},
        block_high_risk=True,
    ),
    "all": ModePolicyInfo(),
}

AGENT_PREFERRED_TOOLS = {
    "rex": {"tool_search", "delegate_task", "task", "question"},
    "hephaestus": {"tool_search", "run_workflow", "run_workflow_node", "question"},
    "plan": {"tool_search", "question", "todoread", "todowrite"},
}

TOOL_POLICY_PRESETS = {
    "orchestration": ModePolicyInfo(
        preferred_categories={
            ToolCategory.SYSTEM,
            ToolCategory.CODE,
            ToolCategory.SEARCH,
        },
        preferred_tools={"tool_search", "question", "task", "delegate_task"},
    ),
    "coding": ModePolicyInfo(
        preferred_categories={
            ToolCategory.FILE,
            ToolCategory.SEARCH,
            ToolCategory.CODE,
        },
        preferred_tools={"read", "glob", "grep", "edit", "apply_patch"},
    ),
    "planning": ModePolicyInfo(
        preferred_categories={ToolCategory.SYSTEM, ToolCategory.SEARCH},
        preferred_tools={"tool_search", "question", "todoread", "todowrite"},
    ),
}


class ToolGovernanceService:
    @staticmethod
    def _matches_legacy_permission(agent: Any, tool_name: str) -> Optional[bool]:
        permission_rules = getattr(agent, "permission", None) or []
        if not permission_rules:
            return None

        try:
            from flocks.permission.next import PermissionNext
        except Exception:
            return None

        for rule in reversed(permission_rules):
            rule_perm = getattr(rule, "permission", None) or (
                rule.get("permission") if isinstance(rule, dict) else None
            )
            if not rule_perm:
                continue
            if not PermissionNext._pattern_matches(tool_name, rule_perm):
                continue
            level = getattr(rule, "level", None) or getattr(rule, "action", None) or (
                rule.get("level") if isinstance(rule, dict) else None
            )
            level_value = level.value if isinstance(level, PermissionLevel) else str(level)
            if level_value == "deny":
                return False
            if level_value == "allow":
                return True
        return None

    @classmethod
    def get_mode_policy(cls, mode: str) -> ModePolicyInfo:
        return MODE_POLICIES.get(mode or "all", MODE_POLICIES["all"])

    @classmethod
    def get_preferred_tool_names(cls, agent: Any) -> Set[str]:
        preferred = set(cls.get_mode_policy(getattr(agent, "mode", "all")).preferred_tools)
        preferred.update(AGENT_PREFERRED_TOOLS.get(getattr(agent, "name", ""), set()))
        preset_name = getattr(agent, "tool_policy_preset", None)
        if preset_name:
            preferred.update(TOOL_POLICY_PRESETS.get(preset_name, ModePolicyInfo()).preferred_tools)
        preferred.update(set(getattr(agent, "allowed_tools", None) or []))
        return preferred

    @classmethod
    def get_preferred_categories(cls, agent: Any) -> Set[ToolCategory]:
        preferred = set(cls.get_mode_policy(getattr(agent, "mode", "all")).preferred_categories)
        preset_name = getattr(agent, "tool_policy_preset", None)
        if preset_name:
            preferred.update(TOOL_POLICY_PRESETS.get(preset_name, ModePolicyInfo()).preferred_categories)
        return preferred

    @classmethod
    def evaluate_agent_gate(
        cls,
        agent: Any,
        tool_info: Any,
        tool_policy: Optional[ToolPolicyInfo] = None,
    ) -> ToolExposureDecision:
        tool_name = getattr(tool_info, "name", "")
        policy = tool_policy or get_tool_policy(tool_name, tool_info)

        explicit_disallowed = set(getattr(agent, "disallowed_tools", None) or [])
        if tool_name in explicit_disallowed:
            return ToolExposureDecision(allowed=False, reason="explicit_disallowed", policy=policy)

        explicit_allowed = set(getattr(agent, "allowed_tools", None) or [])
        if explicit_allowed and tool_name not in explicit_allowed:
            return ToolExposureDecision(allowed=False, reason="explicit_allowlist", policy=policy)

        if policy.disallowed_agents and getattr(agent, "name", "") in policy.disallowed_agents:
            return ToolExposureDecision(allowed=False, reason="disallowed_agent", policy=policy)

        mode = getattr(agent, "mode", "all")
        if policy.allowed_modes and mode not in policy.allowed_modes:
            return ToolExposureDecision(allowed=False, reason="disallowed_mode", policy=policy)

        mode_policy = cls.get_mode_policy(mode)
        if mode_policy.allowed_categories and getattr(tool_info, "category", None) not in mode_policy.allowed_categories:
            return ToolExposureDecision(allowed=False, reason="disallowed_category", policy=policy)

        if mode_policy.block_high_risk and policy.risk_level == "high" and tool_name not in explicit_allowed:
            return ToolExposureDecision(allowed=False, reason="high_risk_blocked", policy=policy)

        legacy_decision = cls._matches_legacy_permission(agent, tool_name)
        if legacy_decision is False:
            return ToolExposureDecision(allowed=False, reason="legacy_permission_deny", policy=policy)
        if legacy_decision is True:
            return ToolExposureDecision(allowed=True, reason="legacy_permission_allow", policy=policy)

        return ToolExposureDecision(allowed=True, reason="allowed", policy=policy)

    @classmethod
    def evaluate_exposure(
        cls,
        agent: Any,
        tool_info: Any,
        *,
        trusted: bool,
        discovered_tools: Optional[Set[str]] = None,
        tool_policy: Optional[ToolPolicyInfo] = None,
    ) -> ToolExposureDecision:
        decision = cls.evaluate_agent_gate(agent, tool_info, tool_policy=tool_policy)
        if not decision.allowed:
            return decision

        policy = decision.policy
        if policy.requires_trust and not trusted:
            return ToolExposureDecision(allowed=False, reason="workspace_trust", policy=policy)

        discovered = discovered_tools or set()
        if policy.should_defer and getattr(tool_info, "name", "") not in discovered and not policy.always_load:
            return ToolExposureDecision(allowed=False, reason="deferred_until_discovered", policy=policy)

        return ToolExposureDecision(allowed=True, reason="allowed", policy=policy)

    @classmethod
    def evaluate_execution(
        cls,
        tool_name: str,
        *,
        trusted: bool,
        tool_info: Optional[Any] = None,
    ) -> ToolExposureDecision:
        policy = get_tool_policy(tool_name, tool_info)
        if policy.requires_trust and not trusted:
            return ToolExposureDecision(allowed=False, reason="workspace_trust", policy=policy)
        return ToolExposureDecision(allowed=True, reason="allowed", policy=policy)
