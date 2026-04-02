"""
Backward-compatible wrappers for unified tool governance.
"""

from __future__ import annotations

from typing import Any, Set

from flocks.tool.governance import ModePolicyInfo, ToolGovernanceService
from flocks.tool.policy import ToolPolicyInfo
from flocks.tool.registry import ToolCategory


def get_mode_policy(mode: str) -> ModePolicyInfo:
    return ToolGovernanceService.get_mode_policy(mode)


def get_preferred_tool_names(agent: Any) -> Set[str]:
    return ToolGovernanceService.get_preferred_tool_names(agent)


def get_preferred_categories(agent: Any) -> Set[ToolCategory]:
    return ToolGovernanceService.get_preferred_categories(agent)


def allows_tool(agent: Any, tool_info: Any, tool_policy: ToolPolicyInfo) -> bool:
    return ToolGovernanceService.evaluate_agent_gate(
        agent,
        tool_info,
        tool_policy=tool_policy,
    ).allowed
