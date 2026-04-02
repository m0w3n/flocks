"""
Centralized built-in tool governance policy.

This module keeps tool declaration defaults out of runner.py so request-time
selection, tool search, and permission gating can share the same policy view.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from flocks.tool.registry import ToolCategory


class ToolPolicyInfo(BaseModel):
    should_defer: bool = False
    always_load: bool = False
    risk_level: str = "low"
    requires_trust: bool = False
    permission_key: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    allowed_modes: List[str] = Field(default_factory=list)
    disallowed_agents: List[str] = Field(default_factory=list)


ALWAYS_LOAD_TOOL_NAMES: Set[str] = {
    "question",
    "tool_search",
}

CORE_TOOL_NAMES: Set[str] = {
    "read",
    "list",
    "glob",
    "grep",
    "edit",
    "multiedit",
    "write",
    "apply_patch",
    "bash",
    "question",
    "skill",
    "tool_search",
}

DEFERRED_TOOL_NAMES: Set[str] = {
    "task",
    "task_create",
    "task_list",
    "task_status",
    "task_update",
    "task_delete",
    "task_rerun",
    "run_workflow",
    "run_workflow_node",
    "run_slash_command",
    "session_list",
    "session_get",
    "session_create",
    "session_update",
    "session_delete",
    "session_archive",
    "list_providers",
    "add_provider",
    "add_model",
    "background_output",
    "background_cancel",
    "plan_enter",
    "plan_exit",
    "delegate_task",
    "call_omo_agent",
    "webfetch",
    "websearch",
    "memory",
    "model_config",
    "batch",
    "slash_command",
    "wecom_mcp",
    "channel_message",
}

HIGH_RISK_TOOL_NAMES: Set[str] = {
    "bash",
    "edit",
    "multiedit",
    "write",
    "apply_patch",
    "run_workflow",
    "run_workflow_node",
    "task",
    "task_create",
    "task_update",
    "task_delete",
    "delegate_task",
    "call_omo_agent",
    "background_output",
    "background_cancel",
    "add_provider",
    "add_model",
    "wecom_mcp",
}

PERMISSION_KEYS: Dict[str, str] = {
    "read": "read",
    "read_file": "read",
    "list": "list",
    "glob": "glob",
    "grep": "grep",
    "write": "write",
    "edit": "edit",
    "multiedit": "edit",
    "apply_patch": "apply_patch",
    "bash": "bash",
    "task": "task",
    "task_create": "task",
    "task_list": "task",
    "task_status": "task",
    "task_update": "task",
    "task_delete": "task",
    "task_rerun": "task",
    "delegate_task": "delegate_task",
    "call_omo_agent": "delegate_task",
    "run_workflow": "workflow",
    "run_workflow_node": "workflow",
    "background_output": "background",
    "background_cancel": "background",
    "webfetch": "web",
    "websearch": "web",
    "question": "question",
    "tool_search": "tool_search",
}

TOOL_TAGS: Dict[str, List[str]] = {
    "read": ["code-reading", "file-inspection"],
    "read_file": ["code-reading", "file-inspection"],
    "list": ["file-navigation", "workspace"],
    "glob": ["file-search", "workspace"],
    "grep": ["code-search", "text-search"],
    "edit": ["code-editing", "refactor"],
    "multiedit": ["code-editing", "refactor"],
    "write": ["file-creation", "code-editing"],
    "apply_patch": ["patching", "code-editing"],
    "bash": ["terminal", "command-execution"],
    "webfetch": ["web", "http-fetch"],
    "websearch": ["web", "research"],
    "delegate_task": ["agent", "delegation"],
    "call_omo_agent": ["agent", "delegation"],
    "task": ["planning", "task-management"],
    "task_create": ["planning", "task-management"],
    "task_list": ["planning", "task-management"],
    "task_status": ["planning", "task-management"],
    "task_update": ["planning", "task-management"],
    "task_delete": ["planning", "task-management"],
    "task_rerun": ["planning", "task-management"],
    "run_workflow": ["workflow", "execution"],
    "run_workflow_node": ["workflow", "execution"],
    "question": ["user-interaction", "clarification"],
    "skill": ["knowledge", "skill"],
    "tool_search": ["tool-discovery", "capability-search"],
    "session_list": ["session", "history"],
    "session_get": ["session", "history"],
    "session_create": ["session", "management"],
    "session_update": ["session", "management"],
    "session_delete": ["session", "management"],
    "session_archive": ["session", "management"],
    "background_output": ["background-task", "process"],
    "background_cancel": ["background-task", "process"],
    "memory": ["memory", "context"],
    "model_config": ["model", "configuration"],
    "batch": ["batch", "orchestration"],
    "slash_command": ["slash-command", "orchestration"],
    "plan_enter": ["planning", "mode"],
    "plan_exit": ["planning", "mode"],
    "channel_message": ["messaging", "channel"],
    "wecom_mcp": ["enterprise", "wecom"],
}


def _infer_risk_level(tool_name: str, tool_info: Optional[Any] = None) -> str:
    if tool_name in HIGH_RISK_TOOL_NAMES:
        return "high"
    if getattr(tool_info, "requires_confirmation", False):
        return "high"
    category = getattr(tool_info, "category", None)
    if category == ToolCategory.TERMINAL:
        return "high"
    if category in {ToolCategory.BROWSER, ToolCategory.SYSTEM} and tool_name in DEFERRED_TOOL_NAMES:
        return "medium"
    return "low"


def _infer_should_defer(tool_name: str, tool_info: Optional[Any] = None) -> bool:
    if tool_name in ALWAYS_LOAD_TOOL_NAMES or tool_name in CORE_TOOL_NAMES:
        return False
    if tool_name in DEFERRED_TOOL_NAMES:
        return True
    category = getattr(tool_info, "category", None)
    return category in {ToolCategory.BROWSER, ToolCategory.SYSTEM}


def _default_permission_key(tool_name: str) -> str:
    return PERMISSION_KEYS.get(tool_name, tool_name)


def get_tool_policy(tool_name: str, tool_info: Optional[Any] = None) -> ToolPolicyInfo:
    tags = list(dict.fromkeys(
        list(TOOL_TAGS.get(tool_name, [])) + list(getattr(tool_info, "tags", None) or [])
    ))
    risk_level = getattr(tool_info, "risk_level", None) or _infer_risk_level(tool_name, tool_info)
    requires_trust = getattr(tool_info, "requires_trust", None)
    if requires_trust is None:
        requires_trust = risk_level == "high"
    return ToolPolicyInfo(
        should_defer=(
            getattr(tool_info, "should_defer", None)
            if getattr(tool_info, "should_defer", None) is not None
            else _infer_should_defer(tool_name, tool_info)
        ),
        always_load=(
            getattr(tool_info, "always_load", None)
            if getattr(tool_info, "always_load", None) is not None
            else tool_name in ALWAYS_LOAD_TOOL_NAMES
        ),
        risk_level=risk_level,
        requires_trust=bool(requires_trust),
        permission_key=getattr(tool_info, "permission_key", None) or _default_permission_key(tool_name),
        tags=tags,
        allowed_modes=[],
        disallowed_agents=[],
    )


def is_high_risk(tool_name: str, tool_info: Optional[Any] = None) -> bool:
    return get_tool_policy(tool_name, tool_info).risk_level == "high"


def apply_policy_defaults(tool_info: Any) -> Any:
    policy = get_tool_policy(getattr(tool_info, "name", ""), tool_info)
    if getattr(tool_info, "should_defer", None) is None:
        tool_info.should_defer = policy.should_defer
    if getattr(tool_info, "always_load", None) is None:
        tool_info.always_load = policy.always_load
    if getattr(tool_info, "risk_level", None) is None:
        tool_info.risk_level = policy.risk_level
    if getattr(tool_info, "requires_trust", None) is None:
        tool_info.requires_trust = policy.requires_trust
    if getattr(tool_info, "permission_key", None) is None:
        tool_info.permission_key = policy.permission_key
    if not getattr(tool_info, "tags", None):
        tool_info.tags = list(policy.tags)
    else:
        tool_info.tags = list(dict.fromkeys(list(tool_info.tags) + list(policy.tags)))
    return tool_info
