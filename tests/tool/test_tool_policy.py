from flocks.tool.policy import apply_policy_defaults, get_tool_policy, is_high_risk
from flocks.tool.registry import ToolCategory, ToolInfo, ToolRegistry


def test_apply_policy_defaults_for_core_tool() -> None:
    info = ToolInfo(
        name="read",
        description="Read file contents",
        category=ToolCategory.FILE,
        native=True,
    )

    enriched = apply_policy_defaults(info)

    assert enriched.should_defer is False
    assert enriched.always_load is False
    assert enriched.permission_key == "read"
    assert "file-inspection" in enriched.tags


def test_policy_uses_real_registered_read_tool_name() -> None:
    policy = get_tool_policy("read")

    assert policy.permission_key == "read"
    assert "code-reading" in policy.tags


def test_registry_uses_read_not_read_file() -> None:
    tool_ids = set(ToolRegistry.all_tool_ids())

    assert "read" in tool_ids
    assert "read_file" not in tool_ids


def test_policy_marks_high_risk_tools_as_trusted_only() -> None:
    info = ToolInfo(
        name="bash",
        description="Run shell command",
        category=ToolCategory.CODE,
        native=True,
    )

    policy = get_tool_policy("bash", info)

    assert policy.risk_level == "high"
    assert policy.requires_trust is True
    assert policy.permission_key == "bash"
    assert is_high_risk("bash", info) is True


def test_explicit_tags_are_merged_with_defaults() -> None:
    info = ToolInfo(
        name="websearch",
        description="Search the web",
        category=ToolCategory.BROWSER,
        native=True,
        tags=["research"],
    )

    enriched = apply_policy_defaults(info)

    assert "research" in enriched.tags
    assert "web" in enriched.tags
