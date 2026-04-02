from flocks.agent.agent import AgentInfo
from flocks.tool.agent_mode_policy import allows_tool
from flocks.tool.policy import get_tool_policy
from flocks.tool.registry import ToolCategory, ToolInfo


def _tool(name: str, category: ToolCategory = ToolCategory.FILE) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=f"{name} description",
        category=category,
        native=True,
        enabled=True,
    )


def test_allowed_tools_forms_whitelist() -> None:
    agent = AgentInfo(name="custom", mode="primary", allowed_tools=["read"])

    assert allows_tool(agent, _tool("read"), get_tool_policy("read")) is True
    assert allows_tool(agent, _tool("grep", ToolCategory.SEARCH), get_tool_policy("grep")) is False


def test_disallowed_tools_override_allowance() -> None:
    agent = AgentInfo(name="custom", mode="primary", disallowed_tools=["bash"])

    assert allows_tool(agent, _tool("bash", ToolCategory.CODE), get_tool_policy("bash")) is False


def test_subagent_blocks_high_risk_tools_by_default() -> None:
    agent = AgentInfo(name="helper", mode="subagent")
    bash_tool = _tool("bash", ToolCategory.CODE)

    assert allows_tool(agent, bash_tool, get_tool_policy("bash", bash_tool)) is False
    assert allows_tool(agent, _tool("read"), get_tool_policy("read")) is True
