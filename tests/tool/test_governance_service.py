from flocks.agent.agent import AgentInfo
from flocks.tool.governance import ToolGovernanceService
from flocks.tool.registry import ToolCategory, ToolInfo


def _tool(name: str, category: ToolCategory = ToolCategory.FILE) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=f"{name} description",
        category=category,
        native=True,
        enabled=True,
    )


def test_evaluate_agent_gate_respects_allowlist() -> None:
    agent = AgentInfo(name="custom", mode="primary", allowed_tools=["read"])

    decision = ToolGovernanceService.evaluate_agent_gate(agent, _tool("grep", ToolCategory.SEARCH))

    assert decision.allowed is False
    assert decision.reason == "explicit_allowlist"


def test_evaluate_exposure_blocks_untrusted_high_risk_tool() -> None:
    agent = AgentInfo(name="rex", mode="primary")

    decision = ToolGovernanceService.evaluate_exposure(
        agent,
        _tool("bash", ToolCategory.CODE),
        trusted=False,
        discovered_tools=set(),
    )

    assert decision.allowed is False
    assert decision.reason == "workspace_trust"


def test_primary_mode_allows_terminal_tools_for_rex() -> None:
    agent = AgentInfo(name="rex", mode="primary")

    decision = ToolGovernanceService.evaluate_exposure(
        agent,
        _tool("bash", ToolCategory.TERMINAL),
        trusted=True,
        discovered_tools=set(),
    )

    assert decision.allowed is True
    assert decision.reason == "allowed"


def test_evaluate_exposure_hides_undiscovered_deferred_tool() -> None:
    agent = AgentInfo(name="rex", mode="primary")

    decision = ToolGovernanceService.evaluate_exposure(
        agent,
        _tool("websearch", ToolCategory.BROWSER),
        trusted=True,
        discovered_tools=set(),
    )

    assert decision.allowed is False
    assert decision.reason == "deferred_until_discovered"


def test_evaluate_exposure_allows_discovered_deferred_tool() -> None:
    agent = AgentInfo(name="rex", mode="primary")

    decision = ToolGovernanceService.evaluate_exposure(
        agent,
        _tool("websearch", ToolCategory.BROWSER),
        trusted=True,
        discovered_tools={"websearch"},
    )

    assert decision.allowed is True
    assert decision.reason == "allowed"
