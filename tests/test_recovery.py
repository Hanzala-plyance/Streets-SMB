import importlib
import json
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from streets_agent.models import AgentAction, PageObservation
from streets_agent.recovery import DOMINOS_ENTRY_POINTS, recover_access

graph_module = importlib.import_module("streets_agent.graph")


def access_stop() -> AgentAction:
    return AgentAction(
        reasoning="The ordering page needs to load", action="stop", summary="Unable to access the ordering interface"
    )


def test_reported_bing_failure_navigates_to_dominos():
    result = recover_access(
        access_stop(),
        {"instruction": "Order a pizza from Domino’s for delivery"},
        PageObservation(url="https://www.bing.com/search?q=Dominos&rdr=1"),
    )
    assert result.action == "navigate"
    assert result.value == "https://www.dominos.com/"


def test_order_discovery_skips_search_when_official_destination_is_known():
    result = recover_access(
        AgentAction(reasoning="Find merchant", action="search", value="Domino's order online pizza"),
        {"instruction": "Order pizza from Domino's"},
        PageObservation(url="https://www.google.com/"),
    )
    assert result.action == "navigate"


def test_failed_entry_points_are_not_retried_indefinitely():
    result = recover_access(
        access_stop(),
        {"instruction": "Order from Dominos", "attempted_urls": list(DOMINOS_ENTRY_POINTS)},
        PageObservation(url="https://pizza.dominos.com/", page_text="Access denied"),
    )
    assert result.action == "browser_help"


def test_unknown_merchant_asks_for_url_without_guessing_one():
    result = recover_access(
        access_stop(),
        {"instruction": "Order from the bakery"},
        PageObservation(url="https://www.bing.com/search?q=bakery"),
    )
    assert result.action == "ask_user"
    assert result.field_label == "Website URL"


def test_user_supplied_url_is_used_for_recovery():
    result = recover_access(
        access_stop(),
        {
            "instruction": "Open the bakery",
            "human_answers": [{"field": "Website URL", "question": "Where?", "value": "https://bakery.example/"}],
        },
        PageObservation(url="https://www.bing.com/search?q=bakery"),
    )
    assert result.value == "https://bakery.example/"


def test_unrelated_stop_is_not_rewritten_as_browser_recovery():
    action = AgentAction(reasoning="The requested item is unavailable", action="stop")
    assert (
        recover_access(action, {"instruction": "Order from Domino's"}, PageObservation(url="https://www.dominos.com/"))
        is action
    )


@pytest.mark.asyncio
async def test_graph_preserves_input_and_session_across_access_failure_and_manual_resume(monkeypatch):
    """Exercise real graph checkpoints/interrupt replay with simulated browser/model IO."""

    class Page:
        url = "https://www.google.com/"

        async def goto(self, url, **kwargs):
            self.url = url

    page = Page()
    release = AsyncMock()
    monkeypatch.setattr(graph_module, "get_page", AsyncMock(return_value=page))
    monkeypatch.setattr(graph_module, "dismiss_popups", AsyncMock(return_value=[]))
    monkeypatch.setattr(graph_module, "screenshot_data_url", AsyncMock(return_value=None))
    monkeypatch.setattr(graph_module, "release_session", release)

    async def observe(_):
        return PageObservation(url=page.url, page_text="Ordering page")

    monkeypatch.setattr(graph_module, "observe_page", observe)
    decisions = iter(
        [
            AgentAction(
                reasoning="Address needs a city",
                action="ask_user",
                field_label="Delivery city",
                input_question="What city and ZIP code should I use?",
            ),
            access_stop(),
            access_stop(),
            access_stop(),
            AgentAction(reasoning="Ordering page is visible", action="finish", summary="Opened the ordering page."),
        ]
    )
    prompts = []

    class Model:
        async def ainvoke(self, messages):
            prompts.append(json.loads(messages[1].content[0]["text"]))
            return next(decisions)

    monkeypatch.setattr(graph_module, "get_model", lambda: Model())
    graph = graph_module.builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "merchant-recovery"}, "recursion_limit": 80}
    state = await graph.ainvoke(
        {
            "instruction": "Open Domino's ordering interface and ask me for the delivery city",
            "session_id": "test-session",
            "start_url": page.url,
        },
        config,
    )
    assert state["__interrupt__"][0].value["kind"] == "input"

    # Advance history beyond the old planner window: the answer must remain in
    # its own checkpoint state, regardless of future action/history trimming.
    await graph.aupdate_state(config, {"history": ["Old browser action"] * 15})
    state = await graph.ainvoke(Command(resume={"decision": "approve", "value": "Plano, TX 75024"}), config)
    assert state["__interrupt__"][0].value["kind"] == "browser_help"
    assert state["attempted_urls"][-2:] == list(DOMINOS_ENTRY_POINTS)
    assert state["human_answers"][0]["value"] == "Plano, TX 75024"
    assert prompts[-1]["saved_user_answers"][0]["value"] == "Plano, TX 75024"
    release.assert_not_awaited()

    # Manual navigation occurs while the graph is interrupted. Resume reads
    # the live page instead of reopening a stale URL or approving any purchase.
    page.url = "https://www.dominos.com/en/restaurants"
    state = await graph.ainvoke(Command(resume={"decision": "continue"}), config)
    assert state["done"] is True
    assert state["current_url"] == page.url
    assert "no order approval was given" in " ".join(state["history"])
    assert state["human_answers"][0]["value"] == "Plano, TX 75024"
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_browser_help_releases_session_without_browser_action(monkeypatch):
    release = AsyncMock()
    page_lookup = AsyncMock()
    monkeypatch.setattr(graph_module, "release_session", release)
    monkeypatch.setattr(graph_module, "get_page", page_lookup)
    monkeypatch.setattr(graph_module, "interrupt", lambda _: {"decision": "reject"})
    result = await graph_module.browser_help_node(
        {
            "session_id": "test-session",
            "pending_action": {"action": "browser_help", "reasoning": "Blocked"},
        }
    )
    assert result.update["cancelled"] is True
    release.assert_awaited_once_with("test-session")
    page_lookup.assert_not_awaited()
