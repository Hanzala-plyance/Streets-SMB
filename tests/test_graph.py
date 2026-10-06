from streets_agent.graph import (
    enforce_goal_progress,
    harden_action,
    human_response,
    normalize_navigation_url,
    search_url,
)
from streets_agent.models import AgentAction, ObservedElement, PageObservation


def observation(
    *elements: ObservedElement,
    url: str = "https://www.google.com/sorry/index",
    page_text: str = "automated traffic",
) -> PageObservation:
    return PageObservation(url=url, title="Page", page_text=page_text, elements=list(elements))


def test_empty_reddit_navigation_is_repaired_from_instruction() -> None:
    url = normalize_navigation_url(
        "",
        "Search Google for site:reddit.com thai food in plano",
        "Use Reddit's own search URL",
        "https://www.google.com/sorry/index",
    )
    assert url == "https://www.reddit.com/search/?q=thai+food+in+plano"


def test_harden_action_repairs_the_exact_failed_navigation() -> None:
    action = AgentAction(
        reasoning="Google blocked automated traffic, so use Reddit's own search URL",
        action="navigate",
        value="",
    )
    result = harden_action(
        action,
        observation(),
        "Search Google for site:reddit.com thai food in plano",
    )
    assert result.action == "navigate"
    assert result.value == "https://www.reddit.com/search/?q=thai+food+in+plano"


def test_relative_and_bare_domain_urls_are_normalized() -> None:
    assert normalize_navigation_url("/search?q=thai", "", "", "https://reddit.com/r/plano") == (
        "https://reddit.com/search?q=thai"
    )
    assert normalize_navigation_url("www.reddit.com/r/plano", "", "", "https://example.com") == (
        "https://www.reddit.com/r/plano"
    )


def test_unsupported_protocol_is_rejected() -> None:
    assert normalize_navigation_url("javascript:alert(1)", "", "", "https://example.com") is None


def test_search_tool_builds_site_specific_fallbacks() -> None:
    assert search_url("thai food in plano", "reddit") == "https://www.reddit.com/search/?q=thai+food+in+plano"
    assert search_url("thai food in plano", "reddit.com") == "https://www.reddit.com/search/?q=thai+food+in+plano"
    assert search_url("thai food in plano", "yelp.com").startswith("https://www.bing.com/search?q=site%3Ayelp.com+")


def test_submit_is_hardened_to_require_approval() -> None:
    button = ObservedElement(selector='[data-agent-id="agent-1"]', tag="button", type="submit", label="Submit order")
    action = AgentAction(reasoning="Submit it", action="click", target=button.selector)
    result = harden_action(action, observation(button, url="https://shop.example/checkout"), "submit the order")
    assert result.confirmation_required is True
    assert "Submit order" in (result.confirmation_message or "")


def test_search_submit_does_not_require_approval() -> None:
    button = ObservedElement(selector='[data-agent-id="agent-1"]', tag="button", type="submit", label="Google Search")
    action = AgentAction(reasoning="Search", action="click", target=button.selector)
    result = harden_action(action, observation(button, url="https://google.com"), "search for restaurants")
    assert result.confirmation_required is False


def test_explicit_site_search_overrides_model_search_engine() -> None:
    action = AgentAction(
        reasoning="Search Bing",
        action="search",
        target="bing",
        value="site:reddit.com thai food in plano",
    )
    result = harden_action(
        action,
        observation(url="https://www.google.com", page_text="Google homepage"),
        "Search for site:reddit.com thai food in plano and open the most relevant result",
    )
    assert result.target == "reddit"
    assert result.value == "thai food in plano"


def test_site_search_uses_bing_when_direct_site_has_a_human_challenge() -> None:
    action = AgentAction(reasoning="Try a fallback", action="search", target="reddit", value="thai food in plano")
    result = harden_action(
        action,
        observation(
            url="https://www.reddit.com/search/?q=thai+food+in+plano",
            page_text="You've been blocked by network security. Verify you are human.",
        ),
        "Search for site:reddit.com thai food in plano and open the most relevant result",
    )
    assert result.target == "bing"
    assert result.value == "site:reddit.com thai food in plano"


def test_submit_confirmation_question_becomes_approval_on_actual_click() -> None:
    submit = ObservedElement(selector='[data-agent-id="agent-2"]', tag="button", type="submit", label="Submit order")
    action = AgentAction(
        reasoning="The form is ready to submit",
        action="ask_user",
        input_question="Would you like me to submit the order?",
    )
    result = harden_action(
        action,
        observation(submit, url="about:blank", page_text="Pizza order"),
        "Fill the form and submit after confirmation",
    )
    assert result.action == "click"
    assert result.target == submit.selector
    assert result.confirmation_required is True


def test_submit_confirmation_waits_for_required_fields() -> None:
    delivery = ObservedElement(
        selector='[data-agent-id="agent-1"]', tag="input", type="time", label="Delivery time", required=True
    )
    submit = ObservedElement(selector='[data-agent-id="agent-2"]', tag="button", type="submit", label="Submit order")
    action = AgentAction(
        reasoning="The form is ready to submit",
        action="ask_user",
        input_question="Should I submit the order?",
    )
    result = harden_action(
        action,
        observation(delivery, submit, url="about:blank", page_text="Pizza order"),
        "Fill the form and submit after confirmation",
    )
    assert result.action == "fill"
    assert result.target == delivery.selector
    assert result.requires_input is True
    assert result.field_label == "Delivery time"


def test_direct_submit_click_waits_until_required_fields_are_complete() -> None:
    email = ObservedElement(
        selector='[data-agent-id="agent-1"]', tag="input", type="email", label="E-mail", required=True
    )
    submit = ObservedElement(selector='[data-agent-id="agent-2"]', tag="button", type="submit", label="Submit order")
    action = AgentAction(reasoning="Submit now", action="click", target=submit.selector)
    result = harden_action(
        action,
        observation(email, submit, url="about:blank", page_text="Pizza order"),
        "Fill and submit the form",
    )
    assert result.action == "fill"
    assert result.target == email.selector
    assert result.requires_input is True


def test_required_field_recovery_reuses_value_explicitly_in_instruction() -> None:
    telephone = ObservedElement(
        selector='[data-agent-id="agent-1"]', tag="input", type="tel", label="Telephone", required=True
    )
    submit = ObservedElement(selector='[data-agent-id="agent-2"]', tag="button", type="submit", label="Submit order")
    result = harden_action(
        AgentAction(reasoning="Submit", action="click", target=submit.selector),
        observation(telephone, submit, url="about:blank", page_text="Pizza order"),
        "Enter telephone 555-0100 and submit after confirmation",
    )
    assert result.action == "fill"
    assert result.target == telephone.selector
    assert result.value == "555-0100"
    assert result.requires_input is False


def test_agent_cannot_finish_while_requested_submit_is_still_visible() -> None:
    submit = ObservedElement(selector='[data-agent-id="agent-2"]', tag="button", type="submit", label="Submit order")
    result = enforce_goal_progress(
        AgentAction(reasoning="Done", action="finish"),
        {"instruction": "Fill and submit the form", "history": []},
        observation(submit, url="about:blank", page_text="Pizza order"),
    )
    assert result.action == "click"
    assert result.target == submit.selector
    assert result.confirmation_required is True


def test_demo_receipt_confirms_submission() -> None:
    result = enforce_goal_progress(
        AgentAction(reasoning="Re-observe", action="wait"),
        {
            "instruction": "Fill and submit the form",
            "history": ["Clicked Submit order"],
            "start_url": "demo://pizza-form",
        },
        PageObservation(url="about:blank", demo_receipt=True),
    )
    assert result.action == "finish"
    assert "submitted successfully" in (result.summary or "")


def test_missing_submit_button_does_not_prove_real_order_success() -> None:
    result = enforce_goal_progress(
        AgentAction(reasoning="Checkout is loading", action="wait"),
        {"instruction": "Purchase a pizza", "history": ["Clicked Checkout"]},
        observation(url="https://www.dominos.com/checkout", page_text="Loading..."),
    )
    assert result.action == "wait"


def test_duplicate_question_for_filled_field_advances_to_submit_approval() -> None:
    delivery = ObservedElement(
        selector='[data-agent-id="agent-1"]',
        tag="input",
        type="time",
        label="Delivery time",
        value="18:00",
        required=True,
    )
    submit = ObservedElement(selector='[data-agent-id="agent-2"]', tag="button", type="submit", label="Submit order")
    result = harden_action(
        AgentAction(
            reasoning="Ask for delivery time",
            action="ask_user",
            field_label="Delivery time",
            input_question="What delivery time should I enter?",
        ),
        observation(delivery, submit, url="about:blank", page_text="Pizza order"),
        "Fill and submit the order after confirmation",
    )
    assert result.action == "click"
    assert result.target == submit.selector
    assert result.confirmation_required is True


def test_human_response_supports_ui_and_studio_payloads() -> None:
    assert human_response({"decision": "approve", "value": "Plano"}) == (True, "Plano")
    assert human_response("Plano") == (True, "Plano")
    assert human_response({"decision": "reject"}) == (False, "")


def test_agent_cannot_stop_before_requested_search_runs() -> None:
    action = AgentAction(reasoning="Cannot verify results", action="stop")
    result = enforce_goal_progress(
        action,
        {
            "instruction": "Search for site:reddit.com thai food in plano and open the most relevant result",
            "history": ["Opened starting page: https://www.google.com"],
        },
        observation(url="https://www.google.com/"),
    )
    assert result.action == "search"
    assert result.target == "reddit"
    assert result.value == "thai food in plano"


def test_agent_cannot_finish_before_opening_requested_result() -> None:
    action = AgentAction(reasoning="Search complete", action="finish")
    result = enforce_goal_progress(
        action,
        {
            "instruction": "Search for Thai food and open the most relevant result",
            "history": ["Searched via www.reddit.com: Thai food"],
        },
        observation(url="https://www.reddit.com/search/?q=Thai+food"),
    )
    assert result.action == "wait"
