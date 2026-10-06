"""Bounded, observable recovery for website access failures."""

import re
from urllib.parse import urlparse

from streets_agent.models import AgentAction, AgentState, PageObservation

# Official entry points, not guessed checkout URLs. Additional sites can be
# supplied in the instruction or in the Website URL input popup.
DOMINOS_ENTRY_POINTS = ("https://www.dominos.com/", "https://pizza.dominos.com/")
ACCESS_FAILURE = re.compile(
    r"unable to access|cannot access|can't access|could not (?:access|load|open)|"
    r"unable to (?:load|open|interact)|not (?:loaded|loading|available|accessible)|"
    r"blocked|captcha|verification|no (?:interactive elements|result links)|"
    r"ordering interface|browser.tool loop|step limit|tool.step limit",
    re.IGNORECASE,
)


def is_search_page(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"google.com", "www.google.com", "bing.com", "www.bing.com", "duckduckgo.com"}


def destination_urls(state: AgentState) -> list[str]:
    instruction = state.get("instruction", "")
    sources = [instruction]
    sources.extend(answer["value"] for answer in state.get("human_answers", []) if answer.get("field") == "Website URL")
    urls = []
    for source in sources:
        for raw in re.findall(r"https?://[^\s<>]+", source):
            url = raw.rstrip(".,)'\"")
            parsed = urlparse(url)
            if parsed.hostname and not parsed.username and not is_search_page(url) and url not in urls:
                urls.append(url)
    if re.search(r"\bdomino[’']?s\b", instruction, re.IGNORECASE):
        urls.extend(url for url in DOMINOS_ENTRY_POINTS if url not in urls)
    return urls


def recover_access(action: AgentAction, state: AgentState, observation: PageObservation) -> AgentAction:
    """Try known destinations once, then ask for help without ending the session."""
    attempted = state.get("attempted_urls", [])
    candidates = destination_urls(state)
    ordering = bool(re.search(r"\b(order|delivery|deliver|buy|purchase)\b", state.get("instruction", ""), re.I))
    early_search = action.action == "search" and ordering and is_search_page(observation.url) and candidates
    failure = action.action == "stop" and bool(ACCESS_FAILURE.search(f"{action.reasoning} {action.summary or ''}"))
    premature_finish = action.action == "finish" and ordering and is_search_page(observation.url)
    if not (early_search or failure or premature_finish):
        return action
    for url in candidates:
        if url.rstrip("/") != observation.url.rstrip("/") and url not in attempted:
            return AgentAction(
                reasoning="The ordering website has not been reached. Trying its direct entry point.",
                action="navigate",
                value=url,
            )
    if early_search:
        return action
    if not candidates and is_search_page(observation.url):
        return AgentAction(
            reasoning="Search did not expose a usable destination. A website URL will let me continue directly.",
            action="ask_user",
            field_label="Website URL",
            input_question="Which website should I use? Paste its full https:// URL so I can open it directly.",
        )
    return AgentAction(
        reasoning="Automatic access recovery has stalled. Keeping this browser open for your help.",
        action="browser_help",
        input_question=(
            "Use the live browser to finish any verification or open the page you want me to use. "
            "Then choose Resume agent. Your previous answers are saved; this does not approve an order."
        ),
    )
