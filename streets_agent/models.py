from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class SelectOption(BaseModel):
    label: str
    value: str


class ObservedElement(BaseModel):
    selector: str
    tag: str = ""
    type: str = ""
    name: str = ""
    label: str = ""
    href: str = ""
    value: str = ""
    role: str = ""
    checked: bool = False
    disabled: bool = False
    required: bool = False
    options: list[SelectOption] = Field(default_factory=list)


class PageObservation(BaseModel):
    url: str
    title: str = ""
    page_text: str = ""
    elements: list[ObservedElement] = Field(default_factory=list)
    demo_receipt: bool = False


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    action: Literal[
        "navigate",
        "search",
        "click",
        "fill",
        "select",
        "ask_user",
        "browser_help",
        "press_enter",
        "wait",
        "finish",
        "stop",
    ]
    target: str | None = None
    value: str | None = None
    field_label: str | None = None
    requires_input: bool = False
    input_question: str | None = None
    confirmation_required: bool = False
    confirmation_message: str | None = None
    summary: str | None = None


class ActivityEvent(TypedDict, total=False):
    type: str
    label: str
    draftValue: str
    currentUrl: str
    ts: str
    resultUrl: str
    serverReceived: str
    fields: list[dict[str, str]]
    log: list[dict[str, str]]
    reason: str
    message: str
    kind: str
    title: str
    fieldLabel: str
    proposedValue: str
    sensitive: bool
    options: list[dict[str, str]]
    actionLabel: str


class AgentState(TypedDict, total=False):
    instruction: str
    start_url: str
    session_id: str
    current_url: str
    steps: int
    history: Annotated[list[str], operator.add]
    activity: Annotated[list[ActivityEvent], operator.add]
    fields: Annotated[list[dict[str, str]], operator.add]
    observed_elements: list[dict]
    pending_action: dict | None
    contains_human_data: bool
    done: bool
    cancelled: bool
    summary: str
    human_answers: Annotated[list[dict[str, str]], operator.add]
    attempted_urls: Annotated[list[str], operator.add]
