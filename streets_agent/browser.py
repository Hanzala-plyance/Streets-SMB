from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass

from browserbase import Browserbase
from playwright.async_api import Browser, Page, Playwright, async_playwright


@dataclass(slots=True)
class BrowserSessionInfo:
    session_id: str
    connect_url: str
    live_view_url: str


@dataclass(slots=True)
class BrowserRuntime:
    playwright: Playwright
    browser: Browser
    page: Page


_runtimes: dict[str, BrowserRuntime] = {}
_connect_urls: dict[str, str] = {}
_runtime_lock = asyncio.Lock()


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required. Add it to your .env file.")
    return value


def assert_browserbase_config() -> None:
    required_env("BROWSERBASE_API_KEY")
    required_env("BROWSERBASE_PROJECT_ID")


def _client() -> Browserbase:
    return Browserbase(api_key=required_env("BROWSERBASE_API_KEY"))


def create_browserbase_session() -> BrowserSessionInfo:
    """Create a keep-alive Browserbase session and fetch its embeddable live view."""
    assert_browserbase_config()
    bb = _client()
    session = bb.sessions.create(
        project_id=required_env("BROWSERBASE_PROJECT_ID"),
        keep_alive=True,
        api_timeout=300,
        browser_settings={
            "viewport": {"width": 1440, "height": 900},
            "blockAds": True,
            "solveCaptchas": True,
        },
        user_metadata={"app": "streets-python-langgraph-agent"},
    )
    debug = bb.sessions.debug(session.id)
    info = BrowserSessionInfo(
        session_id=session.id,
        connect_url=session.connect_url,
        live_view_url=debug.debugger_fullscreen_url,
    )
    _connect_urls[info.session_id] = info.connect_url
    return info


def _retrieve_connect_url(session_id: str) -> str:
    session = _client().sessions.retrieve(session_id)
    return session.connect_url


async def get_page(session_id: str, current_url: str | None = None) -> Page:
    """Return one live Playwright page per session, reconnecting after process-local loss."""
    async with _runtime_lock:
        existing = _runtimes.get(session_id)
        if existing and existing.browser.is_connected():
            pages = [page for context in existing.browser.contexts for page in context.pages if not page.is_closed()]
            if pages:
                if current_url:
                    matching = next((page for page in reversed(pages) if page.url == current_url), None)
                    existing.page = matching or pages[-1]
                else:
                    existing.page = pages[-1]
                return existing.page

        connect_url = _connect_urls.get(session_id)
        if not connect_url:
            connect_url = await asyncio.to_thread(_retrieve_connect_url, session_id)
            _connect_urls[session_id] = connect_url

        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.connect_over_cdp(connect_url)
            if not browser.contexts:
                raise RuntimeError("Browserbase returned a browser without a default context")
            context = browser.contexts[0]
            pages = [page for page in context.pages if not page.is_closed()]
            page = pages[-1] if pages else await context.new_page()
            runtime = BrowserRuntime(playwright=playwright, browser=browser, page=page)
            _runtimes[session_id] = runtime
            return page
        except Exception:
            await playwright.stop()
            raise


def _release_remote_session(session_id: str) -> None:
    # Closing CDP normally ends the session; this explicit release is
    # best-effort for keep-alive sessions.
    with suppress(Exception):
        _client().sessions.update(session_id, status="REQUEST_RELEASE")


async def release_session(session_id: str | None) -> None:
    if not session_id:
        return
    async with _runtime_lock:
        runtime = _runtimes.pop(session_id, None)
        _connect_urls.pop(session_id, None)
    if runtime:
        with suppress(Exception):
            await runtime.browser.close()
        with suppress(Exception):
            await runtime.playwright.stop()
    await asyncio.to_thread(_release_remote_session, session_id)
