# Streets Python LangGraph browser agent

An instruction-driven Python LangGraph agent that operates a hosted Chromium browser through Browserbase. The included web UI embeds Browserbase Live View, streams the graph's activity, asks for missing form values, and requires explicit approval immediately before consequential actions.

The Python graph is also exported through `langgraph.json`, so it can run in the LangGraph Agent Server and be inspected in LangSmith Studio.

## Improvements over the previous runner

- The LangGraph runtime, Browserbase integration, HITL interrupts, and web server are now Python.
- Browser actions are recoverable. A stale selector or failed click is returned to the planner for another attempt.
- Navigation accepts absolute URLs, relative URLs, and bare domains.
- The graph has a deterministic `search` tool and does not depend on the model constructing search URLs.
- Empty Reddit fallback URLs are repaired from the original instruction. For example, `site:reddit.com thai food in plano` becomes `https://www.reddit.com/search/?q=thai+food+in+plano`.
- Google automated-traffic pages are identified in the observation, and the agent is told to switch to Bing or direct Reddit search.
- Old `data-agent-id` attributes are cleared before every observation, preventing hidden stale elements from receiving clicks.
- The graph uses native `interrupt()`/`Command(resume=...)` for both missing information and final approval.
- LangSmith tracing captures graph nodes and the `ChatOpenAI` model call when enabled.

## Setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Add your Browserbase and OpenAI credentials to `.env`.

## Run the browser UI

```bash
source .venv/bin/activate
uvicorn streets_agent.server:app --reload --port 3000
```

Or use the convenience command:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), enter an instruction, and watch the Browserbase session in the iframe.

Choose **Built-in HITL form demo** in the starting-page menu to exercise the
input and approval popups. The form is rendered inside the remote Browserbase
tab, so it works without exposing your local server to the internet.

## Running an end-to-end test as a developer

`pytest` only covers pure functions (URL repair, hardening, approval checks) — it never opens a browser. To exercise the full stack (FastAPI → LangGraph → Browserbase → OpenAI → WebSocket UI), run it manually:

1. Confirm `.env` has valid `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`, and `OPENAI_API_KEY`.
2. Start the UI server: `npm run dev` (or `uvicorn streets_agent.server:app --reload --port 3000`).
3. Open [http://localhost:3000](http://localhost:3000).
4. Set the starting page dropdown to **Built-in HITL form demo** — this is the fastest repeatable e2e path since it doesn't depend on a third-party site's current markup.
5. Enter an instruction that requires a missing form field (e.g. `Fill out the pizza order form for a large pepperoni pizza`) and click **Run agent**.
6. Verify, in order:
   - The Browserbase Live View iframe loads and shows the remote browser.
   - The step ticker on the left streams each planner/browser-tool action.
   - A **HITL input popup** appears when the agent hits a field it can't infer; submit a value and confirm the run resumes.
   - A **HITL approval popup** appears before the final submit action; approving lets the run finish, rejecting cancels it.
   - The result panel shows the submitted field values and the raw event log (`<details>` at the bottom).
7. Repeat once rejecting the approval popup, to confirm the run cancels cleanly and releases the Browserbase session (check the Browserbase dashboard — the session should end shortly after).

For a second, real-site pass, point the instruction at a live target (e.g. `TARGET_URL` in `.env`) and confirm navigation, the Reddit/Google fallback behavior, and consequential-action approval described in [Improvements over the previous runner](#improvements-over-the-previous-runner).

## Run and inspect the graph in LangSmith Studio

Add a LangSmith key to `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=streets-browser-agent
```

Then start the LangGraph Agent Server:

```bash
source .venv/bin/activate
langgraph dev --port 2024
```

Or:

```bash
npm run graph
```

The command prints the local API, API docs, and LangSmith Studio URL. The graph name is `browser_agent`, as configured in [`langgraph.json`](./langgraph.json).

### Studio can't connect to the local server

If Studio shows `Failed to connect to Agent Server because the domain "unknown" is not allowed`, your browser is blocking Studio (an `https://smith.langchain.com` page) from reaching a plain `http://localhost` server. This is common in Safari, Brave, and Chrome 142+.

Restart the server with a tunnel:

```bash
source .venv/bin/activate
langgraph dev --port 2024 --tunnel
```

Then, in the terminal output, copy the `https://*.trycloudflare.com` tunnel URL and connect manually:

1. Open `https://smith.langchain.com/studio/` (no `?baseUrl=` query string).
2. Click **Connect to a local server**.
3. Paste the tunnel URL and add it to **Allowed Origins**.
4. Click **Connect**.

Studio requires this explicit confirmation step; opening a `?baseUrl=` link directly does not satisfy it. Only run one `langgraph dev` process at a time — a stale instance on the same port will cause Studio to connect to the wrong server.

Example Studio input:

```json
{
  "instruction": "Search for site:reddit.com thai food in plano and open the most relevant result",
  "start_url": "https://www.google.com"
}
```

When the graph pauses, Studio displays the interrupt payload. Resume it with a string for requested input, `true` to approve an action, or a payload such as:

```json
{
  "decision": "approve",
  "value": "Plano"
}
```

Traces appear in the `streets-browser-agent` LangSmith project. Set `LANGSMITH_TRACING=false` if you want local Studio execution without uploading traces.

## Tests and linting

```bash
source .venv/bin/activate
pytest -q
ruff check streets_agent tests
```

The tests include the exact empty-URL failure reported for the Reddit search fallback, URL normalization, unsafe protocol rejection, search URL construction, and consequential-action approval checks.

## Architecture

```text
plain-text instruction
        │
        ▼
observe page + screenshot ──► structured model decision
        ▲                              │
        │                              ▼
        └──────── browser tool ◄── LangGraph router
                         │
                         ├── missing value ──► interrupt ──► input popup
                         └── external effect ► interrupt ──► approval popup
```

- [`streets_agent/graph.py`](./streets_agent/graph.py): graph, model policy, browser tools, URL repair, and HITL.
- [`streets_agent/browser.py`](./streets_agent/browser.py): Browserbase session lifecycle and async Playwright connection.
- [`streets_agent/server.py`](./streets_agent/server.py): FastAPI UI server, WebSocket events, and graph resume bridge.
- [`public/`](./public): embedded Live View and HITL dialog.

## Security and operational notes

The Browserbase Live View URL grants control over the remote browser. This localhost demo sends that URL to the frontend; add authentication and authorization before deployment.

Common personal fields are redacted from DOM observations. After human input is entered, screenshots are no longer sent to the model for that run. A production agent should additionally use domain allowlists, durable checkpoint storage, encrypted secrets, rate limits, and a stricter domain-specific action policy.

Browserbase sessions consume account minutes. If the API returns HTTP 402, the configured account has exhausted its allowance and must wait for a reset or move to a plan with available browser minutes.
