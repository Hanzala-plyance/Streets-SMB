const $ = (selector) => document.querySelector(selector);
const startButton = $("#start");
const confirmButton = $("#confirm");
const cancelButton = $("#cancel");
const instructionInput = $("#instruction");
const fieldValueInput = $("#field-value");
const fieldChoiceInput = $("#field-choice");
const iframe = $("#live-view");
const steps = $("#steps");
let runId = null;
let socket = null;
let stepCount = 0;
let currentHitlKind = null;

let configuredTargetUrl = "";

fetch("/config")
  .then((response) => response.json())
  .then(({ targetUrl }) => {
    configuredTargetUrl = targetUrl;
    $("#target-url option").textContent = `Configured: ${targetUrl}`;
    $("#target-url").parentElement.title = targetUrl;
  })
  .catch(() => undefined);

function setState(state, label) {
  document.body.dataset.state = state;
  $("#status").textContent = label;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("visible");
  window.setTimeout(() => element.classList.remove("visible"), 4500);
}

function reset() {
  stepCount = 0;
  steps.innerHTML = '<li class="empty">Connecting to the hosted browser…</li>';
  $("#step-count").textContent = "Starting";
  $("#pause-bar").classList.remove("visible");
  $("#pause-bar").classList.remove("browser-help");
  currentHitlKind = null;
  $("#result").classList.remove("visible");
  $("#ended").classList.remove("visible");
  confirmButton.disabled = true;
  cancelButton.disabled = true;
  iframe.classList.remove("active");
  iframe.removeAttribute("src");
  $("#browser-empty").style.display = "grid";
  $("#address").textContent = "Creating Browserbase session…";
  setState("starting", "Starting");
}

function addStep(label, value) {
  steps.querySelector(".inflight")?.remove();
  if (steps.querySelector(".empty")) steps.innerHTML = "";
  const item = document.createElement("li");
  item.className = "step";
  const copy = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = label;
  copy.append(strong);
  if (value) {
    const small = document.createElement("small");
    small.textContent = value;
    copy.append(small);
  }
  item.append(document.createElement("span"), copy);
  steps.append(item);
  steps.scrollTop = steps.scrollHeight;
  stepCount += 1;
  $("#step-count").textContent = `${stepCount} step${stepCount === 1 ? "" : "s"}`;
}

function showInflight(label, value) {
  steps.querySelector(".inflight")?.remove();
  if (steps.querySelector(".empty")) steps.innerHTML = "";
  const item = document.createElement("li");
  item.className = "step inflight";
  const copy = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = label;
  copy.append(strong);
  if (value) {
    const small = document.createElement("small");
    small.textContent = value;
    copy.append(small);
  }
  item.append(document.createElement("span"), copy);
  steps.append(item);
  steps.scrollTop = steps.scrollHeight;
}

function showSession(url) {
  if (iframe.src !== url) iframe.src = url;
  iframe.classList.add("active");
  $("#browser-empty").style.display = "none";
  $("#address").textContent = url;
}

function renderResult(event) {
  $("#result").classList.add("visible");
  $("#result-title").textContent = `Finished at ${new Date(event.ts).toLocaleTimeString()}`;
  $("#readback").innerHTML = "";
  for (const field of event.fields) {
    const row = document.createElement("tr");
    if (field.draftValue !== field.finalValue) row.className = "changed";
    for (const value of [field.name, field.draftValue, field.finalValue]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    $("#readback").append(row);
  }
  $("#echo").textContent = typeof event.serverReceived === "string" ? event.serverReceived : JSON.stringify(event.serverReceived, null, 2);
  $("#run-log").textContent = JSON.stringify(event.log, null, 2);
}

function onEvent(event) {
  if (event.type === "session") {
    showSession(event.liveViewUrl);
    setState("filling", "Agent running");
  } else if (event.type === "step-start") {
    showInflight(event.label, event.draftValue);
    if (event.currentUrl) $("#address").textContent = event.currentUrl;
    setState("filling", "Agent running");
  } else if (event.type === "step") {
    addStep(event.label, event.draftValue);
    if (event.currentUrl) $("#address").textContent = event.currentUrl;
    setState("filling", "Agent running");
  } else if (event.type === "hitl-request") {
    currentHitlKind = event.kind;
    const browserHelp = event.kind === "browser_help";
    $("#pause-bar").classList.toggle("browser-help", browserHelp);
    $("#pause-bar").setAttribute("aria-modal", browserHelp ? "false" : "true");
    addStep(browserHelp ? "Waiting for browser assistance" : event.kind === "input" ? `Waiting for: ${event.fieldLabel || "information"}` : "Waiting for approval", event.actionLabel);
    $("#hitl-eyebrow").textContent = browserHelp ? "Browser needs your help" : event.kind === "input" ? "Agent needs your input" : "Human approval required";
    $("#hitl-title").textContent = event.title;
    $("#hitl-message").textContent = event.message;
    $("#field-wrap").hidden = event.kind !== "input";
    $("#field-label").textContent = event.fieldLabel || "Value";
    fieldValueInput.type = event.sensitive ? "password" : "text";
    fieldValueInput.value = event.proposedValue ?? "";
    fieldChoiceInput.innerHTML = "";
    const hasOptions = event.kind === "input" && Array.isArray(event.options) && event.options.length > 0;
    fieldValueInput.hidden = hasOptions;
    fieldChoiceInput.hidden = !hasOptions;
    if (hasOptions) {
      for (const option of event.options) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        item.selected = option.value === event.proposedValue;
        fieldChoiceInput.append(item);
      }
    }
    confirmButton.textContent = browserHelp ? "Resume agent" : event.kind === "input" ? "Use this value" : "Approve action";
    cancelButton.textContent = browserHelp || event.kind === "input" ? "Stop run" : "Do not do it";
    $("#pause-bar").classList.add("visible");
    confirmButton.disabled = false;
    cancelButton.disabled = false;
    setState("paused", browserHelp ? "Needs browser help" : event.kind === "input" ? "Needs input" : "Needs approval");
    if (event.kind === "input") window.setTimeout(() => (hasOptions ? fieldChoiceInput : fieldValueInput).focus(), 50);
  } else if (event.type === "submitted") {
    addStep("Finished", event.resultUrl);
    $("#address").textContent = event.resultUrl;
    $("#pause-bar").classList.remove("visible");
    renderResult(event);
    setState("done", "Done");
    startButton.disabled = false;
    startButton.textContent = "Run again";
  } else if (event.type === "cancelled") {
    addStep("Run stopped by you");
    $("#pause-bar").classList.remove("visible");
    setState("idle", "Cancelled");
    startButton.disabled = false;
    startButton.textContent = "Start again";
  } else if (event.type === "error") {
    addStep("Run stopped", event.message);
    $("#pause-bar").classList.remove("visible");
    setState("idle", "Error");
    startButton.disabled = false;
    startButton.textContent = "Try again";
    toast(event.message);
  }
}

function connectWebSocket(id) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws?runId=${encodeURIComponent(id)}`);
  socket.addEventListener("message", ({ data }) => {
    try { onEvent(JSON.parse(data)); } catch { toast("Received an unreadable run event."); }
  });
  socket.addEventListener("error", () => toast("The run event connection was interrupted."));
}

startButton.addEventListener("click", async () => {
  const instruction = instructionInput.value.trim();
  if (!instruction) return toast("Tell the agent what to do first.");
  socket?.close();
  reset();
  startButton.disabled = true;
  startButton.textContent = "Starting…";
  try {
    const selectedStartUrl = $("#target-url").value;
    const response = await fetch("/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ instruction, start_url: selectedStartUrl || configuredTargetUrl }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || "Could not start the run");
    runId = payload.runId;
    showSession(payload.liveViewUrl);
    connectWebSocket(runId);
  } catch (error) {
    setState("idle", "Error");
    startButton.disabled = false;
    startButton.textContent = "Try again";
    toast(error instanceof Error ? error.message : String(error));
  }
});

confirmButton.addEventListener("click", () => {
  if (!runId || socket?.readyState !== WebSocket.OPEN) return toast("The run is not connected yet.");
  confirmButton.disabled = true;
  cancelButton.disabled = true;
  $("#pause-bar").classList.remove("visible");
  const value = fieldChoiceInput.hidden ? fieldValueInput.value : fieldChoiceInput.value;
  socket.send(JSON.stringify({ type: "confirm", runId, value: currentHitlKind === "input" ? value : undefined }));
  currentHitlKind = null;
  setState("filling", "Agent running");
});

cancelButton.addEventListener("click", () => {
  if (!runId || socket?.readyState !== WebSocket.OPEN) return toast("The run is not connected yet.");
  confirmButton.disabled = true;
  cancelButton.disabled = true;
  $("#pause-bar").classList.remove("visible");
  socket.send(JSON.stringify({ type: "cancel", runId }));
});

fieldValueInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && currentHitlKind === "input" && !confirmButton.disabled) confirmButton.click();
});

$("#target-url").addEventListener("change", ({ currentTarget }) => {
  if (currentTarget.value === "demo://pizza-form") {
    instructionInput.value = "Fill out this pizza order form. Ask me for every missing detail, then ask for confirmation before submitting it.";
  }
});

window.addEventListener("message", (event) => {
  if (event.data === "browserbase-disconnected") $("#ended").classList.add("visible");
});
