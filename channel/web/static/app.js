"use strict";

const state = {
  csrfToken: "",
  sessions: [],
  sessionId: null,
  activeRunId: null,
  activeRunSessionId: null,
  eventSource: null,
  streaming: null,
  showThinking: false,
  executionMode: "auto",
};

const elements = {
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  sidebarToggleMobile: document.getElementById("sidebar-toggle-mobile"),
  newSession: document.getElementById("new-session"),
  sessionList: document.getElementById("session-list"),
  sessionTitle: document.getElementById("session-title"),
  renameSession: document.getElementById("rename-session"),
  deleteSession: document.getElementById("delete-session"),
  thinkingToggle: document.getElementById("thinking-toggle"),
  executionMode: document.getElementById("execution-mode"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  messageInput: document.getElementById("message-input"),
  sendMessage: document.getElementById("send-message"),
  stopRun: document.getElementById("stop-run"),
  statusBanner: document.getElementById("status-banner"),
};

async function apiRequest(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  if (["POST", "PATCH", "DELETE"].includes(method)) {
    headers.set("x-csrf-token", state.csrfToken);
  }
  if (options.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const response = await fetch(path, { ...options, method, headers });
  if (response.status === 204) {
    return null;
  }
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error?.message || "请求失败。请稍后重试。");
  }
  return payload;
}

function showStatus(message, isError = false) {
  elements.statusBanner.textContent = message;
  elements.statusBanner.classList.toggle("error", isError);
  elements.statusBanner.hidden = !message;
}

async function loadAuthStatus() {
  const status = await apiRequest("/api/auth/status");
  state.csrfToken = status.csrf_token;
  const saved = localStorage.getItem("myagentShowThinking");
  const savedMode = localStorage.getItem("myagentExecutionMode");
  state.showThinking = saved === null ? status.show_thinking_default : saved === "true";
  state.executionMode = ["auto", "lightweight", "full"].includes(savedMode) ? savedMode : "auto";
  elements.thinkingToggle.checked = state.showThinking;
  elements.executionMode.value = state.executionMode;
}

async function loadSessions() {
  const payload = await apiRequest("/api/sessions");
  state.sessions = payload.sessions;
  if (!state.sessions.length) {
    await createSession();
    return;
  }
  if (!state.sessions.some((session) => session.id === state.sessionId)) {
    await selectSession(state.sessions[0].id);
    return;
  }
  renderSessionList();
}

function renderSessionList() {
  elements.sessionList.replaceChildren();
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.textContent = session.title;
    button.classList.toggle("active", session.id === state.sessionId);
    button.addEventListener("click", () => selectSession(session.id));
    elements.sessionList.append(button);
  }
  const selected = state.sessions.find((session) => session.id === state.sessionId);
  elements.sessionTitle.textContent = selected?.title || "新对话";
}

async function createSession() {
  const session = await apiRequest("/api/sessions", { method: "POST" });
  state.sessions.unshift(session);
  await selectSession(session.id);
  elements.messageInput.focus();
}

async function selectSession(sessionId) {
  state.sessionId = sessionId;
  renderSessionList();
  await renderMessages();
  elements.sidebar.classList.remove("open");
}

function appendToolEvent(container, event) {
  const line = document.createElement("pre");
  line.className = "tool-event";
  const detail = event.type === "tool_start"
    ? `开始工具：${event.name || "unknown"}\n${JSON.stringify(event.args || {}, null, 2)}`
    : `结束工具：${event.name || "unknown"}\n${event.success === false ? "执行失败\n" : ""}${String(event.output || "")}`;
  line.textContent = detail;
  container.append(line);
}

function createMessageElement(message) {
  const article = document.createElement("article");
  article.className = `message ${message.kind || "assistant"} ${message.status || "complete"}`;
  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = message.kind === "user" ? "你" : "MyAgent";
  if (message.status && message.status !== "complete") {
    const status = document.createElement("span");
    status.className = "message-status";
    status.textContent = message.status === "stopped" ? "已停止" : message.status === "error" ? "生成失败" : "生成中";
    header.append(status);
  }
  const content = document.createElement("pre");
  content.className = "message-content";
  content.textContent = message.content || "";
  article.append(header, content);
  if (message.thinking) {
    const thinking = document.createElement("details");
    thinking.open = state.showThinking;
    const summary = document.createElement("summary");
    summary.textContent = "模型思考";
    const body = document.createElement("pre");
    body.className = "thinking-content";
    body.textContent = message.thinking;
    thinking.append(summary, body);
    article.append(thinking);
  }
  if (message.execution_events?.length) {
    const tools = document.createElement("details");
    tools.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "执行详情";
    tools.append(summary);
    for (const event of message.execution_events) {
      appendToolEvent(tools, event);
    }
    article.append(tools);
  }
  return article;
}

async function renderMessages() {
  if (!state.sessionId) {
    return;
  }
  const payload = await apiRequest(`/api/sessions/${state.sessionId}/messages`);
  elements.messages.replaceChildren();
  if (!payload.messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "从这里开始和 MyAgent 对话。";
    elements.messages.append(empty);
  } else {
    for (const message of payload.messages) {
      elements.messages.append(createMessageElement(message));
    }
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function renameSelectedSession() {
  const current = state.sessions.find((session) => session.id === state.sessionId);
  if (!current) {
    return;
  }
  const title = window.prompt("请输入新的会话标题：", current.title);
  if (title === null) {
    return;
  }
  const updated = await apiRequest(`/api/sessions/${state.sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
  state.sessions = state.sessions.map((session) => session.id === updated.id ? updated : session);
  renderSessionList();
}

async function deleteSelectedSession() {
  const current = state.sessions.find((session) => session.id === state.sessionId);
  if (!current || !window.confirm(`确定删除“${current.title}”及其全部消息吗？`)) {
    return;
  }
  await apiRequest(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
  state.sessions = state.sessions.filter((session) => session.id !== state.sessionId);
  state.sessionId = null;
  await loadSessions();
}

function setGenerating(active) {
  elements.messageInput.disabled = active;
  elements.sendMessage.disabled = active;
  elements.stopRun.hidden = !active;
  if (!active) {
    elements.messageInput.focus();
  }
}

function startStreamingMessage() {
  const message = createMessageElement({ kind: "assistant", status: "generating" });
  elements.messages.querySelector(".empty-state")?.remove();
  elements.messages.append(message);
  state.streaming = {
    article: message,
    content: message.querySelector(".message-content"),
    header: message.querySelector(".message-header"),
    thinking: "",
    thinkingPanel: null,
    toolsPanel: null,
  };
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function sendMessage() {
  const message = elements.messageInput.value.trim();
  if (!message || state.activeRunId) {
    return;
  }
  if (!state.sessionId) {
    await createSession();
  }
  showStatus("");
  const run = await apiRequest(`/api/sessions/${state.sessionId}/runs`, {
    method: "POST",
    body: JSON.stringify({ message, mode: state.executionMode }),
  });
  elements.messageInput.value = "";
  elements.messages.querySelector(".empty-state")?.remove();
  elements.messages.append(createMessageElement({ kind: "user", content: message, status: "complete" }));
  startStreamingMessage();
  state.activeRunId = run.run_id;
  state.activeRunSessionId = state.sessionId;
  sessionStorage.setItem("myagentActiveRunId", state.activeRunId);
  sessionStorage.setItem("myagentActiveRunSessionId", state.activeRunSessionId);
  setGenerating(true);
  connectRun(state.activeRunId);
}

function connectRun(runId) {
  state.eventSource?.close();
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.eventSource = source;
  for (const type of ["text", "thinking", "tool_start", "tool_end", "done", "error", "stopped"]) {
    source.addEventListener(type, (event) => handleRunEvent(type, event));
  }
  source.addEventListener("heartbeat", () => showStatus(""));
  source.onerror = () => {
    if (state.activeRunId === runId) {
      showStatus("实时连接暂时中断，浏览器正在尝试重连。", true);
    }
  };
}

async function handleRunEvent(type, event) {
  const data = event.data ? JSON.parse(event.data) : {};
  if (!state.streaming) {
    startStreamingMessage();
  }
  if (type === "text") {
    state.streaming.content.textContent += data.content || "";
  } else if (type === "thinking") {
    state.streaming.thinking += data.content || "";
    if (!state.streaming.thinkingPanel) {
      const panel = document.createElement("details");
      panel.open = state.showThinking;
      const summary = document.createElement("summary");
      summary.textContent = "模型思考";
      const body = document.createElement("pre");
      body.className = "thinking-content";
      panel.append(summary, body);
      state.streaming.article.append(panel);
      state.streaming.thinkingPanel = body;
    }
    state.streaming.thinkingPanel.textContent = state.streaming.thinking;
  } else if (type === "tool_start" || type === "tool_end") {
    if (!state.streaming.toolsPanel) {
      const panel = document.createElement("details");
      panel.open = true;
      const summary = document.createElement("summary");
      summary.textContent = "执行详情";
      panel.append(summary);
      state.streaming.article.append(panel);
      state.streaming.toolsPanel = panel;
    }
    appendToolEvent(state.streaming.toolsPanel, { type, ...data });
  } else {
    const status = type === "done" ? "complete" : type;
    state.streaming.article.classList.remove("generating");
    state.streaming.article.classList.add(status);
    const label = state.streaming.header.querySelector(".message-status");
    if (label) {
      label.textContent = type === "done" ? "" : type === "stopped" ? "已停止" : "生成失败";
    }
    state.eventSource?.close();
    state.eventSource = null;
    state.streaming = null;
    state.activeRunId = null;
    state.activeRunSessionId = null;
    sessionStorage.removeItem("myagentActiveRunId");
    sessionStorage.removeItem("myagentActiveRunSessionId");
    setGenerating(false);
    showStatus(type === "error" ? data.message || "生成失败，请稍后重试。" : "", type === "error");
    await loadSessions();
    if (state.sessionId) {
      await renderMessages();
    }
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function stopRun() {
  if (!state.activeRunId) {
    return;
  }
  await apiRequest(`/api/runs/${state.activeRunId}/stop`, { method: "POST" });
  showStatus("已请求停止生成，正在等待当前步骤结束。");
}

function setThinkingVisibility() {
  state.showThinking = elements.thinkingToggle.checked;
  localStorage.setItem("myagentShowThinking", String(state.showThinking));
  for (const panel of elements.messages.querySelectorAll("details")) {
    if (panel.querySelector(".thinking-content")) {
      panel.open = state.showThinking;
    }
  }
}

function setExecutionMode() {
  state.executionMode = elements.executionMode.value;
  localStorage.setItem("myagentExecutionMode", state.executionMode);
}

async function bootstrap() {
  try {
    await loadAuthStatus();
    await loadSessions();
    const savedRunId = sessionStorage.getItem("myagentActiveRunId");
    const savedSessionId = sessionStorage.getItem("myagentActiveRunSessionId");
    if (savedRunId && savedSessionId) {
      await selectSession(savedSessionId);
      state.activeRunId = savedRunId;
      state.activeRunSessionId = savedSessionId;
      setGenerating(true);
      startStreamingMessage();
      connectRun(savedRunId);
    }
  } catch (error) {
    showStatus(error.message, true);
  }
}

elements.newSession.addEventListener("click", () => createSession().catch((error) => showStatus(error.message, true)));
elements.renameSession.addEventListener("click", () => renameSelectedSession().catch((error) => showStatus(error.message, true)));
elements.deleteSession.addEventListener("click", () => deleteSelectedSession().catch((error) => showStatus(error.message, true)));
elements.stopRun.addEventListener("click", () => stopRun().catch((error) => showStatus(error.message, true)));
elements.thinkingToggle.addEventListener("change", setThinkingVisibility);
elements.executionMode.addEventListener("change", setExecutionMode);
elements.sidebarToggle.addEventListener("click", () => elements.sidebar.classList.remove("open"));
elements.sidebarToggleMobile.addEventListener("click", () => elements.sidebar.classList.add("open"));
elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage().catch((error) => showStatus(error.message, true));
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

bootstrap();
