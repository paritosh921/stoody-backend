const STORAGE_KEY = "stoody-book-session";
const sourceText = document.querySelector("#sourceText");
const sourceStatus = document.querySelector("#sourceStatus");
const conversation = document.querySelector("#conversation");
const askForm = document.querySelector("#askForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const formStatus = document.querySelector("#formStatus");
const clearSourceButton = document.querySelector("#clearSourceButton");
const newSessionButton = document.querySelector("#newSessionButton");

let state = {
  sessionId: crypto.randomUUID(),
  source: "",
  messages: [],
};

function loadState() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (stored && typeof stored === "object") {
      state = {
        sessionId: stored.sessionId || crypto.randomUUID(),
        source: stored.source || "",
        messages: Array.isArray(stored.messages) ? stored.messages : [],
      };
    }
  } catch {
    state = { sessionId: crypto.randomUUID(), source: "", messages: [] };
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function renderConversation() {
  conversation.innerHTML = "";

  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "message assistant";
    empty.textContent =
      "Paste source material, then ask Stoody Book to explain, quiz, summarize, or guide you through it.";
    conversation.appendChild(empty);
    return;
  }

  for (const message of state.messages) {
    const node = document.createElement("div");
    node.className = `message ${message.role === "user" ? "user" : "assistant"}`;
    node.textContent = message.content;
    conversation.appendChild(node);
  }

  conversation.scrollTop = conversation.scrollHeight;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Sending" : "Send";
  formStatus.textContent = isBusy ? "Stoody Book is thinking..." : "";
}

function buildSystemPrompt() {
  const source = state.source.trim();
  return [
    "You are Stoody Book, Stoody's page-grounded learning workspace.",
    "Help the student understand the supplied material. Be concise, anchored, and instructional.",
    "When the prompt looks like homework or an exam answer request, guide the student through reasoning before giving any final answer.",
    source ? `Source material:\n${source}` : "No source material was supplied. Ask for material when grounding is needed.",
  ].join("\n\n");
}

async function sendMessage(text) {
  const userMessage = { role: "user", content: text };
  state.messages.push(userMessage);
  saveState();
  renderConversation();
  setBusy(true);

  try {
    const response = await fetch("/api/v1/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        sessionId: state.sessionId,
        userId: "stoody-book-web",
        mode: "stoody_book",
        subject: "learning",
        conversationHistory: state.messages.slice(-10),
        systemPrompt: buildSystemPrompt(),
      }),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload?.success === false) {
      throw new Error(payload?.detail || payload?.error || "Stoody Book could not answer right now.");
    }

    const answer = payload?.data?.response || "Stoody Book returned an empty response.";
    state.messages.push({ role: "assistant", content: answer });
  } catch (error) {
    state.messages.push({
      role: "assistant",
      content: error?.message || "Stoody Book could not answer right now.",
    });
  } finally {
    saveState();
    renderConversation();
    setBusy(false);
  }
}

loadState();
sourceText.value = state.source;
renderConversation();

sourceText.addEventListener("input", () => {
  state.source = sourceText.value;
  saveState();
  sourceStatus.textContent = "Autosaved locally";
});

clearSourceButton.addEventListener("click", () => {
  state.source = "";
  sourceText.value = "";
  saveState();
  sourceText.focus();
});

newSessionButton.addEventListener("click", () => {
  state.sessionId = crypto.randomUUID();
  state.messages = [];
  saveState();
  renderConversation();
  messageInput.focus();
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.prompt || "";
    messageInput.focus();
  });
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  messageInput.value = "";
  await sendMessage(text);
});
