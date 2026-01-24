/* specsage chat client: POSTs to /ask and renders the SSE event stream. */

const log = document.getElementById("log");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const send = document.getElementById("send");
const tpl = document.getElementById("tpl-exchange");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (question.length < 3) return;
  input.value = "";
  setBusy(true);
  const view = newExchange(question);
  try {
    await streamAsk(question, view);
  } catch (err) {
    showStatus(view, `stream failed: ${err.message}`, true);
  } finally {
    view.answer.classList.remove("streaming");
    setBusy(false);
    input.focus();
  }
});

function setBusy(busy) {
  send.disabled = busy;
  input.disabled = busy;
}

function newExchange(question) {
  const node = tpl.content.cloneNode(true);
  const section = node.querySelector(".exchange");
  node.querySelector(".q-text").textContent = question;
  log.appendChild(node);
  section.scrollIntoView({ block: "end" });
  return {
    section,
    status: section.querySelector(".status-line"),
    answer: section.querySelector(".answer"),
    sources: section.querySelector(".sources"),
    sourceList: section.querySelector(".source-list"),
    raw: "",
  };
}

function showStatus(view, text, isError = false) {
  view.status.hidden = false;
  view.status.textContent = text;
  view.status.classList.toggle("error-line", isError);
}

async function streamAsk(question, view) {
  const resp = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (!frame.startsWith("data: ")) continue;
      const payload = frame.slice(6);
      if (payload === "[DONE]") return;
      handleEvent(JSON.parse(payload), view);
    }
  }
}

function handleEvent(event, view) {
  switch (event.type) {
    case "queries":
      showStatus(view, `retrieving: ${event.queries.join("  |  ")}`);
      break;
    case "sources":
      renderSources(view, event.sources);
      break;
    case "token":
      view.answer.hidden = false;
      view.answer.classList.add("streaming");
      view.raw += event.text;
      view.answer.textContent = view.raw;
      view.section.scrollIntoView({ block: "end" });
      break;
    case "final":
      renderFinal(view, event.result);
      break;
    case "error":
      showStatus(view, event.message, true);
      break;
  }
}

function renderSources(view, sources) {
  view.sources.hidden = false;
  view.sources.open = false;
  view.sourceList.replaceChildren(
    ...sources.map((s) => {
      const li = document.createElement("li");
      li.id = sourceRowId(view, s.n);
      const n = document.createElement("span");
      n.className = "source-n";
      n.textContent = `[${s.n}]`;
      const a = document.createElement("a");
      a.href = s.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = `${s.label} — ${s.title}`;
      const score = document.createElement("span");
      score.className = "source-score";
      score.textContent = s.score.toFixed(2);
      li.append(n, a, score);
      return li;
    })
  );
}

function sourceRowId(view, n) {
  if (!view.uid) view.uid = `x${Math.random().toString(36).slice(2, 8)}`;
  return `${view.uid}-src-${n}`;
}

function renderFinal(view, result) {
  view.answer.hidden = false;
  view.answer.classList.remove("streaming");
  if (result.refused) {
    view.answer.classList.add("refused");
    view.answer.textContent = result.answer;
    return;
  }
  // Re-render the verified answer with citation chips linking to source rows.
  view.answer.replaceChildren(...withChips(result.answer, view));
  view.status.hidden = true;
}

function withChips(text, view) {
  const nodes = [];
  let last = 0;
  for (const m of text.matchAll(/\[(\d{1,2})\]/g)) {
    if (m.index > last) nodes.push(document.createTextNode(text.slice(last, m.index)));
    const chip = document.createElement("a");
    chip.className = "chip";
    chip.href = `#${sourceRowId(view, Number(m[1]))}`;
    chip.textContent = m[1];
    chip.setAttribute("aria-label", `citation ${m[1]}`);
    chip.addEventListener("click", (e) => {
      e.preventDefault();
      flashSource(view, Number(m[1]));
    });
    nodes.push(chip);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(document.createTextNode(text.slice(last)));
  return nodes;
}

function flashSource(view, n) {
  view.sources.open = true;
  const row = document.getElementById(sourceRowId(view, n));
  if (!row) return;
  row.scrollIntoView({ block: "nearest" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1200);
}

input.focus();
