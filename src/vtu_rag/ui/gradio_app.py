"""Student-facing chat UI. Talks to the FastAPI service over HTTP.

Laid out as a desk rather than a chat window: what you're studying on the left,
the answer in the middle, and the diagrams you have to reproduce in the exam kept
open on the right, where they stay put instead of scrolling away.
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import gradio as gr
import httpx

from vtu_rag.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
API = settings.api_base_url.rstrip("/")
# Images and PDFs are fetched by the browser, not by this process
PUBLIC_API = settings.public_api_base_url.rstrip("/")
ALL = "All"
MODES = {
    "Careful": "agentic-ask",  # guardrail, grading, retry
    "Quick": "ask",  # one retrieval, one answer
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Literata:opsz,wght@7..72,400;7..72,500&display=swap');

:root {
    --ink: #16243F;
    --ink-soft: #4A5A78;
    --paper: #FBFAF7;
    --pane: #FFFFFF;
    --rule: #E6E1D6;
    --teal: #0F766E;
    --marker: #FCE7A2;
}

@media (prefers-color-scheme: dark) {
    :root {
        --ink: #E8EAF0;
        --ink-soft: #9AA6BD;
        --paper: #131722;
        --pane: #1A1F2D;
        --rule: #2C3444;
        --teal: #4CC2B4;
        --marker: #4A421F;
    }
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: var(--paper) !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}

/* masthead ------------------------------------------------------------- */
#masthead {
    border-bottom: 1px solid var(--rule);
    padding: 14px 22px 12px;
    background: var(--pane);
}
#masthead h1 {
    font-size: 19px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin: 0;
}
#masthead p {
    margin: 2px 0 0;
    font-size: 13px;
    color: var(--ink-soft);
}

/* three panes ----------------------------------------------------------- */
#shell { gap: 0 !important; }
#scope, #findings {
    background: var(--pane);
    padding: 18px 16px;
    min-height: calc(100vh - 62px);
}
#scope { border-right: 1px solid var(--rule); }
#findings { border-left: 1px solid var(--rule); }
#answer { padding: 0 18px 12px; }

.pane-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 10px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--rule);
}
.hint { font-size: 12px; color: var(--ink-soft); line-height: 1.5; }

/* the answer itself reads like prose, not chat chrome -------------------- */
#chat { border: none !important; background: transparent !important; }
#chat .message-row { max-width: 100% !important; }
#chat .bot, #chat .user, #chat .user-row { border-radius: 4px !important; }
#chat .bot, #chat .bot-row .message {
    font-family: 'Literata', Georgia, serif !important;
    font-size: 15.5px !important;
    line-height: 1.68 !important;
    color: var(--ink) !important;
    background: transparent !important;
    border: none !important;
    max-width: 70ch;
}
#chat .user, #chat .user-row .message {
    background: var(--marker) !important;
    color: var(--ink) !important;
    border: none !important;
    font-size: 14.5px !important;
}
#chat h2, #chat h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 15px !important;
}

#askbox textarea {
    font-size: 15px !important;
    border-radius: 6px !important;
    border: 1px solid var(--rule) !important;
    background: var(--pane) !important;
}
#askbox textarea:focus { border-color: var(--teal) !important; box-shadow: none !important; }

/* diagram rail ---------------------------------------------------------- */
#figures { background: transparent !important; border: none !important; }
#figures .grid-wrap { gap: 10px !important; }
#sources a { color: var(--teal); text-decoration: none; border-bottom: 1px solid var(--rule); }
#sources a:hover { border-bottom-color: var(--teal); }
#sources p, #sources li { font-size: 13px; line-height: 1.55; color: var(--ink); }
#status { font-size: 12px; color: var(--ink-soft); min-height: 18px; }

/* respect reduced motion and keep focus visible */
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
:focus-visible { outline: 2px solid var(--teal) !important; outline-offset: 2px; }

@media (max-width: 900px) {
    #scope, #findings { min-height: auto; border: none; border-top: 1px solid var(--rule); }
}
"""


# ----------------------------------------------------------------- API calls
def _get(path: str, **params: Any) -> Any:
    response = httpx.get(f"{API}{path}", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def load_subjects(semester: str) -> list[tuple[str, str]]:
    params: dict[str, Any] = {"branch": settings.default_branch, "scheme": settings.default_scheme}
    if semester != ALL:
        params["semester"] = int(semester)
    try:
        subjects = _get("/api/v1/subjects", **params)
    except httpx.HTTPError as exc:
        logger.warning("Could not load subjects: %s", exc)
        return []
    return [(f"{s['code']} — {s['name']}", s["code"]) for s in subjects]


def load_modules(subject_code: str) -> list[tuple[str, int]]:
    if not subject_code or subject_code == ALL:
        return []
    try:
        data = _get(f"/api/v1/subjects/{subject_code}/modules")
    except httpx.HTTPError:
        return []
    choices = []
    for m in data["modules"]:
        label = f"M{m['number']}"
        if m["title"]:
            label += f" {m['title']}"
        # Say plainly which modules have nothing to search yet
        if not m["indexed_notes"]:
            label += " (no notes yet)"
        choices.append((label, m["number"]))
    return choices


# ----------------------------------------------------------------- rendering
def figure_gallery(figures: list[dict[str, Any]]) -> list[tuple[str, str]]:
    items = []
    for figure in figures:
        caption = figure.get("caption") or (
            f"Scanned page {figure['page']}"
            if figure.get("kind") == "page"
            else f"{figure['subject_code']} module {figure['module_number']}, page {figure['page']}"
        )
        items.append((f"{PUBLIC_API}{figure['url']}", caption))
    return items


def format_notes(notes: list[dict[str, Any]]) -> str:
    """Links the whole source note. PDF viewers honour #page=N, so it opens where it matters."""
    if not notes:
        return ""
    lines = ["**Where this came from**", ""]
    for note in notes:
        pages = note.get("pages") or []
        anchor = f"#page={pages[0]}" if pages else ""
        where = ""
        if pages:
            where = (
                f", page {pages[0]}"
                if len(pages) == 1
                else f", pages {', '.join(map(str, pages[:6]))}"
            )
        lines.append(
            f"- [{note['subject_code']} module {note['module_number']} notes]"
            f"({PUBLIC_API}{note['url']}{anchor}){where}"
        )
    return "\n".join(lines)


def format_status(data: dict[str, Any]) -> str:
    if data.get("in_scope") is False:
        return "Outside the syllabus, so nothing was searched."
    bits = []
    if data.get("rewritten_queries"):
        bits.append(f"searched again as “{data['rewritten_queries'][-1]}”")
    if data.get("search_mode"):
        bits.append(f"{data['search_mode']} search")
    bits.append("from cache" if data.get("cached") else f"{data.get('latency_ms', 0) / 1000:.1f}s")
    return " · ".join(bits)


# ------------------------------------------------------------------ chat fn
async def ask(
    message: str,
    history: list[dict[str, Any]],
    semester: str,
    subject_code: str | None,
    modules: list[int],
    mode_label: str,
) -> AsyncIterator[tuple]:
    question = (message or "").strip()
    history = list(history or [])
    if len(question) < 3:
        yield history, gr.skip(), gr.skip(), "Ask a question about your notes.", message
        return

    history.append({"role": "user", "content": question})
    yield history, [], "", "Searching your notes…", ""

    payload: dict[str, Any] = {
        "question": question,
        "branch": settings.default_branch,
        "scheme": settings.default_scheme,
        "module_numbers": modules or [],
    }
    if semester != ALL:
        payload["semester"] = int(semester)
    if subject_code and subject_code != ALL:
        payload["subject_code"] = subject_code

    endpoint = MODES.get(mode_label, "agentic-ask")
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(f"{API}/api/v1/{endpoint}", json=payload)
    except httpx.HTTPError as exc:
        history.append({"role": "assistant", "content": f"Can't reach the API at {API}."})
        logger.warning("API unreachable: %s", exc)
        yield history, [], "", "The API isn't responding. Is the stack running?", ""
        return

    if response.status_code >= 400:
        detail = response.json().get("detail", "") if response.content else ""
        history.append({"role": "assistant", "content": f"That didn't work. {detail}"})
        yield history, [], "", f"Error {response.status_code}", ""
        return

    data = response.json()
    history.append({"role": "assistant", "content": data["answer"]})
    yield (
        history,
        figure_gallery(data.get("figures", [])),
        format_notes(data.get("notes", [])),
        format_status(data),
        "",
    )


# ---------------------------------------------------------------------- app
EXAMPLES = [
    "Explain dual-mode operation with a neat diagram.",
    "What is a system call? List its types.",
    "Difference between monolithic and microkernel structures.",
]


# Gradio 6 takes theme and css at launch() rather than on the Blocks
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    primary_hue="teal",
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VTU Exam Buddy", fill_width=True) as demo:
        with gr.Row(elem_id="masthead"):
            gr.HTML(
                "<h1>VTU Exam Buddy</h1>"
                f"<p>Answers built only from your {settings.default_branch.upper()} "
                f"{settings.default_scheme}-scheme notes, with the diagrams to redraw.</p>"
            )

        with gr.Row(elem_id="shell", equal_height=False):
            # ---------------------------------------------------- scope rail
            with gr.Column(scale=2, min_width=230, elem_id="scope"):
                gr.HTML("<p class='pane-title'>What to search</p>")
                semester = gr.Dropdown(
                    [ALL] + [str(i) for i in range(1, 9)],
                    value="3",
                    label="Semester",
                    container=True,
                )
                subject = gr.Dropdown([(ALL, ALL)] + load_subjects("3"), value=ALL, label="Subject")
                modules = gr.CheckboxGroup([], label="Modules")
                mode = gr.Radio(
                    list(MODES),
                    value="Careful",
                    label="Answering",
                    info="Careful checks the question is on-syllabus, grades what it finds "
                    "and searches again when the results are weak. Quick answers in one pass.",
                )

            # -------------------------------------------------- answer column
            with gr.Column(scale=7, elem_id="answer"):
                chat = gr.Chatbot(
                    elem_id="chat",
                    show_label=False,
                    height="calc(100vh - 235px)",
                    placeholder="Ask anything from your notes — a definition, a comparison, "
                    "a diagram to draw.",
                )
                question = gr.Textbox(
                    placeholder="Ask a question from your notes",
                    show_label=False,
                    submit_btn=True,
                    elem_id="askbox",
                    max_lines=4,
                )
                gr.Examples(EXAMPLES, inputs=question, label="Try")

            # ------------------------------------------------- diagram rail
            with gr.Column(scale=3, min_width=250, elem_id="findings"):
                gr.HTML("<p class='pane-title'>Diagrams to redraw</p>")
                figures = gr.Gallery(
                    show_label=False,
                    elem_id="figures",
                    columns=1,
                    height=330,
                    object_fit="contain",
                    preview=False,
                )
                sources = gr.Markdown("", elem_id="sources")
                status = gr.Markdown("", elem_id="status")

        # ------------------------------------------------------------ wiring
        inputs = [question, chat, semester, subject, modules, mode]
        outputs = [chat, figures, sources, status, question]
        question.submit(ask, inputs, outputs)

        def on_semester(sem: str):
            return (
                gr.update(choices=[(ALL, ALL)] + load_subjects(sem), value=ALL),
                gr.update(choices=[], value=[]),
            )

        semester.change(on_semester, semester, [subject, modules])
        subject.change(
            lambda code: gr.update(choices=load_modules(code), value=[]), subject, modules
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    build_app().launch(
        server_name="0.0.0.0",
        enable_monitoring=False,
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
        css=CSS,
    )
