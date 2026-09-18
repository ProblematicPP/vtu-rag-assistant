"""Student-facing UI. Talks to the FastAPI service over HTTP.

A conversation, in the shape people already know from Claude: the thread holds
everything you have asked this session, the composer sits under it, and what to
search — semester, subject, modules — lives in the sidebar. A question paper can
be attached to the composer, and every question in it is answered into the thread.
"""

import html
import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from vtu_rag.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
API = settings.api_base_url.rstrip("/")
# Images and PDFs are loaded by the browser, not by this process
PUBLIC_API = settings.public_api_base_url.rstrip("/")
ALL = "All"
MODES = {
    "Careful": "agentic-ask",  # guardrail, grading, retry
    "Quick": "ask",  # one retrieval, one answer
}

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600&"
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap"
)

CSS = f"""
@import url('{FONTS}');

:root {{
    --bg: #FAF9F5;
    --sidebar: #F0EEE6;
    --card: #FFFFFF;
    --bubble: #F0EEE6;
    --ink: #1F1E1D;
    --ink-2: #3D3A36;
    --ink-3: #6B6760;  /* muted, but still 5:1 on the sidebar — small text has to pass */
    --line: #E3E0D8;
    --hover: rgba(0, 0, 0, 0.05);
    --accent: #D97757;
    --ui: 'Inter', system-ui, -apple-system, sans-serif;
    --serif: 'Source Serif 4', Georgia, serif;
}}

/* Gradio puts `dark` on <body> when the OS asks for it. Flip every token
   together — a half-flip is what left dark text sitting on a black panel. */
body.dark {{
    --bg: #262624;
    --sidebar: #1F1E1D;
    --card: #2E2D2A;
    --bubble: #35342F;
    --ink: #F5F4EF;
    --ink-2: #DAD7D1;
    --ink-3: #9A958D;
    --line: #3B3936;
    --hover: rgba(255, 255, 255, 0.07);
}}

.gradio-container, .gradio-container * {{ font-family: var(--ui); }}
.gradio-container, body, body.dark {{
    max-width: 100% !important; padding: 0 !important;
    background: var(--bg) !important; color: var(--ink) !important;
}}
footer {{ display: none !important; }}

/* Every Gradio surface takes its colour from the tokens above, in both modes */
.block, .form, .wrap, .wrap-inner, .container,
.dark .block, .dark .form, .dark .wrap, .dark .wrap-inner {{
    background: transparent !important; color: var(--ink) !important;
}}
input, textarea, select, .dark input, .dark textarea, .dark select {{
    background: var(--card) !important; color: var(--ink) !important;
}}
label, label span, .dark label, .dark label span {{ color: var(--ink-2) !important; }}

/* ------------------------------------------------------------------ sidebar */
#sidebar {{
    background: var(--sidebar) !important;
    border-right: 1px solid var(--line);
    padding: 16px 12px 24px !important;
    min-height: 100vh;
    gap: 2px !important;
}}
#sidebar .block, #sidebar .form {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}}
#brand {{
    display: flex; align-items: center; gap: 8px;
    font-size: 15px; font-weight: 600; color: var(--ink);
    padding: 4px 8px 14px;
}}
#brand .star {{ color: var(--accent); font-size: 17px; line-height: 1; }}

#sidebar button.new-chat {{
    background: var(--bg) !important; color: var(--ink) !important;
    border: 1px solid var(--line) !important; border-radius: 10px !important;
    font-size: 13.5px !important; font-weight: 500 !important;
    padding: 9px 12px !important; box-shadow: none !important; text-align: left !important;
    margin-bottom: 16px !important;
}}
#sidebar button.new-chat:hover {{
    background: var(--card) !important; border-color: var(--ink-3) !important;
}}

.side-label {{
    font-size: 11.5px; font-weight: 600; color: var(--ink-3);
    padding: 0 8px; margin: 14px 0 6px; letter-spacing: 0.01em;
}}
#recents {{ display: flex; flex-direction: column; gap: 1px; }}
#recents button {{
    background: transparent !important; border: none !important; box-shadow: none !important;
    text-align: left !important; font-size: 13px !important; color: var(--ink-2) !important;
    padding: 7px 8px !important; border-radius: 8px !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
#recents button:hover {{ background: var(--hover) !important; }}
#recents .empty {{ font-size: 12.5px; color: var(--ink-3); padding: 4px 8px; line-height: 1.5; }}

#sidebar label > span:first-child {{
    font-size: 12px !important; font-weight: 500 !important; color: var(--ink-3) !important;
    margin-bottom: 4px !important;
}}
#sidebar .wrap, #sidebar input, #sidebar .wrap-inner {{
    background: var(--bg) !important; border-radius: 8px !important;
    font-size: 13px !important; box-shadow: none !important;
}}
#sidebar .wrap {{ border: 1px solid var(--line) !important; }}
#sidebar .wrap-inner, #sidebar input {{ border: none !important; }}
#sidebar .block:not(:last-child) {{ margin-bottom: 12px !important; }}
#sidebar [data-testid='checkbox-group'] label, #sidebar .gr-check-radio label {{
    font-size: 12.5px !important;
}}
#index-note {{ font-size: 11.5px; color: var(--ink-3); line-height: 1.55; padding: 10px 8px 0; }}
#index-note b {{ color: var(--ink-2); font-weight: 600; }}

/* ------------------------------------------------------------------- thread */
#main {{ padding: 0 !important; min-width: 0; background: var(--bg) !important; }}
#thread, #thread .bubble-wrap, #thread .message-wrap {{
    background: var(--bg) !important;
}}
#thread {{
    border: none !important;
    max-width: 760px; margin: 0 auto !important; padding: 26px 20px 0 !important;
}}
#thread .message-row {{ max-width: 100% !important; }}
#thread .user-row, #thread .user {{
    background: var(--bubble) !important; border: none !important;
    border-radius: 14px !important; color: var(--ink) !important;
    font-size: 15px !important; line-height: 1.55 !important; padding: 11px 15px !important;
}}
#thread .bot-row, #thread .bot {{
    background: transparent !important; border: none !important; padding: 2px 0 10px !important;
}}
#thread .bot p, #thread .bot li {{
    font-size: 15.5px !important; line-height: 1.68 !important; color: var(--ink-2) !important;
}}
#thread .bot strong {{ color: var(--ink) !important; font-weight: 600 !important; }}
#thread .bot h1, #thread .bot h2, #thread .bot h3, #thread .bot h4 {{
    font-size: 15px !important; font-weight: 600 !important;
    color: var(--ink) !important; margin: 18px 0 6px !important;
}}
#thread .bot img {{
    border: 1px solid var(--line); border-radius: 10px; background: #FFFFFF;
    padding: 8px; max-width: 420px; margin: 10px 0 4px;
}}
#thread .bot em {{
    color: var(--ink-3) !important; font-style: normal !important; font-size: 12.5px !important;
}}
#thread .bot a {{ color: var(--accent) !important; text-decoration: none !important; }}
#thread .bot a:hover {{ text-decoration: underline !important; }}
#thread .placeholder-content, #thread .placeholder {{ display: none !important; }}

/* greeting shown before the first question */
#greeting {{
    max-width: 760px; margin: 0 auto; padding: 22vh 20px 0; text-align: center;
    min-height: calc(100vh - 330px);
}}
#greeting .star {{ color: var(--accent); font-size: 26px; display: block; margin-bottom: 14px; }}
#greeting h1 {{
    font-family: var(--serif) !important; font-weight: 400 !important;
    font-size: 30px !important; color: var(--ink) !important; margin: 0 0 10px !important;
}}
#greeting p {{ font-size: 14px !important; color: var(--ink-3) !important; margin: 0 !important; }}

/* ----------------------------------------------------------------- composer */
#composer {{
    max-width: 760px; margin: 0 auto !important; padding: 10px 20px 22px !important;
    gap: 8px !important;
}}
#composer .block, #composer .form, #ask, #ask > div {{
    background: transparent !important; border: none !important; box-shadow: none !important;
    padding: 0 !important;
}}
#ask textarea {{
    font-size: 15.5px !important; line-height: 1.5 !important;
    height: auto !important; min-height: 58px !important;
    padding: 16px 16px !important; color: var(--ink) !important;
    background: var(--card) !important;
    border: 1px solid var(--line) !important; border-radius: 16px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
}}
#ask textarea:focus {{ border-color: var(--ink-3) !important; }}
#ask textarea::placeholder {{ color: var(--ink-3) !important; }}
#ask button {{
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    width: 34px !important; height: 34px !important; align-self: flex-end !important;
    margin-bottom: 12px !important;
}}
#ask button:hover {{ filter: brightness(0.94); }}

#tools {{ gap: 8px !important; justify-content: flex-start !important; flex-wrap: wrap; }}
#tools > *, #tools .block {{
    flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
}}
#tools button {{
    background: transparent !important; color: var(--ink-3) !important;
    border: 1px solid var(--line) !important; border-radius: 9px !important;
    font-size: 12.5px !important; font-weight: 500 !important;
    padding: 7px 12px !important; box-shadow: none !important;
    min-width: 0 !important; width: auto !important;
}}
#tools button:hover {{ background: var(--sidebar) !important; color: var(--ink) !important; }}
#tools .download-button {{
    border-color: var(--accent) !important; color: var(--accent) !important;
}}

.chips {{ display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; margin: 16px 0 0; }}
.chips button {{
    font-size: 12.5px; color: var(--ink-2); background: var(--bg);
    border: 1px solid var(--line); border-radius: 999px; padding: 7px 13px; cursor: pointer;
}}
.chips button:hover {{ background: var(--card); border-color: var(--ink-3); }}

/* Gradio's own chrome, toned to this palette */
.progress-bar, .progress-level-inner {{ background: var(--accent) !important; }}
.progress-text, .progress-level {{ color: var(--ink-3) !important; font-size: 11px !important; }}
:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
}}
@media (max-width: 860px) {{
    #sidebar {{ min-height: auto; border-right: none; border-bottom: 1px solid var(--line); }}
}}
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
    return [(f"{s['code']}  {s['name']}", s["code"]) for s in subjects]


def load_modules(subject_code: str) -> list[tuple[str, int]]:
    if not subject_code or subject_code == ALL:
        return []
    try:
        data = _get(f"/api/v1/subjects/{subject_code}/modules")
    except httpx.HTTPError:
        return []
    choices = []
    for module in data["modules"]:
        label = f"M{module['number']}"
        if module["title"]:
            label += f" · {module['title'][:28]}"
        # Say plainly which modules have nothing to search yet
        if not module["indexed_notes"]:
            label += " (empty)"
        choices.append((label, module["number"]))
    return choices


def index_summary() -> str:
    """One honest line about what can actually be searched."""
    try:
        health = _get("/health")
        notes = _get("/api/v1/notes", limit=500)
    except httpx.HTTPError:
        return "Can't reach the API — check that the stack is running."

    indexed = [n for n in notes if n["status"] == "indexed"]
    sections = health["services"]["opensearch"].get("chunks") or 0
    if not indexed:
        return "No notes indexed yet. Put PDFs in <b>data/</b>, then run a sync."
    subjects = {n["subject_code"] for n in indexed}
    search = "keyword and meaning" if health["services"]["embeddings"]["ok"] else "keyword only"
    return (
        f"<b>{len(indexed)}</b> note{'s' if len(indexed) != 1 else ''} from "
        f"<b>{len(subjects)}</b> subject{'s' if len(subjects) != 1 else ''} · "
        f"<b>{sections}</b> sections · matching by {search}"
    )


# ----------------------------------------------------------------- rendering
def answer_markdown(data: dict[str, Any]) -> str:
    """An answer as one chat message: prose, then diagrams, then where it came from."""
    parts = [data.get("answer") or "_No answer came back._"]

    for figure in data.get("figures") or []:
        url = f"{PUBLIC_API}{figure['url']}"
        caption = figure.get("caption") or (
            f"Scanned page {figure['page']}"
            if figure.get("kind") == "page"
            else f"Diagram to redraw — page {figure['page']}"
        )
        parts.append(f"\n![{caption}]({url})\n\n*{caption}*")

    footer = []
    for note in data.get("notes") or []:
        pages = note.get("pages") or []
        anchor = f"#page={pages[0]}" if pages else ""
        where = f", p. {pages[0]}" if pages else ""
        footer.append(
            f"[{note['subject_code']} module {note['module_number']} notes"
            f"{where}]({PUBLIC_API}{note['url']}{anchor})"
        )
    run = []
    if data.get("in_scope") is False:
        run.append("outside the syllabus, nothing searched")
    if data.get("rewritten_queries"):
        run.append(f"searched again as “{data['rewritten_queries'][-1]}”")
    if data.get("search_mode"):
        run.append(f"{data['search_mode']} search")
    if data.get("cached"):
        run.append("from cache")
    elif data.get("latency_ms"):
        run.append(f"{data['latency_ms'] / 1000:.1f}s")

    if footer or run:
        line = " · ".join(footer + run)
        parts.append(f"\n\n*From your notes: {line}*" if footer else f"\n\n*{line}*")
    return "".join(parts)


def recents_html(questions: list[str]) -> str:
    if not questions:
        return "<div id='recents'><p class='empty'>Questions you ask will be listed here.</p></div>"
    items = "".join(
        f"<button type='button' title='{html.escape(q)}'>{html.escape(q[:46])}</button>"
        for q in reversed(questions[-12:])
    )
    return f"<div id='recents'>{items}</div>"


GREETING = (
    "<div id='greeting'><span class='star'>✳</span>"
    "<h1>What are we revising?</h1>"
    "<p>Answers come only from the notes you've added — never from anywhere else.</p>"
    "</div>"
)


# -------------------------------------------------------------------- asking
def _filters(semester: str, subject_code: str | None, modules: list[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branch": settings.default_branch,
        "scheme": settings.default_scheme,
        "module_numbers": modules or [],
    }
    if semester != ALL:
        payload["semester"] = int(semester)
    if subject_code and subject_code != ALL:
        payload["subject_code"] = subject_code
    return payload


async def ask(
    question: str,
    history: list[dict[str, Any]],
    recents: list[str],
    semester: str,
    subject_code: str | None,
    modules: list[int],
    mode_label: str,
) -> AsyncIterator[tuple]:
    asked = (question or "").strip()
    history = list(history or [])
    if len(asked) < 3:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    recents = [*(recents or []), asked]
    history.append({"role": "user", "content": asked})
    history.append({"role": "assistant", "content": "_Reading your notes…_"})
    yield history, recents, recents_html(recents), "", gr.update(visible=False)

    endpoint = MODES.get(mode_label, "agentic-ask")
    payload = {"question": asked, **_filters(semester, subject_code, modules)}
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(f"{API}/api/v1/{endpoint}", json=payload)
        data = response.json() if response.status_code < 400 else {}
        if response.status_code >= 400:
            data = {"answer": f"That didn't work: {data.get('detail', response.status_code)}"}
    except httpx.HTTPError as exc:
        logger.warning("API unreachable: %s", exc)
        data = {"answer": f"I can't reach the API at {API}. Is the stack running?"}

    history[-1] = {"role": "assistant", "content": answer_markdown(data)}
    yield history, recents, recents_html(recents), "", gr.update(visible=False)


async def solve_paper(
    file_path: str | None,
    history: list[dict[str, Any]],
    semester: str,
    subject_code: str | None,
    modules: list[int],
) -> AsyncIterator[tuple]:
    """Attaching a question paper answers every question into the thread."""
    history = list(history or [])
    if not file_path:
        yield gr.skip(), gr.skip(), gr.skip()
        return

    name = Path(file_path).name
    history.append({"role": "user", "content": f"📄 **{name}** — answer every question"})
    history.append({"role": "assistant", "content": "_Reading the question paper…_"})
    yield history, gr.update(visible=False), gr.skip()

    try:
        with open(file_path, "rb") as handle:
            extract = httpx.post(
                f"{API}/api/v1/papers/extract",
                files={"file": (name, handle, "application/octet-stream")},
                timeout=900,
            )
        payload = extract.json()
    except httpx.HTTPError as exc:
        history[-1] = {"role": "assistant", "content": f"I couldn't read that file: {exc}"}
        yield history, gr.update(visible=False), gr.skip()
        return

    if extract.status_code >= 400:
        history[-1] = {
            "role": "assistant",
            "content": f"I couldn't read that paper. {payload.get('detail', '')}",
        }
        yield history, gr.update(visible=False), gr.skip()
        return

    questions = payload.get("questions") or []
    if not questions:
        history[-1] = {
            "role": "assistant",
            "content": "I couldn't find any questions in that file. VTU papers number their "
            "parts like “Q.1 a.” — a straighter, sharper scan usually fixes it.",
        }
        yield history, gr.update(visible=False), gr.skip()
        return

    marks = sum(q.get("marks") or 0 for q in questions)
    listing = "\n".join(
        f"**Q{q['number']}** {q['text']}" + (f"  ·  {q['marks']} marks" if q.get("marks") else "")
        for q in questions
    )
    history[-1] = {
        "role": "assistant",
        "content": f"Found **{len(questions)} questions**"
        + (f" worth {marks} marks" if marks else "")
        + f" in {name}. Answering them now.\n\n{listing}",
    }
    yield history, gr.update(visible=False), gr.skip()

    filters = _filters(semester, subject_code, modules)
    async with httpx.AsyncClient(timeout=900) as client:
        for index, question in enumerate(questions, start=1):
            heading = f"**Q{question['number']}** · {question['text']}"
            history.append({"role": "assistant", "content": f"{heading}\n\n_Answering…_"})
            yield history, gr.update(visible=False), gr.skip()

            try:
                response = await client.post(
                    f"{API}/api/v1/ask", json={"question": question["text"], **filters}
                )
                data = response.json() if response.status_code < 400 else {}
            except httpx.HTTPError as exc:
                logger.warning("Question %s failed: %s", question["number"], exc)
                data = {}
            if not data.get("answer"):
                data = {"answer": "_I couldn't answer this one from the indexed notes._"}

            history[-1] = {
                "role": "assistant",
                "content": f"{heading}\n\n{answer_markdown(data)}",
            }
            logger.info("Answered %d of %d", index, len(questions))
            yield history, gr.update(visible=False), gr.skip()

    pdf_path = await build_paper_pdf(questions, Path(name).stem, filters)
    history.append(
        {
            "role": "assistant",
            "content": "All questions answered. **Download as PDF** below the composer gives you "
            "the whole set to print, diagrams included.",
        }
    )
    yield history, gr.update(value=pdf_path, visible=bool(pdf_path)), gr.skip()


async def build_paper_pdf(
    questions: list[dict[str, Any]], title: str, filters: dict[str, Any]
) -> str | None:
    payload = {"questions": questions, "title": title or "Question paper", **filters}
    try:
        async with httpx.AsyncClient(timeout=900) as client:
            response = await client.post(f"{API}/api/v1/papers/pdf", json=payload)
        if response.status_code >= 400:
            logger.warning("PDF build failed: %s", response.text[:200])
            return None
    except httpx.HTTPError as exc:
        logger.warning("PDF build failed: %s", exc)
        return None

    safe = re.sub(r"[^A-Za-z0-9 _-]", "", title or "answers").strip() or "answers"
    path = Path(tempfile.gettempdir()) / f"{safe} - answers.pdf"
    path.write_bytes(response.content)
    return str(path)


# ---------------------------------------------------------------------- app
STARTERS = [
    "Explain dual-mode operation with a neat diagram",
    "What is a system call? List its types",
    "Difference between monolithic and microkernel structures",
]

# Clicking a starter chip or a recent question fills the composer
FILL_JS = """() => {
    const fill = (text) => {
        const box = document.querySelector('#ask textarea');
        box.value = text;
        box.dispatchEvent(new Event('input', { bubbles: true }));
        box.focus();
    };
    document.addEventListener('click', (event) => {
        const chip = event.target.closest('.chips button');
        if (chip) { fill(chip.textContent.trim()); return; }
        const recent = event.target.closest('#recents button');
        if (recent) fill(recent.getAttribute('title') || recent.textContent.trim());
    });
}"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VTU Exam Buddy", fill_width=True) as demo:
        recents_state = gr.State([])

        with gr.Row(equal_height=False):
            # ---------------------------------------------------------- sidebar
            with gr.Column(scale=2, min_width=235, elem_id="sidebar"):
                gr.HTML("<div id='brand'><span class='star'>✳</span> VTU Exam Buddy</div>")
                new_chat = gr.Button("＋  New question", elem_classes="new-chat")

                gr.HTML("<p class='side-label'>What to search</p>")
                semester = gr.Dropdown(
                    [ALL] + [str(i) for i in range(1, 9)], value="3", label="Semester"
                )
                subject = gr.Dropdown([(ALL, ALL)] + load_subjects("3"), value=ALL, label="Subject")
                modules = gr.CheckboxGroup([], label="Modules")
                mode = gr.Radio(list(MODES), value="Careful", label="Answering")

                gr.HTML("<p class='side-label'>This session</p>")
                recents = gr.HTML(recents_html([]))
                # Read at page load, not at startup: the API may still be booting
                index_note = gr.HTML("<p id='index-note'>Checking what's indexed…</p>")

            # ----------------------------------------------------------- thread
            with gr.Column(scale=9, elem_id="main"):
                greeting = gr.HTML(GREETING)
                thread = gr.Chatbot(
                    elem_id="thread",
                    show_label=False,
                    height="calc(100vh - 330px)",
                    buttons=["copy"],
                    visible=False,
                )
                with gr.Column(elem_id="composer"):
                    question = gr.Textbox(
                        placeholder="Ask anything from your notes",
                        show_label=False,
                        submit_btn=True,
                        elem_id="ask",
                        max_lines=6,
                    )
                    with gr.Row(elem_id="tools"):
                        paper = gr.UploadButton(
                            "📄  Attach a question paper",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                            size="sm",
                        )
                        download = gr.DownloadButton("Download answers as PDF", visible=False)
                    starters = gr.HTML(
                        "<div class='chips'>"
                        + "".join(
                            f"<button type='button'>{html.escape(s)}</button>" for s in STARTERS
                        )
                        + "</div>"
                    )

        # ------------------------------------------------------------- wiring
        def open_thread() -> tuple:
            """The greeting gives way to the thread once there is something in it."""
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        question.submit(
            ask,
            [question, thread, recents_state, semester, subject, modules, mode],
            [thread, recents_state, recents, question, download],
        ).then(open_thread, None, [greeting, starters, thread])

        paper.upload(
            solve_paper,
            [paper, thread, semester, subject, modules],
            [thread, download, question],
        ).then(open_thread, None, [greeting, starters, thread])

        def start_over():
            return (
                [],
                "",
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        new_chat.click(start_over, None, [thread, question, greeting, starters, download])

        def on_semester(sem: str):
            return (
                gr.update(choices=[(ALL, ALL)] + load_subjects(sem), value=ALL),
                gr.update(choices=[], value=[]),
            )

        semester.change(on_semester, semester, [subject, modules])
        subject.change(
            lambda code: gr.update(choices=load_modules(code), value=[]), subject, modules
        )
        demo.load(lambda: f"<p id='index-note'>{index_summary()}</p>", None, index_note)
        demo.load(None, None, None, js=FILL_JS)
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    build_app().launch(
        server_name="0.0.0.0",
        enable_monitoring=False,
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
    )
