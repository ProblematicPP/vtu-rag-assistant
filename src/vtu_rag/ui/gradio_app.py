"""Student-facing UI. Talks to the FastAPI service over HTTP.

Shaped like the exam it prepares you for rather than like a chat app: you pick a
subject and module the way a question paper is organised, ask in the words a VTU
paper would use, and get back an answer script — prose to learn, the diagram to
redraw beside it, and the page of your own notes it came from.
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
    "family=Archivo:wght@400;500;600;700&"
    "family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap"
)

CSS = f"""
@import url('{FONTS}');

:root {{
    --paper: #F3F1EC;
    --sheet: #FFFFFF;
    --ink: #000000;
    --ink-2: #33363C;
    --ink-3: #6B7078;
    --rule: #D9D5CC;
    --marker: #EAFF3C;
    --pen: #14309B;
    --ui: 'Archivo', system-ui, sans-serif;
    --read: 'Newsreader', Georgia, serif;
}}

/* One deliberate look: photocopied paper. A prefers-color-scheme flip would
   only repaint these tokens — Gradio's own controls stay light, which left
   invisible text in the dropdowns. */
.gradio-container, .gradio-container * {{ font-family: var(--ui); }}
.gradio-container {{
    max-width: 100% !important;
    padding: 0 !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
}}
footer {{ display: none !important; }}

/* Gradio paints its own controls dark when the OS asks for dark. This design is
   a single paper look, so pin its surfaces and text back to the palette. */
.dark, .dark .block, .dark .form, .dark .gradio-container {{
    background: var(--paper) !important; color: var(--ink) !important;
}}
.dark input, .dark textarea, .dark select,
.dark .wrap, .dark .wrap-inner, .dark .secondary-wrap {{
    background: var(--paper) !important; color: var(--ink) !important;
}}
.dark label, .dark label span, .dark .block .label-wrap span {{
    color: var(--ink) !important; background: var(--sheet) !important;
}}
.dark input[type='radio'], .dark input[type='checkbox'] {{
    background: var(--sheet) !important; border-color: var(--rule) !important;
}}
.dark label.selected, .dark input:checked + span {{
    background: var(--marker) !important; color: var(--ink) !important;
}}
#masthead .wordmark b {{ color: var(--ink) !important; }}

/* ---------------------------------------------------------------- masthead */
#masthead {{
    display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    padding: 16px 28px 14px;
    border-bottom: 2px solid var(--ink);
    background: var(--sheet);
}}
#masthead .wordmark {{
    font-weight: 700; font-size: 17px; letter-spacing: -0.02em; color: var(--ink);
}}
#masthead .wordmark b {{ background: var(--marker); padding: 0 5px; }}
#masthead .tag {{ font-size: 13px; color: var(--ink-3); }}

/* ------------------------------------------------------------------- shell */
#shell {{ gap: 0 !important; align-items: stretch !important; }}
#scope {{
    background: var(--sheet);
    border-right: 1px solid var(--rule);
    padding: 20px 18px 28px;
    min-height: calc(100vh - 56px);
}}
#stage {{ padding: 26px 34px 40px; min-width: 0; }}

#scope .block, #scope .form, #stage .block, #stage .form {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
}}
#scope .block:not(:last-child) {{ margin-bottom: 16px; }}

#scope span[data-testid='block-info'], #scope label > span {{
    font-size: 12px !important; font-weight: 600 !important; color: var(--ink) !important;
}}
#scope .wrap, #scope .wrap-inner, #scope input, #scope select {{
    background: var(--paper) !important;
    border-radius: 2px !important;
    font-size: 13.5px !important;
    box-shadow: none !important;
}}
#scope .wrap {{ border: 1px solid var(--rule) !important; }}
#scope .wrap-inner, #scope input {{ border: none !important; }}
#scope label.selected, #scope label:has(input:checked) {{
    background: var(--marker) !important; border-color: var(--ink) !important;
}}
#scope .wrap label {{ font-size: 13px !important; }}

/* --------------------------------------------------------------- ask field */
#ask textarea {{
    font-family: var(--ui) !important;
    font-size: 17px !important;
    line-height: 1.45 !important;
    padding: 16px 18px !important;
    color: var(--ink) !important;
    background: var(--sheet) !important;
    border: 2px solid var(--ink) !important;
    border-radius: 2px !important;
    box-shadow: 4px 4px 0 var(--rule) !important;
}}
#ask textarea::placeholder {{ color: var(--ink-3) !important; }}
#ask textarea:focus {{ box-shadow: 4px 4px 0 var(--marker) !important; }}
#ask button {{
    background: var(--ink) !important; color: var(--sheet) !important;
    border: none !important; border-radius: 2px !important;
    width: 46px !important; height: 46px !important; align-self: center !important;
}}
#ask button:hover {{ background: var(--pen) !important; }}

.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 0; }}
.chips button {{
    font-family: var(--ui); font-size: 12.5px; color: var(--ink-2);
    background: var(--sheet); border: 1px solid var(--rule); border-radius: 2px;
    padding: 6px 11px; cursor: pointer;
}}
.chips button:hover {{ border-color: var(--ink); background: var(--marker); color: var(--ink); }}

/* ------------------------------------------------------------- empty state */
.paper-note {{ margin-top: 34px; max-width: 62ch; }}
#answer .paper-note h2, .paper-note h2 {{
    font-family: var(--read) !important; font-weight: 400 !important;
    font-size: 28px !important; line-height: 1.22 !important;
    color: var(--ink) !important; margin: 0 0 12px !important;
    letter-spacing: -0.01em !important;
}}
#answer .paper-note p, .paper-note p {{
    font-size: 14.5px !important; line-height: 1.65 !important;
    color: var(--ink-2) !important; margin: 0 0 20px !important;
    font-family: var(--ui) !important;
}}
.paper-note .index {{
    border-top: 1px solid var(--rule); padding-top: 14px;
    font-size: 13px; color: var(--ink-3); line-height: 1.7;
}}
.paper-note .index b {{ color: var(--ink); font-weight: 600; }}

/* ------------------------------------------------------------ answer sheet */
.asked {{
    font-size: 15px; font-weight: 500; line-height: 1.7; color: var(--ink);
    margin: 22px 0 20px; max-width: 62ch;
}}
.asked span {{ background: var(--marker); padding: 3px 6px; box-decoration-break: clone; }}

#answer {{ max-width: 66ch; }}
#answer p, #answer li {{
    font-family: var(--read) !important;
    font-size: 17px !important;
    line-height: 1.72 !important;
    color: var(--ink-2) !important;
}}
#answer h1, #answer h2, #answer h3, #answer strong {{
    font-family: var(--ui) !important; color: var(--ink) !important;
}}
#answer h1, #answer h2, #answer h3 {{
    font-size: 14.5px !important; font-weight: 600 !important; margin: 26px 0 8px !important;
}}
#answer ul, #answer ol {{ padding-left: 20px !important; }}
#answer li {{ margin: 6px 0 !important; }}
#answer code {{ font-size: 14px !important; background: var(--paper) !important; }}

/* ---------------------------------------------------------------- diagrams */
.figures {{ display: grid; gap: 18px; }}
.figures.strip {{
    grid-template-columns: repeat(auto-fill, minmax(230px, 320px));
    margin: 4px 0 8px; max-width: 78ch;
}}
#figure-slot {{ margin-top: 26px; }}
#figure-slot .rail-title {{ max-width: 78ch; }}
figure.fig {{ margin: 0; }}
figure.fig a {{
    display: block; background: var(--sheet); border: 1px solid var(--rule); padding: 10px;
}}
figure.fig a:hover {{ border-color: var(--ink); }}
figure.fig img {{ display: block; width: 100%; height: auto; }}
figure.fig figcaption {{
    font-size: 12px; line-height: 1.5; color: var(--ink-3); margin-top: 7px;
}}
figure.fig figcaption b {{ display: block; color: var(--ink-2); font-weight: 500; }}
.rail-title {{
    font-size: 11.5px; font-weight: 700; color: var(--ink); margin: 0 0 12px;
    padding-bottom: 6px; border-bottom: 2px solid var(--ink);
}}

/* ------------------------------------------------------- progress + status */
.progress-bar, .progress-level-inner, .eta-bar {{
    background: var(--ink) !important; color: var(--sheet) !important;
}}
.progress-text, .progress-level {{ color: var(--ink-3) !important; font-size: 11.5px !important; }}
.wrap.default.full, .wrap.default.generating {{ background: transparent !important; }}

/* ------------------------------------------------------------ tabs, buttons */
.tab-container button, .tab-nav button, button.tab {{
    font-size: 13.5px !important; font-weight: 500 !important;
    color: var(--ink-3) !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    padding: 8px 2px !important; margin-right: 22px !important; background: none !important;
}}
.tab-container button.selected, .tab-nav button.selected, button.tab.selected {{
    color: var(--ink) !important; border-bottom-color: var(--ink) !important;
}}
.tab-container, .tab-nav {{ border-bottom: 1px solid var(--rule) !important; }}
.tab-container.visually-hidden {{ border: none !important; }}

#stage button.primary, #stage button.lg, #stage .download-button {{
    background: var(--ink) !important; color: var(--sheet) !important;
    border: none !important; border-radius: 2px !important;
    font-size: 13.5px !important; font-weight: 600 !important;
    padding: 11px 20px !important; box-shadow: 3px 3px 0 var(--rule) !important;
}}
#stage button.primary:hover, #stage .download-button:hover {{
    background: var(--pen) !important; box-shadow: 3px 3px 0 var(--marker) !important;
}}
#stage button.secondary {{
    background: var(--sheet) !important; color: var(--ink) !important;
    border: 1px solid var(--rule) !important; border-radius: 2px !important;
    font-size: 13.5px !important;
}}
#actions {{ gap: 12px !important; margin: 4px 0 8px; }}
#actions > * {{ flex: 0 0 auto !important; min-width: 0 !important; }}

/* ----------------------------------------------------- question paper tab */
.lede {{
    font-size: 14.5px; line-height: 1.6; color: var(--ink-2);
    max-width: 62ch; margin: 18px 0 16px;
}}
#paper-drop {{ max-width: 520px; }}
#paper-drop .block {{ border: 2px dashed var(--rule) !important; border-radius: 3px !important; }}

ol.paper-qs {{ list-style: none; margin: 18px 0 20px; padding: 0; max-width: 70ch; }}
ol.paper-qs li {{
    display: flex; gap: 12px; align-items: baseline;
    padding: 8px 0; border-bottom: 1px solid var(--rule);
    font-size: 13.5px; line-height: 1.5; color: var(--ink-2);
}}
ol.paper-qs .qn {{
    font-weight: 700; color: var(--ink); min-width: 34px; font-variant-numeric: tabular-nums;
}}
ol.paper-qs .qt {{ flex: 1; }}
ol.paper-qs .qm {{ color: var(--ink-3); font-size: 12px; min-width: 22px; text-align: right; }}

/* -------------------------------------------------------------- study sheet */
#sheet {{ margin-top: 22px; }}
section.qa {{
    padding: 22px 0 18px; border-top: 2px solid var(--ink); max-width: 78ch;
}}
section.qa h3 {{
    font-size: 15px !important; font-weight: 700 !important; color: var(--ink) !important;
    margin: 0 0 6px !important; display: flex; gap: 10px; align-items: baseline;
}}
section.qa h3 .qm {{ font-size: 11.5px; font-weight: 500; color: var(--ink-3); }}
section.qa .qtext {{
    font-size: 14.5px; line-height: 1.55; color: var(--ink); margin: 0 0 14px;
    background: var(--marker); display: inline; box-decoration-break: clone; padding: 2px 5px;
}}
section.qa .qbody {{ margin-top: 14px; }}
section.qa .qbody p, section.qa .qbody li {{
    font-family: var(--read); font-size: 15.5px; line-height: 1.7; color: var(--ink-2);
    margin: 0 0 8px;
}}
section.qa .qbody h4 {{
    font-size: 13px; font-weight: 600; color: var(--ink); margin: 16px 0 6px;
}}
section.qa .qbody ul {{ margin: 0 0 10px; padding-left: 20px; }}
figure.fig.inline {{ max-width: 380px; margin: 14px 0 6px; }}
section.qa .from {{
    display: inline-block; margin-top: 10px; font-size: 12px;
    color: var(--pen); text-decoration: none; border-bottom: 1px solid currentColor;
}}

/* ----------------------------------------------------------------- sources */
.whence {{
    margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--rule);
    font-size: 13px; line-height: 1.7; color: var(--ink-3); max-width: 66ch;
}}
.whence a {{ color: var(--pen); text-decoration: none; border-bottom: 1px solid currentColor; }}
.whence .run {{ display: block; margin-top: 6px; font-size: 11.5px; color: var(--ink-3); }}

.thinking {{ font-size: 14px; color: var(--ink-3); }}

:focus-visible {{ outline: 2px solid var(--pen) !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
}}
@media (max-width: 900px) {{
    #scope {{ min-height: auto; border-right: none; border-bottom: 1px solid var(--rule); }}
    #stage {{ padding: 20px 18px 32px; }}
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
            label += f"  {module['title'][:36]}"
        # Say plainly which modules have nothing to search yet
        if not module["indexed_notes"]:
            label += "  (empty)"
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
        f"<b>{len(subjects)}</b> subject{'s' if len(subjects) != 1 else ''}, "
        f"<b>{sections}</b> searchable sections. Matching by {search}."
    )


# ----------------------------------------------------------------- rendering
def empty_state() -> str:
    return (
        "<div class='paper-note'>"
        "<h2>Answers from your own notes, and nothing else.</h2>"
        "<p>Ask the way a VTU paper asks — “Explain dual-mode operation with a neat diagram”, "
        "“Differentiate between paging and segmentation”. You get the answer, the diagram to "
        "redraw, and the page of your notes it came from.</p>"
        f"<div class='index'>{index_summary()}</div>"
        "</div>"
    )


def format_question(question: str) -> str:
    return f"<p class='asked'><span>{html.escape(question)}</span></p>"


def format_figures(figures: list[dict[str, Any]]) -> str:
    if not figures:
        return ""
    blocks = ["<p class='rail-title'>Diagrams to redraw</p><div class='figures strip'>"]
    for figure in figures:
        url = f"{PUBLIC_API}{figure['url']}"
        caption = figure.get("caption") or (
            f"Scanned page {figure['page']}"
            if figure.get("kind") == "page"
            else f"Diagram, page {figure['page']}"
        )
        where = f"{figure['subject_code']} module {figure['module_number']}, page {figure['page']}"
        blocks.append(
            f"<figure class='fig'><a href='{url}' target='_blank' rel='noopener'>"
            f"<img src='{url}' alt='{html.escape(caption)}' loading='lazy'></a>"
            f"<figcaption><b>{html.escape(caption)}</b>{html.escape(where)}</figcaption></figure>"
        )
    blocks.append("</div>")
    return "".join(blocks)


def format_whence(data: dict[str, Any]) -> str:
    notes = data.get("notes") or []
    if not notes and data.get("in_scope") is False:
        return "<p class='whence'>Nothing was searched — this looks outside the syllabus.</p>"
    if not notes:
        return ""

    links = []
    for note in notes:
        pages = note.get("pages") or []
        anchor = f"#page={pages[0]}" if pages else ""
        page_text = ""
        if pages:
            page_text = (
                f", page {pages[0]}"
                if len(pages) == 1
                else f", pages {', '.join(map(str, pages[:6]))}"
            )
        links.append(
            f"<a href='{PUBLIC_API}{note['url']}{anchor}' target='_blank' rel='noopener'>"
            f"{note['subject_code']} module {note['module_number']} notes</a>{page_text}"
        )

    run = []
    if data.get("rewritten_queries"):
        run.append(f"searched again as “{html.escape(data['rewritten_queries'][-1])}”")
    if data.get("search_mode"):
        run.append(f"{data['search_mode']} search")
    run.append("from cache" if data.get("cached") else f"{data.get('latency_ms', 0) / 1000:.1f}s")

    return (
        f"<p class='whence'>Taken from {'; '.join(links)}"
        f"<span class='run'>{' · '.join(run)}</span></p>"
    )


# -------------------------------------------------------------------- asking
async def ask(
    question: str,
    semester: str,
    subject_code: str | None,
    modules: list[int],
    mode_label: str,
) -> AsyncIterator[tuple]:
    asked = (question or "").strip()
    if len(asked) < 3:
        yield (gr.skip(),) * 5
        return

    yield format_question(asked), "<p class='thinking'>Reading your notes…</p>", "", "", ""

    payload: dict[str, Any] = {
        "question": asked,
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
        logger.warning("API unreachable: %s", exc)
        yield (
            format_question(asked),
            f"The API at {API} isn't responding. Check that the stack is running.",
            "",
            "",
            "",
        )
        return

    if response.status_code >= 400:
        detail = response.json().get("detail", "") if response.content else ""
        yield format_question(asked), f"That didn't work. {detail}", "", "", ""
        return

    data = response.json()
    yield (
        format_question(asked),
        data["answer"],
        format_figures(data.get("figures") or []),
        format_whence(data),
        "",
    )


# ---------------------------------------------------------------------- app
STARTERS = [
    "Explain dual-mode operation with a neat diagram",
    "What is a system call? List its types",
    "Differentiate between monolithic and microkernel structures",
]

# Clicking a starter fills the ask box — plain JS, no round trip to the server
CHIP_JS = """() => {
    document.querySelectorAll('.chips button').forEach((chip) => {
        chip.addEventListener('click', () => {
            const box = document.querySelector('#ask textarea');
            box.value = chip.textContent;
            box.dispatchEvent(new Event('input', { bubbles: true }));
            box.focus();
        });
    });
}"""


def read_paper_file(path: str | None) -> tuple[list[dict[str, Any]], str, str]:
    """Sends the uploaded paper to the API and lists the questions it found."""
    if not path:
        return [], "", ""
    name = Path(path).name
    try:
        with open(path, "rb") as handle:
            response = httpx.post(
                f"{API}/api/v1/papers/extract",
                files={"file": (name, handle, "application/octet-stream")},
                timeout=600,
            )
    except httpx.HTTPError as exc:
        logger.warning("Paper upload failed: %s", exc)
        return [], f"<p class='thinking'>Couldn't reach the API: {exc}</p>", ""

    if response.status_code >= 400:
        detail = response.json().get("detail", "") if response.content else ""
        return [], f"<p class='thinking'>{html.escape(str(detail))}</p>", ""

    data = response.json()
    questions = data["questions"]
    if not questions:
        return (
            [],
            "<p class='thinking'>No questions found in that file. VTU papers number their "
            "parts like “Q.1 a.” — a clearer scan usually fixes this.</p>",
            "",
        )
    return questions, format_paper_questions(questions), data.get("title", "Question paper")


def format_paper_questions(questions: list[dict[str, Any]]) -> str:
    marks = sum(q.get("marks") or 0 for q in questions)
    rows = "".join(
        f"<li><span class='qn'>{html.escape(q['number'])}</span>"
        f"<span class='qt'>{html.escape(q['text'])}</span>"
        f"<span class='qm'>{q['marks'] or ''}</span></li>"
        for q in questions
    )
    return (
        f"<p class='rail-title'>{len(questions)} questions found"
        f"{f' · {marks} marks' if marks else ''}</p>"
        f"<ol class='paper-qs'>{rows}</ol>"
    )


def _sheet_figure(figure: dict[str, Any]) -> str:
    url = f"{PUBLIC_API}{figure['url']}"
    caption = figure.get("caption") or f"Diagram, page {figure['page']}"
    return (
        f"<figure class='fig inline'><a href='{url}' target='_blank' rel='noopener'>"
        f"<img src='{url}' alt='diagram' loading='lazy'></a>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _answer_block(question: dict[str, Any], data: dict[str, Any]) -> str:
    """One question and its answer, laid out as a revision sheet entry."""
    marks = f"<span class='qm'>{question['marks']} marks</span>" if question.get("marks") else ""
    figures = "".join(_sheet_figure(f) for f in data.get("figures") or [])
    notes = data.get("notes") or []
    where = ""
    if notes:
        note = notes[0]
        pages = note.get("pages") or []
        anchor = f"#page={pages[0]}" if pages else ""
        where = (
            f"<a class='from' href='{PUBLIC_API}{note['url']}{anchor}' target='_blank' "
            f"rel='noopener'>{note['subject_code']} module {note['module_number']} notes</a>"
        )
    body = md_to_html(data.get("answer", ""))
    return (
        f"<section class='qa'><h3>Q{html.escape(question['number'])}{marks}</h3>"
        f"<p class='qtext'>{html.escape(question['text'])}</p>"
        f"<div class='qbody'>{body}</div>"
        f"{f'<div class=figures>{figures}</div>' if figures else ''}"
        f"{where}</section>"
    )


_MD_HEADING = re.compile(r"^#{1,6}\s+")
_MD_BULLET = re.compile(r"^[-*•+o]\s+")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")


def md_to_html(text: str) -> str:
    """Just enough markdown for answers: headings, bullets, bold."""
    out: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _MD_BOLD.sub(r"<strong>\1</strong>", html.escape(line))
        if _MD_HEADING.match(line):
            close_list()
            out.append(f"<h4>{_MD_HEADING.sub('', line)}</h4>")
        elif _MD_BULLET.match(line):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_MD_BULLET.sub('', line)}</li>")
        else:
            close_list()
            out.append(f"<p>{line}</p>")
    close_list()
    return "".join(out)


async def solve_paper(
    questions: list[dict[str, Any]],
    title: str,
    semester: str,
    subject_code: str | None,
    modules: list[int],
) -> AsyncIterator[tuple]:
    """Answers each question in turn so the sheet fills in as it goes."""
    if not questions:
        yield gr.skip(), gr.skip(), gr.skip()
        return

    filters: dict[str, Any] = {
        "branch": settings.default_branch,
        "scheme": settings.default_scheme,
        "module_numbers": modules or [],
    }
    if semester != ALL:
        filters["semester"] = int(semester)
    if subject_code and subject_code != ALL:
        filters["subject_code"] = subject_code

    blocks: list[str] = []
    async with httpx.AsyncClient(timeout=600) as client:
        for index, question in enumerate(questions, start=1):
            progress = (
                f"<p class='thinking'>Answering {index} of {len(questions)} — "
                f"Q{html.escape(question['number'])}…</p>"
            )
            yield "".join(blocks) + progress, gr.update(interactive=False), gr.skip()

            try:
                response = await client.post(
                    f"{API}/api/v1/ask", json={"question": question["text"], **filters}
                )
                data = response.json() if response.status_code < 400 else {}
            except httpx.HTTPError as exc:
                logger.warning("Question %s failed: %s", question["number"], exc)
                data = {}

            if not data.get("answer"):
                data = {"answer": "_Couldn't answer this one from the indexed notes._"}
            blocks.append(_answer_block(question, data))
            yield "".join(blocks), gr.update(interactive=False), gr.skip()

    # Everything is answered and cached, so the PDF pass is quick
    pdf_path = await build_paper_pdf(questions, title, filters)
    yield (
        "".join(blocks),
        gr.update(interactive=True),
        gr.update(value=pdf_path, visible=bool(pdf_path)),
    )


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
    path = Path(tempfile.gettempdir()) / f"{safe} — answers.pdf"
    path.write_bytes(response.content)
    return str(path)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VTU Exam Buddy", fill_width=True) as demo:
        gr.HTML(
            "<div id='masthead'>"
            "<span class='wordmark'>VTU <b>Exam Buddy</b></span>"
            f"<span class='tag'>{settings.default_branch.upper()} · "
            f"{settings.default_scheme} scheme</span>"
            "</div>"
        )

        with gr.Row(elem_id="shell", equal_height=False):
            # ------------------------------------------------------ scope rail
            with gr.Column(scale=2, min_width=225, elem_id="scope"):
                semester = gr.Dropdown(
                    [ALL] + [str(i) for i in range(1, 9)], value="3", label="Semester"
                )
                subject = gr.Dropdown([(ALL, ALL)] + load_subjects("3"), value=ALL, label="Subject")
                modules = gr.CheckboxGroup([], label="Modules")
                mode = gr.Radio(
                    list(MODES),
                    value="Careful",
                    label="Answering",
                    info="Careful checks the question is on-syllabus and searches again when "
                    "the results are weak. Quick answers in one pass.",
                )

            # ----------------------------------------------------------- stage
            with gr.Column(scale=8, elem_id="stage"), gr.Tabs():
                with gr.Tab("One question"):
                    question = gr.Textbox(
                        placeholder="Ask a question from your notes",
                        show_label=False,
                        submit_btn=True,
                        elem_id="ask",
                        max_lines=3,
                    )
                    gr.HTML(
                        "<div class='chips'>"
                        + "".join(
                            f"<button type='button'>{html.escape(s)}</button>" for s in STARTERS
                        )
                        + "</div>"
                    )
                    asked = gr.HTML("")
                    answer = gr.Markdown(empty_state(), elem_id="answer")
                    figure_html = gr.HTML("", elem_id="figure-slot")
                    whence = gr.HTML("")

                with gr.Tab("Whole question paper"):
                    gr.HTML(
                        "<p class='lede'>Upload a question paper — a PDF or a photo of one — "
                        "and get every question answered from your notes, ready to print.</p>"
                    )
                    paper_file = gr.File(
                        label="Question paper (PDF or photo)",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
                        elem_id="paper-drop",
                    )
                    found = gr.HTML("")
                    with gr.Row(elem_id="actions"):
                        solve_btn = gr.Button(
                            "Answer every question", variant="primary", scale=0, min_width=210
                        )
                        download = gr.DownloadButton(
                            "Download as PDF", visible=False, scale=0, min_width=180
                        )
                    sheet = gr.HTML("", elem_id="sheet")
                    questions_state = gr.State([])
                    title_state = gr.State("Question paper")

        # ------------------------------------------------------------ wiring
        question.submit(
            ask,
            [question, semester, subject, modules, mode],
            [asked, answer, figure_html, whence, question],
        )

        def on_semester(sem: str):
            return (
                gr.update(choices=[(ALL, ALL)] + load_subjects(sem), value=ALL),
                gr.update(choices=[], value=[]),
            )

        semester.change(on_semester, semester, [subject, modules])
        subject.change(
            lambda code: gr.update(choices=load_modules(code), value=[]), subject, modules
        )
        paper_file.change(
            read_paper_file, paper_file, [questions_state, found, title_state], show_progress="full"
        )
        solve_btn.click(
            solve_paper,
            [questions_state, title_state, semester, subject, modules],
            [sheet, solve_btn, download],
        )

        demo.load(None, None, None, js=CHIP_JS)
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    build_app().launch(
        server_name="0.0.0.0",
        enable_monitoring=False,
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
    )
