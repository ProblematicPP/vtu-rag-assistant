"""Student-facing chat UI. Talks to the FastAPI service over HTTP."""

import logging
import os
from typing import Any

import gradio as gr
import httpx

from vtu_rag.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
API = settings.api_base_url.rstrip("/")
# Images are fetched by the browser, not by this process
PUBLIC_API = settings.public_api_base_url.rstrip("/")
ALL = "All"
MODES = {"Agentic (guardrails + self-correction)": "agentic-ask", "Quick answer": "ask"}

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#title h1 { margin-bottom: 0; }
#title p { margin-top: 4px; opacity: 0.75; }
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
        label = f"M{m['number']}" + (f": {m['title']}" if m["title"] else "")
        if not m["indexed_notes"]:
            label += " (no notes yet)"
        choices.append((label, m["number"]))
    return choices


# ----------------------------------------------------------------- rendering
def format_notes(notes: list[dict[str, Any]]) -> str:
    """Links the whole source note. PDF viewers honour #page=N, so it opens where it matters."""
    if not notes:
        return ""
    lines = ["", "---", "**From your notes**"]
    for note in notes:
        module = f"Module {note['module_number']}"
        if note.get("module_title"):
            module += f" ({note['module_title']})"
        pages = note.get("pages") or []
        anchor = f"#page={pages[0]}" if pages else ""
        where = ""
        if pages:
            where = (
                f" — page {pages[0]}"
                if len(pages) == 1
                else f" — pages {', '.join(map(str, pages))}"
            )
        lines.append(
            f"- [{note['subject_code']} {note['subject_name']} · {module} — {note['title']}]"
            f"({PUBLIC_API}{note['url']}{anchor}){where}"
        )
    return "\n".join(lines)


def format_agent_trace(data: dict[str, Any]) -> str:
    bits = []
    if data.get("rewritten_queries"):
        rewrites = "; ".join(f"“{q}”" for q in data["rewritten_queries"])
        bits.append(f"searched again as {rewrites}")
    if data.get("retrieval_attempts"):
        bits.append(f"{data['retrieval_attempts']} retrieval(s)")
    if data.get("search_mode"):
        bits.append(f"{data['search_mode']} search")
    if data.get("cached"):
        bits.append("cached")
    if data.get("latency_ms") and not data.get("cached"):
        bits.append(f"{data['latency_ms'] / 1000:.1f}s")
    return f"\n\n<sub>{' · '.join(bits)}</sub>" if bits else ""


# ------------------------------------------------------------------ chat fn
def format_figures(figures: list[dict[str, Any]]) -> str:
    """Diagrams are the part students copy into the answer sheet, so show them inline."""
    if not figures:
        return ""
    blocks = ["\n\n---\n\n**Diagrams from these notes**\n"]
    for figure in figures:
        caption = figure.get("caption") or (
            f"Scanned page {figure['page']}"
            if figure.get("kind") == "page"
            else f"Diagram on page {figure['page']}"
        )
        where = f"{figure['subject_code']} · Module {figure['module_number']} · p. {figure['page']}"
        blocks.append(f"![{caption}]({PUBLIC_API}{figure['url']})\n\n*{caption} — {where}*\n")
    return "\n".join(blocks)


async def respond(
    message: str,
    history: list[dict[str, Any]],
    semester: str,
    subject_code: str | None,
    modules: list[int],
    mode_label: str,
) -> str:
    question = (message or "").strip()
    if len(question) < 3:
        return "Please type a question about your notes."

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
        return f"⚠️ Could not reach the API at {API}: {exc}"
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else ""
        return f"⚠️ The assistant is unavailable right now ({response.status_code}): {detail}"

    data = response.json()
    return (
        data["answer"]
        + format_figures(data.get("figures", []))
        + format_notes(data.get("notes", []))
        + format_agent_trace(data)
    )


# ---------------------------------------------------------------------- app
def build_app() -> gr.Blocks:
    with gr.Blocks(title="VTU Exam Buddy", fill_height=True) as demo:
        gr.Markdown(
            "# 📘 VTU Exam Buddy\n"
            f"Ask questions about your {settings.default_branch.upper()} "
            f"{settings.default_scheme}-scheme notes. Answers cite the subject, module and note "
            "they came from.",
            elem_id="title",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                semester = gr.Dropdown(
                    [ALL] + [str(i) for i in range(1, 9)], value="3", label="Semester"
                )
                subject = gr.Dropdown([(ALL, ALL)] + load_subjects("3"), value=ALL, label="Subject")
                modules = gr.CheckboxGroup([], label="Modules (optional)")
                mode = gr.Radio(list(MODES), value=next(iter(MODES)), label="Mode")
                gr.Markdown(
                    "<sub>Agentic mode checks the question is on-syllabus, grades the retrieved "
                    "notes and retries with a better query when they're weak.</sub>"
                )
            with gr.Column(scale=3):
                gr.ChatInterface(
                    respond,
                    chatbot=gr.Chatbot(
                        height=560,
                        placeholder="Try: *Explain dual-mode operation with an example.*",
                        buttons=["copy"],
                    ),
                    additional_inputs=[semester, subject, modules, mode],
                    # With additional inputs, each example must supply every input
                    examples=[
                        [q, "3", "BCS303", [], next(iter(MODES))]
                        for q in (
                            "What is a system call? List its types.",
                            "Differentiate between monolithic and microkernel structures.",
                            "Explain the role of interrupts in an operating system.",
                        )
                    ],
                    fill_height=True,
                )

        def on_semester(sem: str):
            return gr.update(choices=[(ALL, ALL)] + load_subjects(sem), value=ALL), gr.update(
                choices=[], value=[]
            )

        def on_subject(code: str):
            return gr.update(choices=load_modules(code), value=[])

        semester.change(on_semester, semester, [subject, modules])
        subject.change(on_subject, subject, modules)
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    build_app().launch(
        server_name="0.0.0.0",
        enable_monitoring=False,
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
    )
