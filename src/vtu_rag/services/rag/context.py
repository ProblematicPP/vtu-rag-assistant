"""Turns retrieved chunks into prompt context and tidies the answer that comes back."""

import re

from vtu_rag.schemas.ask import Source
from vtu_rag.services.search import SearchFilters, SearchHit

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A marker together with the space in front of it: "memory [1]." → "memory."
_CITATION_WITH_SPACE = re.compile(r"(?<=\S)[ \t]*\[\d+(?:\s*,\s*\d+)*\]")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_RUNS_OF_BLANK_LINES = re.compile(r"\n{3,}")

# "[the diagram shown]", "**Diagram:** The diagram shown illustrates…" — small
# models narrate the figure even when told not to. The student can already see it.
_DIAGRAM_LINE = re.compile(
    r"^[ \t]*\[?\**[ \t]*(?:the\s+)?(?:diagram|figure)\b[^\n]*\bshown\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_DIAGRAM_BRACKET = re.compile(r"\[[^\]\n]*\bdiagram\s+shown\b[^\]\n]*\]", re.IGNORECASE)

# A bullet in any of the shapes small models write: "- x", "* x", "+ x",
# " • x", "o x", "– x". "**bold**" never matches: the marker must be followed by space.
_BULLET = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+•●▪◦○‣–]|o(?=\s))[ \t]+(?P<body>\S.*)$")
_NUMBERED = re.compile(r"^(?P<indent>[ \t]*)(?P<number>\d{1,2})[.)][ \t]+(?P<body>\S.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Markers that mean "sub-point" even when the model forgets to indent them
_SUB_MARKERS = ("+", "◦", "○", "o ", "o\t")
NEST = "    "
# Point → detail → detail of the detail; anything deeper joins the last level
MAX_NESTING = 2

_FENCED_BLOCK = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
# "**Diagram:** … can be sketched as follows:", "This diagram illustrates …"
_SKETCH_TALK = re.compile(
    r"\b(?:diagram|sketch)\b[^\n]*\b(?:sketch(?:ed)?|follows|illustrat\w*|below|shown|showing)\b"
    r"|\bthis\s+diagram\b"
    r"|^\W*diagram\W*:",  # "**Diagram:** …", a block that exists only to introduce one
    re.IGNORECASE | re.MULTILINE,
)

# Keep prompts inside small local models' context windows
MAX_CONTEXT_WORDS = 3000


def build_sources(hits: list[SearchHit]) -> list[Source]:
    return [Source.from_hit(i, hit) for i, hit in enumerate(hits, start=1)]


def format_context(hits: list[SearchHit]) -> tuple[str, list[Source]]:
    """Numbers hits [1], [2], ... within a word budget.

    Returns the prompt block and the sources that actually made it in.
    """
    blocks: list[str] = []
    sources: list[Source] = []
    budget = MAX_CONTEXT_WORDS
    for source, hit in zip(build_sources(hits), hits, strict=True):
        if budget <= 0:
            break
        words = hit.text.split()
        blocks.append(f"[{source.index}] {source.label}\n{' '.join(words[:budget])}")
        sources.append(source)
        budget -= len(words)
    return "\n\n".join(blocks), sources


def cited_indices(answer: str) -> set[int]:
    found: set[int] = set()
    for group in _CITATION_RE.findall(answer):
        found.update(int(n) for n in group.split(","))
    return found


def strip_citation_markers(answer: str) -> str:
    """Removes stray [1] / [2][3] markers from an answer.

    Excerpts are numbered in the prompt, so small models sometimes cite them even
    though the prompt forbids it. Sources are presented as links to the whole
    note, which would leave those numbers pointing at nothing.
    """
    # Only the marker and the space before it go: collapsing every run of
    # spaces would also flatten list indentation and ASCII-diagram alignment
    cleaned = _CITATION_WITH_SPACE.sub("", answer)
    return _TRAILING_SPACE.sub("", cleaned).strip()


def drop_text_diagrams(answer: str) -> str:
    """Removes an ASCII sketch, and the lines introducing it, from an answer.

    For when diagrams from the notes turned out to be attached after all: the
    decision to allow a sketch is made before the answer exists, and a picture
    from the student's own notes beats a box drawing of the same thing.
    """
    blocks = re.split(r"\n\s*\n", _FENCED_BLOCK.sub("", answer))
    kept = [block for block in blocks if not _SKETCH_TALK.search(block)]
    return _RUNS_OF_BLANK_LINES.sub("\n\n", "\n\n".join(kept)).strip()


def drop_repeated_blocks(answer: str) -> str:
    """Removes paragraphs and list blocks the model has already written once.

    A small model can fall into a loop, printing the same definition and bullet
    block again and again until it runs out of tokens. Every copy after the
    first is dropped; a block only counts as repeated when it matches exactly,
    whitespace aside, so short bold headings like "**Differences**" never do.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for block in re.split(r"\n\s*\n", answer):
        key = " ".join(block.split()).lower()
        if len(key) > 40 and key in seen:
            continue
        seen.add(key)
        kept.append(block)
    return "\n\n".join(kept).strip()


def _indent_width(indent: str) -> int:
    return len(indent.replace("\t", "    "))


def tidy_markdown(answer: str) -> str:
    """Rewrites the model's lists as real Markdown lists.

    A 3B model writes sub-points as " • Definition: …" under "* **Term**:". A
    bullet character after a space is not list syntax, so rendered, every
    sub-point runs into one paragraph. Each list line becomes "- " (or "N. "),
    nested by how deep it sits (two levels at most), with a
    blank line wherever a list meets prose so neither swallows the other.
    Code blocks and tables are left exactly as written.
    """
    out: list[str] = []
    in_fence = in_list = False
    levels: list[int] = [0]  # indentation each open list level was written at
    for raw in answer.splitlines():
        line = raw.rstrip()
        if _FENCE.match(line):
            in_fence, in_list = not in_fence, False
            out.append(line)
            continue
        if in_fence or line.lstrip().startswith("|"):
            out.append(line)
            continue
        if not line.strip():
            out.append("")
            continue

        match = _BULLET.match(line) or _NUMBERED.match(line)
        if match is None:
            if in_list and out and out[-1]:
                out.append("")  # otherwise it becomes a continuation of the last item
            out.append(line.strip())
            in_list = False
            continue

        width = _indent_width(match.group("indent"))
        if not in_list:
            if out and out[-1]:
                out.append("")
            levels = [width]
        else:
            while len(levels) > 1 and width < levels[-1]:
                levels.pop()
            if width > levels[-1] and len(levels) <= MAX_NESTING:
                levels.append(width)
        depth = len(levels) - 1
        if depth == 0 and in_list and line.lstrip().startswith(_SUB_MARKERS):
            depth = 1
        number = match.groupdict().get("number")
        marker = f"{number}." if number else "-"
        out.append(f"{NEST * depth}{marker} {match.group('body').strip()}")
        in_list = True

    return _RUNS_OF_BLANK_LINES.sub("\n\n", "\n".join(out)).strip()


def drop_diagram_narration(answer: str) -> str:
    """Removes chatter about the diagram displayed beneath the answer."""
    cleaned = _DIAGRAM_BRACKET.sub("", answer)
    cleaned = _DIAGRAM_LINE.sub("", cleaned)
    return _RUNS_OF_BLANK_LINES.sub("\n\n", cleaned).strip()


def mark_cited(answer: str, sources: list[Source]) -> list[Source]:
    cited = cited_indices(answer)
    return [s.model_copy(update={"cited": s.index in cited}) for s in sources]


def scope_description(filters: SearchFilters) -> str:
    parts = []
    if filters.subject_code:
        parts.append(filters.subject_code.upper())
    if filters.module_numbers:
        parts.append("module " + ", ".join(map(str, sorted(filters.module_numbers))))
    if filters.semester:
        parts.append(f"semester {filters.semester}")
    return f" for {' '.join(parts)}" if parts else ""
