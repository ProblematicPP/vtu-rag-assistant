"""Multi-part questions: "Distinguish between (i) X and Y (ii) P and Q".

A VTU question often bundles two or three sub-questions. Answered as one, they
compete for the same eight excerpts and the same grading call, so a section
that only one part needs gets judged irrelevant to the whole and dropped — and
every diagram lands after the last part, far from the part it illustrates.

Each part is therefore asked as a question of its own, and the answers are put
back together in order, each part followed by its own diagrams.
"""

import re
from dataclasses import dataclass

from vtu_rag.schemas.ask import AskResponse, FigureOut, NoteRef, PartAnswer, notes_from_sources

# Numbering styles VTU papers use for sub-parts, each tried separately so
# "(i)" never pairs with "b)"
_SCHEMES = [
    ["i", "ii", "iii", "iv", "v", "vi"],
    ["a", "b", "c", "d", "e", "f"],
    ["1", "2", "3", "4", "5", "6"],
]

# "the following terms", "the following:" — the stem's placeholder for each part
_FOLLOWING = re.compile(
    r"\bthe\s+following(?:\s+(?:terms?|concepts?|topics?|questions?|systems?|pairs?))?\s*:?",
    re.IGNORECASE,
)

MIN_PART_CHARS = 3


@dataclass(frozen=True)
class QuestionPart:
    label: str  # "i", "ii", "a"
    text: str  # the part as printed: "Multiprogramming and Multitasking"
    question: str  # what gets searched: "Distinguish between Multiprogramming and Multitasking"


def _marker(label: str) -> re.Pattern[str]:
    # "(ii)", "ii)", "ii." — at the start or after whitespace, never inside a word
    return re.compile(rf"(?:(?<=\s)|^)\(?{re.escape(label)}[).](?=\s)", re.IGNORECASE)


def _find_parts(question: str, labels: list[str]) -> list[tuple[str, int, int]] | None:
    """(label, marker start, text start) for each consecutive label present, in order."""
    found: list[tuple[str, int, int]] = []
    position = 0
    for label in labels:
        match = _marker(label).search(question, position)
        if match is None:
            break
        found.append((label, match.start(), match.end()))
        position = match.end()
    return found if len(found) >= 2 else None


def _standalone(stem: str, text: str) -> str:
    stem = stem.strip().rstrip(":").strip()
    if not stem:
        return text
    if _FOLLOWING.search(stem):
        # "Distinguish between the following terms." + "X and Y"
        #   → "Distinguish between X and Y"
        return " ".join(_FOLLOWING.sub(text, stem, count=1).split()).rstrip(".")
    return f"{stem.rstrip('.')}: {text}"


def split_parts(question: str) -> list[QuestionPart]:
    """The sub-questions, or [] when the question is a single one."""
    for labels in _SCHEMES:
        found = _find_parts(question, labels)
        if not found:
            continue
        stem = question[: found[0][1]]
        parts: list[QuestionPart] = []
        for index, (label, _start, text_start) in enumerate(found):
            end = found[index + 1][1] if index + 1 < len(found) else len(question)
            text = " ".join(question[text_start:end].split()).strip(" ,;")
            if len(text) < MIN_PART_CHARS:
                return []
            parts.append(QuestionPart(label, text, _standalone(stem, text)))
        return parts
    return []


def marks_per_part(marks: int | None, parts: int) -> int | None:
    """A 10-mark question in two parts is two 5-mark answers."""
    if not marks or parts < 1:
        return marks
    return max(2, round(marks / parts))


def combine(question: str, answered: list[tuple[QuestionPart, AskResponse]]) -> AskResponse:
    """One response for the whole question, keeping each part's answer and diagrams."""
    first = answered[0][1]
    parts: list[PartAnswer] = []
    shown: set[int] = set()
    sources = []
    for part, response in answered:
        # A diagram belongs with the first part it illustrates, not repeated under each
        figures: list[FigureOut] = [f for f in response.figures if f.id not in shown]
        shown.update(f.id for f in figures)
        parts.append(
            PartAnswer(
                label=part.label,
                question=part.text,
                answer=response.answer,
                resolved_question=response.resolved_question,
                figures=figures,
                notes=response.notes,
            )
        )
        sources.extend(response.sources)

    text = "\n\n".join(f"**({p.label}) {p.question}**\n\n{p.answer}" for p in parts)
    notes: list[NoteRef] = notes_from_sources(sources)
    update: dict = {
        "question": question,
        "resolved_question": None,
        "answer": text,
        "sources": sources,
        "notes": notes,
        "figures": [f for p in parts for f in p.figures],
        "parts": parts,
        "cached": all(r.cached for _, r in answered),
        "latency_ms": round(sum(r.latency_ms or 0 for _, r in answered), 1),
    }
    # The agent's fields add up across parts the same way
    if hasattr(first, "steps"):
        responses = [r for _, r in answered]
        update |= {
            "in_scope": any(r.in_scope for r in responses),
            "rewritten_queries": [q for r in responses for q in r.rewritten_queries],
            "retrieval_attempts": sum(r.retrieval_attempts for r in responses),
            "steps": [s for r in responses for s in r.steps],
        }
    return first.model_copy(update=update)
