"""Picks the diagrams that belong with an answer.

Page proximity alone is too blunt: a single page often carries two diagrams from
unrelated topics, and a retrieved chunk spanning that page would drag both in.
So candidates are gathered by page, then ranked by what the diagram itself says —
its caption and the labels read out of the image — against the question.
"""

import re

from vtu_rag.schemas.ask import FigureOut, Source

# How many top-ranked passages to use for figures when the answer cites nothing
FALLBACK_SOURCES = 2

# A figure must reach this share of the best score to be shown alongside it
RELATIVE_CUTOFF = 0.5

# Exam phrasing and filler: present in every question, so useless for matching.
# "diagram" especially — every question asking for one would match every figure.
_NOISE_WORDS = frozenset(
    """
    define definition explain explanation describe description discuss write note notes short
    brief briefly list state give given neat diagram diagrams figure figures draw drawing sketch
    with and the for its their what which how why when where does are can you your please
    example examples question answer marks module semester scheme university
    """.split()  # noqa: SIM905 - a readable word list beats a 50-item literal
)

_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")


def _terms(text: str | None) -> set[str]:
    """Topic words, with hyphenated compounds counted whole and in parts.

    A question says "dual-mode" where the diagram is labelled "mode bit", so both
    forms have to be on the table.
    """
    if not text:
        return set()
    terms: set[str] = set()
    for token in _WORD_RE.findall(text.lower()):
        for part in {token, *token.split("-")}:
            if len(part) < 3 or part in _NOISE_WORDS:
                continue
            terms.add(part)
            if (singular := _singular(part)) is not None:
                terms.add(singular)
    return terms


def _singular(word: str) -> str | None:
    """Crude plural folding, so "processes" meets "process" and "policies" "policy"."""
    if len(word) < 5:
        return None
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("sses") or word.endswith("ches") or word.endswith("shes"):
        return word[:-2]
    if word.endswith("es") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return None


def generic_terms(sources: list[Source]) -> set[str]:
    """Words that carry no information inside this subject.

    Every diagram in BCS303 Operating Systems is "about operating systems", so
    those words would score every figure equally and drown out the real topic.
    """
    terms: set[str] = set()
    for source in sources:
        terms |= _terms(source.subject_name)
        terms.add(source.subject_code.lower())
    return terms


def relevant_sources(sources: list[Source], fallback: int = FALLBACK_SOURCES) -> list[Source]:
    """The passages used for *ordering* figures when none can be scored."""
    with_pages = [s for s in sources if s.page_start is not None]
    cited = [s for s in with_pages if s.cited]
    return cited or with_pages[:fallback]


def page_spans(sources: list[Source]) -> list[tuple[int, int, int]]:
    """Candidate pages: every retrieved passage that knows where it came from.

    Deliberately wider than the passages an answer leaned on — the right diagram
    often sits a page away from the best-scoring text (the explanation on page 10,
    the figure on page 11). Scoring, not page proximity, decides what is shown.
    """
    return [
        (s.note_id, s.page_start, s.page_end or s.page_start)
        for s in sources
        if s.page_start is not None
    ]


def figure_score(figure, question_terms: set[str]) -> float:  # noqa: ANN001 - ORM model
    """How much of the question this diagram's own text accounts for."""
    if not question_terms:
        return 0.0
    figure_terms = _terms(figure.caption) | _terms(getattr(figure, "label_text", None))
    if not figure_terms:
        return 0.0
    return len(figure_terms & question_terms) / len(question_terms)


def passage_score(
    figure,  # noqa: ANN001 - ORM model
    question_terms: set[str],
    passages: list[tuple[int, int, int, str]],
) -> float:
    """How well the *text around* this diagram answers the question.

    Diagrams rarely name their own topic — a symmetric-multiprocessing figure is
    just labelled "CPU registers cache" — so the passage the figure sits in is the
    second opinion, used only when no diagram's labels match.
    """
    if not question_terms:
        return 0.0
    best = 0.0
    for note_id, start, end, text in passages:
        if note_id != figure.note_id or not (start <= figure.page <= end):
            continue
        overlap = len(_terms(text) & question_terms) / len(question_terms)
        best = max(best, overlap)
    return best


def select_figures(
    figures: list,
    sources: list[Source],
    limit: int,
    question: str = "",
    passages: list[tuple[int, int, int, str]] | None = None,
    fallback: int = FALLBACK_SOURCES,
) -> list[FigureOut]:
    """Ranks candidates by what the diagram says, then by the text it sits in."""
    if not figures:
        return []

    question_terms = _terms(question) - generic_terms(sources)
    scored = [(figure_score(f, question_terms), f) for f in figures]
    best = max((score for score, _ in scored), default=0.0)

    if best == 0.0 and question_terms and passages:
        # No diagram names the topic; ask which of them sits in text that does
        scored = [(passage_score(f, question_terms, passages), f) for f in figures]
        best = max((score for score, _ in scored), default=0.0)

    readable = any(f.caption or getattr(f, "label_text", None) for f in figures)

    if best > 0:
        # Keep only diagrams that speak to the question about as well as the best one
        candidates = [(score, f) for score, f in scored if score >= best * RELATIVE_CUTOFF]
        candidates.sort(key=lambda pair: (-pair[0], pair[1].page, pair[1].index))
        ordered = [f for _, f in candidates]
    elif question_terms and readable:
        # We could read these diagrams, and neither they nor their surrounding text
        # is about the question. A wrong diagram is worse than none — a student
        # would copy it into the exam.
        return []
    else:
        # Nothing readable inside any diagram: fall back to the ranking of the
        # passage that pulled each one in, then page order
        rank: dict[tuple[int, int, int], int] = {}
        for source in relevant_sources(sources, fallback):
            key = (source.note_id, source.page_start, source.page_end or source.page_start)
            rank.setdefault(key, source.index)

        def sort_key(figure) -> tuple[int, int, int]:  # noqa: ANN001 - ORM model
            position = min(
                (
                    index
                    for (note_id, start, end), index in rank.items()
                    if note_id == figure.note_id and start <= figure.page <= end
                ),
                default=999,
            )
            return (position, figure.page, figure.index)

        ordered = sorted(figures, key=sort_key)

    seen: set[str] = set()
    picked: list[FigureOut] = []
    for figure in ordered:
        if figure.sha256 in seen:  # the same diagram repeated across notes
            continue
        seen.add(figure.sha256)
        picked.append(FigureOut.from_figure(figure))
        if len(picked) >= limit:
            break
    return picked
