"""Picks the diagrams that belong with an answer.

A figure earns its place when it sits on a page the answer drew from: the
student is reading that passage, so that is the diagram they would be asked to
draw in the exam.

Cited sources are preferred. Small local models don't always emit [n] markers
though, and a missing citation is no reason to hide the diagrams — so when
nothing is cited we fall back to the best-ranked retrieved passages.
"""

from vtu_rag.schemas.ask import FigureOut, Source

# How many top-ranked passages to use for figures when the answer cites nothing
FALLBACK_SOURCES = 2


def relevant_sources(sources: list[Source], fallback: int = FALLBACK_SOURCES) -> list[Source]:
    with_pages = [s for s in sources if s.page_start is not None]
    cited = [s for s in with_pages if s.cited]
    return cited or with_pages[:fallback]


def page_spans(
    sources: list[Source], fallback: int = FALLBACK_SOURCES
) -> list[tuple[int, int, int]]:
    """(note_id, first page, last page) for each source the answer drew on."""
    return [
        (s.note_id, s.page_start, s.page_end or s.page_start)
        for s in relevant_sources(sources, fallback)
    ]


def select_figures(
    figures: list, sources: list[Source], limit: int, fallback: int = FALLBACK_SOURCES
) -> list[FigureOut]:
    """Orders figures by the rank of the passage that pulled them in, then by page."""
    if not figures:
        return []

    rank: dict[tuple[int, int, int], int] = {}
    for source in relevant_sources(sources, fallback):
        key = (source.note_id, source.page_start, source.page_end or source.page_start)
        rank.setdefault(key, source.index)

    def sort_key(figure) -> tuple[int, int, int]:  # noqa: ANN001 - ORM model
        best = min(
            (
                index
                for (note_id, start, end), index in rank.items()
                if note_id == figure.note_id and start <= figure.page <= end
            ),
            default=999,
        )
        return (best, figure.page, figure.index)

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
