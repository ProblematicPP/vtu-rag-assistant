import hashlib
from dataclasses import dataclass

from vtu_rag.ingestion.figures import (
    ExtractedFigure,
    FigureConfig,
    FigureKind,
    _drop_repeated,
    _looks_decorative,
    _page_captions,
)
from vtu_rag.schemas.ask import Source
from vtu_rag.services.rag.figures import page_spans, select_figures

CONFIG = FigureConfig()


def figure(
    page: int, index: int = 0, width: int = 600, height: int = 400, data: bytes = b"x" * 9000
):
    return ExtractedFigure(
        page=page,
        index=index,
        data=data,
        extension=".png",
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
    )


class TestFiltering:
    def test_a_real_diagram_is_kept(self):
        assert _looks_decorative(figure(1, width=964, height=449), CONFIG) is False

    def test_tiny_bullet_icons_are_dropped(self):
        assert _looks_decorative(figure(1, width=24, height=24), CONFIG) is True

    def test_horizontal_rules_are_dropped(self):
        assert _looks_decorative(figure(1, width=1200, height=8), CONFIG) is True

    def test_near_empty_images_are_dropped(self):
        assert _looks_decorative(figure(1, data=b"x" * 400), CONFIG) is True

    def test_college_logo_on_every_page_is_dropped(self):
        logo = b"logo-bytes" * 500
        figures = [figure(page, data=logo) for page in range(1, 11)]
        figures.append(figure(3, index=1, data=b"real-diagram" * 800))
        kept = _drop_repeated(figures, page_count=10, config=CONFIG)
        assert len(kept) == 1
        assert kept[0].page == 3

    def test_a_diagram_repeated_twice_survives(self):
        shared = b"diagram" * 900
        figures = [figure(1, data=shared), figure(9, data=shared)]
        assert len(_drop_repeated(figures, page_count=30, config=CONFIG)) == 2


class TestCaptions:
    def test_finds_figure_captions(self):
        text = "some prose\nFig. 1.2 Layered operating system\nmore prose"
        assert _page_captions(text) == ["Fig. 1.2 Layered operating system"]

    def test_accepts_several_spellings(self):
        text = "Figure 3 - Process state diagram\nDiagram 2: Paging hardware"
        assert len(_page_captions(text)) == 2

    def test_ignores_ordinary_sentences(self):
        assert _page_captions("The figure shows how paging works in detail.") == []


def source(index: int, note_id: int, page: int, cited: bool) -> Source:
    return Source(
        index=index,
        subject_code="BCS303",
        subject_name="Operating Systems",
        semester=3,
        module_number=1,
        module_title=None,
        note_id=note_id,
        note_title="module1",
        source_uri="cse/2022/sem3/BCS303/module1.pdf",
        section_heading=None,
        page_start=page,
        page_end=page,
        chunk_id=index,
        score=1.0,
        snippet="...",
        cited=cited,
    )


@dataclass
class FakeSubject:
    code: str = "BCS303"


@dataclass
class FakeModule:
    number: int = 1
    subject: FakeSubject = None


@dataclass
class FakeNote:
    module: FakeModule = None


@dataclass
class FakeFigure:
    id: int
    note_id: int
    page: int
    index: int = 0
    kind: str = FigureKind.FIGURE
    caption: str | None = None
    width: int = 600
    height: int = 400
    sha256: str = "abc"
    note: FakeNote = None

    def __post_init__(self):
        self.note = FakeNote(module=FakeModule(subject=FakeSubject()))


class TestSelection:
    def test_cited_pages_win_when_the_answer_cites(self):
        sources = [source(1, 7, 5, cited=True), source(2, 7, 19, cited=False)]
        assert page_spans(sources) == [(7, 5, 5)]

    def test_uncited_answers_fall_back_to_the_top_passages(self):
        # Small models sometimes skip [n] markers; diagrams should still show
        sources = [source(i, 7, i * 3, cited=False) for i in range(1, 6)]
        assert page_spans(sources, fallback=2) == [(7, 3, 3), (7, 6, 6)]

    def test_sources_without_pages_are_skipped(self):
        markdown_note = source(1, 7, 5, cited=True).model_copy(
            update={"page_start": None, "page_end": None}
        )
        assert page_spans([markdown_note]) == []

    def test_figures_follow_the_rank_of_the_source_that_cited_them(self):
        sources = [source(1, 7, 19, cited=True), source(2, 7, 5, cited=True)]
        figures = [
            FakeFigure(id=10, note_id=7, page=5, sha256="h10"),
            FakeFigure(id=11, note_id=7, page=19, sha256="h11"),
        ]
        picked = select_figures(figures, sources, limit=4)
        # Source [1] cited page 19, so that page's diagram comes first
        assert [f.id for f in picked] == [11, 10]

    def test_duplicate_images_are_shown_once(self):
        sources = [source(1, 7, 5, cited=True)]
        figures = [
            FakeFigure(id=10, note_id=7, page=5, sha256="same"),
            FakeFigure(id=11, note_id=7, page=5, index=1, sha256="same"),
        ]
        assert len(select_figures(figures, sources, limit=4)) == 1

    def test_limit_is_respected(self):
        sources = [source(1, 7, 5, cited=True)]
        figures = [FakeFigure(id=i, note_id=7, page=5, index=i, sha256=f"h{i}") for i in range(10)]
        assert len(select_figures(figures, sources, limit=3)) == 3

    def test_no_figures_means_no_output(self):
        assert select_figures([], [source(1, 7, 5, cited=True)], limit=4) == []
