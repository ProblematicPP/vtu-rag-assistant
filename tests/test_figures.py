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
from vtu_rag.services.rag.figures import (
    _terms,
    figure_score,
    generic_terms,
    page_spans,
    relevant_sources,
    select_figures,
)

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
    label_text: str | None = None
    note: FakeNote = None

    def __post_init__(self):
        self.note = FakeNote(module=FakeModule(subject=FakeSubject()))


class TestSelection:
    def test_every_retrieved_page_is_a_candidate(self):
        # Scoring decides what is shown, so the pool stays wide
        sources = [source(1, 7, 5, cited=True), source(2, 7, 19, cited=False)]
        assert page_spans(sources) == [(7, 5, 5), (7, 19, 19)]

    def test_cited_passages_lead_the_fallback_ordering(self):
        sources = [source(1, 7, 5, cited=False), source(2, 7, 19, cited=True)]
        assert [s.index for s in relevant_sources(sources)] == [2]

    def test_uncited_answers_fall_back_to_the_top_passages(self):
        # Small models often skip [n] markers; diagrams should still show
        sources = [source(i, 7, i * 3, cited=False) for i in range(1, 6)]
        assert [s.index for s in relevant_sources(sources, fallback=2)] == [1, 2]

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


class TestRelevanceToTheQuestion:
    """The reported bug: a dual-mode question also showed a multiprocessing diagram.

    Both sit on page 10 of BCS303 module 1, so page proximity cannot separate
    them — but what is printed inside each diagram can.
    """

    QUESTION = "Define operating systems. Explain the dual-mode operation with a neat diagram."
    DUAL_MODE = FakeFigure(
        id=1, note_id=34, page=10, sha256="dual", label_text="user process kernel mode bit trap"
    )
    MULTIPROCESSING = FakeFigure(
        id=2, note_id=34, page=10, index=1, sha256="multi", label_text="CPU GPU memory"
    )

    def test_only_the_matching_diagram_is_shown(self):
        sources = [source(1, 34, 10, cited=False)]
        picked = select_figures(
            [self.MULTIPROCESSING, self.DUAL_MODE], sources, limit=3, question=self.QUESTION
        )
        assert [f.id for f in picked] == [1]

    def test_exam_phrasing_does_not_match_everything(self):
        """ "neat diagram" appears in most VTU questions; it must not score."""
        assert figure_score(self.MULTIPROCESSING, _terms(self.QUESTION)) == 0.0

    def test_a_caption_counts_as_well_as_labels(self):
        captioned = FakeFigure(
            id=3, note_id=34, page=10, sha256="cap", caption="Fig 1.5 Dual-mode operation"
        )
        picked = select_figures([captioned], [source(1, 34, 10, False)], 3, question=self.QUESTION)
        assert [f.id for f in picked] == [3]

    def test_unreadable_diagrams_fall_back_to_page_ranking(self):
        blank_a = FakeFigure(id=4, note_id=34, page=10, sha256="a")
        blank_b = FakeFigure(id=5, note_id=34, page=19, sha256="b")
        sources = [source(1, 34, 19, cited=True), source(2, 34, 10, cited=True)]
        picked = select_figures([blank_a, blank_b], sources, limit=3, question=self.QUESTION)
        # no labels anywhere, so the top-ranked passage's page wins
        assert [f.id for f in picked] == [5, 4]

    def test_a_question_of_only_filler_words_falls_back(self):
        picked = select_figures(
            [self.DUAL_MODE], [source(1, 34, 10, False)], 3, question="Explain with a neat diagram"
        )
        assert [f.id for f in picked] == [1]


class TestNoGuessing:
    """A wrong diagram gets copied into an exam answer — silence is better."""

    def test_readable_but_unrelated_diagrams_are_dropped(self):
        unrelated = FakeFigure(id=9, note_id=34, page=10, sha256="u", label_text="CPU GPU memory")
        picked = select_figures(
            [unrelated], [source(1, 34, 10, cited=False)], 3, question="Explain paging hardware"
        )
        assert picked == []

    def test_candidate_pages_cover_every_retrieved_passage(self):
        # The explanation may be on page 10 while its figure sits on page 11
        sources = [source(1, 34, 10, cited=False), source(2, 34, 11, cited=False)]
        assert page_spans(sources) == [(34, 10, 10), (34, 11, 11)]


class TestGenericSubjectWords:
    def test_the_subject_name_does_not_score(self):
        """Every diagram in BCS303 is "about operating systems"."""
        assert generic_terms([source(1, 34, 9, cited=False)]) >= {"operating", "systems", "bcs303"}

    def test_generic_matches_do_not_crowd_out_the_real_topic(self):
        question = "Define operating systems. Explain dual-mode operation with a neat diagram."
        sources = [source(i, 34, 9 + i, cited=False) for i in range(1, 4)]
        dual_mode = FakeFigure(id=1, note_id=34, page=11, sha256="d", label_text="mode bit kernel")
        generic_a = FakeFigure(id=2, note_id=34, page=10, sha256="a", label_text="operating system")
        generic_b = FakeFigure(id=3, note_id=34, page=12, sha256="b", label_text="system programs")

        picked = select_figures([generic_a, dual_mode, generic_b], sources, 3, question=question)
        assert [f.id for f in picked] == [1]

    def test_plurals_fold(self):
        assert "process" in _terms("processes")


class TestAnswerVocabulary:
    """Diagrams seldom name their topic: an SMP figure just says "CPU cache"."""

    QUESTION = "Explain symmetric multiprocessing with a diagram"
    ANSWER = (
        "In symmetric multiprocessing each processor has its own registers and private cache, "
        "and all processors share physical memory."
    )
    SMP = FakeFigure(
        id=1, note_id=34, page=8, sha256="smp", label_text="CPU registers cache memory"
    )
    UNRELATED = FakeFigure(id=2, note_id=34, page=20, sha256="x", label_text="disk USB monitor")

    def test_the_answer_vocabulary_finds_the_right_diagram(self):
        sources = [source(1, 34, 8, cited=False), source(2, 34, 20, cited=False)]
        picked = select_figures(
            [self.UNRELATED, self.SMP], sources, 3, question=self.QUESTION, answer=self.ANSWER
        )
        assert [f.id for f in picked] == [1]

    def test_a_page_mate_from_another_topic_is_still_excluded(self):
        """The reported bug: dual-mode question, CPU/GPU diagram on the same page."""
        question = "Explain dual-mode operation with a neat diagram"
        answer = "A mode bit distinguishes kernel mode from user mode; a trap switches modes."
        wrong = FakeFigure(id=4, note_id=34, page=10, sha256="w", label_text="CPU GPU memory")
        sources = [source(1, 34, 10, cited=False)]

        assert select_figures([wrong], sources, 3, question=question, answer=answer) == []

    def test_nothing_shown_when_neither_question_nor_answer_matches(self):
        sources = [source(1, 34, 20, cited=False)]
        picked = select_figures(
            [self.UNRELATED], sources, 3, question="Explain paging", answer="Paging splits memory."
        )
        assert picked == []


class TestTheQuestionComesFirst:
    """Part (i) of a paper question: an answer about multiprogramming talks about
    "CPU utilization" and "memory", and pulled the multiprocessor diagrams
    (labelled "CPU registers cache memory") in under it."""

    MULTIPROGRAMMING = FakeFigure(
        id=152,
        note_id=34,
        page=9,
        sha256="mp",
        caption="Fig - Memory layout for a multiprogramming system",
        label_text="operating system job",
    )
    SMP = FakeFigure(
        id=150, note_id=34, page=8, sha256="smp", label_text="CPU registers cache memory"
    )
    ANSWER = "Multiprogramming increases CPU utilization by keeping several jobs in memory."

    def test_a_diagram_the_question_names_shuts_out_answer_only_matches(self):
        sources = [source(1, 34, 8, cited=False), source(2, 34, 9, cited=False)]
        picked = select_figures(
            [self.SMP, self.MULTIPROGRAMMING],
            sources,
            3,
            question="Distinguish between Multiprogramming and Multitasking",
            answer=self.ANSWER,
        )
        assert [f.id for f in picked] == [152]

    def test_the_answer_decides_when_the_question_names_no_diagram(self):
        sources = [source(1, 34, 8, cited=False), source(2, 34, 9, cited=False)]
        picked = select_figures(
            [self.SMP, self.MULTIPROGRAMMING],
            sources,
            3,
            question="Distinguish between Multiprocessor System and Clustered System",
            answer="Each processor has its own registers and cache and they share memory.",
        )
        assert [f.id for f in picked] == [150]
