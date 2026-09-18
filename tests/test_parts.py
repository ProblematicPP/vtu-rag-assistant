from vtu_rag.schemas.ask import AskResponse, FigureOut
from vtu_rag.services.rag.parts import combine, marks_per_part, split_parts

PAPER_QUESTION = (
    "Distinguish between the following terms. "
    "(i) Multiprogramming and Multitasking "
    "(ii) Multiprocessor System and Clustered System"
)


def test_roman_parts_become_standalone_questions():
    parts = split_parts(PAPER_QUESTION)
    assert [p.label for p in parts] == ["i", "ii"]
    assert [p.text for p in parts] == [
        "Multiprogramming and Multitasking",
        "Multiprocessor System and Clustered System",
    ]
    assert parts[0].question == "Distinguish between Multiprogramming and Multitasking"
    assert parts[1].question == "Distinguish between Multiprocessor System and Clustered System"


def test_lettered_parts_without_a_following_placeholder_keep_their_stem():
    parts = split_parts("Explain with a neat diagram a) paging b) segmentation")
    assert [p.question for p in parts] == [
        "Explain with a neat diagram: paging",
        "Explain with a neat diagram: segmentation",
    ]


def test_parts_on_separate_lines():
    parts = split_parts("Write short notes on the following:\ni) Deadlock\nii) Starvation")
    assert [p.question for p in parts] == [
        "Write short notes on Deadlock",
        "Write short notes on Starvation",
    ]


def test_single_questions_are_left_whole():
    for question in [
        "What is a system call? List its types",
        "Explain dual-mode operation (i.e. user and kernel mode)",
        "Explain the role of (i) the loader",  # one marker is not a list of parts
        "Explain paging with a diagram",
    ]:
        assert split_parts(question) == []


def test_marks_are_shared_between_parts():
    assert marks_per_part(10, 2) == 5
    assert marks_per_part(None, 2) is None
    assert marks_per_part(3, 3) == 2  # never below a short answer


def _figure(figure_id: int) -> FigureOut:
    return FigureOut(
        id=figure_id,
        url=f"/api/v1/figures/{figure_id}",
        note_id=1,
        subject_code="BCS303",
        module_number=1,
        page=9,
        kind="image",
        caption=None,
        width=400,
        height=300,
    )


def test_each_diagram_stays_with_the_first_part_it_belongs_to():
    parts = split_parts(PAPER_QUESTION)
    first = AskResponse(question=parts[0].question, answer="A", sources=[], figures=[_figure(152)])
    # The same diagram matched the second part too; it must not appear twice
    second = AskResponse(
        question=parts[1].question, answer="B", sources=[], figures=[_figure(152), _figure(160)]
    )
    combined = combine(PAPER_QUESTION, [(parts[0], first), (parts[1], second)])

    assert [f.id for f in combined.parts[0].figures] == [152]
    assert [f.id for f in combined.parts[1].figures] == [160]
    assert [f.id for f in combined.figures] == [152, 160]
    # Clients that ignore `parts` still get every answer, in order, under its label
    assert combined.answer.index("(i) Multiprogramming") < combined.answer.index("A")
    assert combined.answer.index("A") < combined.answer.index("(ii) Multiprocessor")
