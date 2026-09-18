"""The answer prompt must know whether a real diagram will be shown."""

import pytest

from tests.fakes import ScriptedLLM, make_hit
from vtu_rag.services.rag import prompts
from vtu_rag.services.rag.figures import any_figure_matches
from vtu_rag.services.rag.service import AnswerGenerator
from vtu_rag.services.search import SearchFilters


class Recorder(ScriptedLLM):
    """Captures the system prompt the generator actually sent."""

    def __init__(self):
        self.system = ""
        super().__init__(self._handle)

    def _handle(self, messages):
        self.system = messages[0].content
        return "Paging splits memory into frames."


@pytest.mark.parametrize(
    ("shown", "expected", "forbidden"),
    [
        (True, prompts.DIAGRAM_SHOWN, prompts.DIAGRAM_MISSING),
        (False, prompts.DIAGRAM_MISSING, prompts.DIAGRAM_SHOWN),
    ],
)
async def test_the_prompt_states_whether_a_diagram_is_shown(shown, expected, forbidden):
    llm = Recorder()
    await AnswerGenerator(llm).generate(
        "Explain paging with a diagram",
        [make_hit(1, "Paging splits memory into frames.")],
        SearchFilters(),
        diagram_shown=shown,
    )
    assert expected in llm.system
    assert forbidden not in llm.system


class FakeFigure:
    def __init__(self, label: str):
        self.caption = None
        self.label_text = label


class FakeHit:
    subject_code = "BCS303"
    subject_name = "Operating Systems"


def test_probe_sees_a_matching_diagram():
    figures = [FakeFigure("page table frame offset")]
    assert any_figure_matches(figures, "Explain paging hardware", [FakeHit()]) is True


def test_probe_ignores_diagrams_about_something_else():
    figures = [FakeFigure("CPU GPU memory")]
    assert any_figure_matches(figures, "Explain paging hardware", [FakeHit()]) is False


def test_probe_with_no_figures():
    assert any_figure_matches([], "Explain paging", [FakeHit()]) is False
