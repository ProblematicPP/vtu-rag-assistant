import pytest

from tests.fakes import make_hit
from vtu_rag.services.cache import ResponseCache
from vtu_rag.services.llm import LLMError, parse_json_object
from vtu_rag.services.rag import context
from vtu_rag.services.search import SearchFilters


@pytest.mark.parametrize(
    "text",
    [
        '{"score": 80, "reason": "ok"}',
        'Sure! Here you go:\n```json\n{"score": 80, "reason": "ok"}\n```',
        'Result -> {"score": 80, "reason": "ok"} <- done',
    ],
)
def test_parse_json_object_tolerates_wrapping(text: str):
    assert parse_json_object(text)["score"] == 80


def test_parse_json_object_rejects_non_objects():
    with pytest.raises(LLMError):
        parse_json_object("[1, 2, 3]")
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_cited_indices_handles_groups():
    answer = "Paging avoids external fragmentation [1][3]. TLB speeds lookup [2, 4]."
    assert context.cited_indices(answer) == {1, 2, 3, 4}


def test_format_context_numbers_sources_and_labels_them():
    hits = [make_hit(0, "Paging divides memory."), make_hit(1, "Segmentation is logical.")]
    block, sources = context.format_context(hits)
    assert block.startswith("[1] BCS303 Operating Systems — Module 4: Memory Management")
    assert "\n\n[2] " in block
    assert [s.index for s in sources] == [1, 2]


def test_format_context_respects_word_budget(monkeypatch):
    monkeypatch.setattr(context, "MAX_CONTEXT_WORDS", 5)
    hits = [make_hit(0, "one two three four five six"), make_hit(1, "never included")]
    block, sources = context.format_context(hits)
    assert len(sources) == 1
    assert "six" not in block


def test_mark_cited_flags_only_cited_sources():
    _, sources = context.format_context([make_hit(0, "a"), make_hit(1, "b")])
    marked = context.mark_cited("Answer [2].", sources)
    assert [s.cited for s in marked] == [False, True]


def test_scope_description():
    assert context.scope_description(SearchFilters()) == ""
    desc = context.scope_description(SearchFilters(subject_code="bcs303", module_numbers=[2]))
    assert desc == " for BCS303 module 2"


def test_cache_key_is_stable_and_sensitive():
    k1 = ResponseCache.make_key("ask", q="paging", top_k=5)
    k2 = ResponseCache.make_key("ask", top_k=5, q="paging")
    k3 = ResponseCache.make_key("ask", q="paging", top_k=6)
    assert k1 == k2 != k3


class TestStripCitationMarkers:
    def test_markers_and_their_spacing_are_removed(self):
        assert (
            context.strip_citation_markers("Paging splits memory [1]. It avoids waste [2][3].")
            == "Paging splits memory. It avoids waste."
        )

    def test_grouped_markers(self):
        assert context.strip_citation_markers("Three states exist [1, 2].") == "Three states exist."

    def test_plain_answers_are_untouched(self):
        answer = "A semaphore is an integer variable.\n\n- wait()\n- signal()"
        assert context.strip_citation_markers(answer) == answer
