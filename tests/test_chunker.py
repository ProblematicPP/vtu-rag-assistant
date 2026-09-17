from pathlib import Path

import pytest

from vtu_rag.ingestion.chunker import SectionChunker, _heading_level
from vtu_rag.ingestion.parsers import ParsedDocument, ParsedPage, parse_text

SAMPLE = Path(__file__).parents[1] / "data/cse/2022/sem3/BCS303/module1-sample.md"


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("# Module 1", 1),
        ("### Interrupts", 3),
        ("MODULE 2", 1),
        ("Module-3: Process Synchronization", 1),
        ("2.3 Paging", 2),
        ("4 Memory Management", 1),
        ("DEADLOCK PREVENTION", 1),
        ("The kernel runs in privileged mode.", None),
        ("3 processes are waiting for the CPU and none can proceed.", None),
        ("OS", None),  # too short to be a caps heading
    ],
)
def test_heading_detection(line: str, level: int | None):
    assert _heading_level(line) == level


def test_sections_become_separate_chunks_with_heading_paths():
    text = f"# Memory\n\n## Paging\n\n{words(120, 'p')}\n\n## Segmentation\n\n{words(120, 's')}"
    doc = parse_text(text.encode())
    chunks = SectionChunker(target_words=300, overlap_words=20, min_words=50).chunk(doc)
    assert [c.section_heading for c in chunks] == ["Memory > Paging", "Memory > Segmentation"]
    assert chunks[0].text.startswith("p0") and chunks[1].text.startswith("s0")


def test_long_section_is_split_with_overlap():
    doc = parse_text(f"# Big\n\n{words(1000)}".encode())
    chunks = SectionChunker(target_words=300, overlap_words=50, min_words=50).chunk(doc)
    assert len(chunks) >= 4
    assert all(c.word_count <= 350 for c in chunks)
    # consecutive chunks share the overlap words
    first_tail = chunks[0].text.split()[-50:]
    assert chunks[1].text.split()[:50] == first_tail
    # every source word appears somewhere
    covered = set(" ".join(c.text for c in chunks).split())
    assert covered == set(words(1000).split())


def test_small_sections_are_merged_forward():
    doc = parse_text(f"# Tiny\n\nshort intro\n\n# Real\n\n{words(100)}".encode())
    chunks = SectionChunker(target_words=300, overlap_words=20, min_words=50).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].section_heading == "Real"
    assert chunks[0].text.startswith("short intro")


def test_page_numbers_are_tracked():
    doc = ParsedDocument(
        pages=[
            ParsedPage(1, f"MEMORY MANAGEMENT\n{words(200, 'a')}"),
            ParsedPage(2, words(200, "b")),
            ParsedPage(3, words(200, "c")),
        ]
    )
    chunks = SectionChunker(target_words=250, overlap_words=0, min_words=50).chunk(doc)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_zero_overlap_does_not_duplicate_text():
    doc = parse_text(f"# Big\n\n{words(900)}".encode())
    chunks = SectionChunker(target_words=300, overlap_words=0, min_words=50).chunk(doc)
    assert sum(c.word_count for c in chunks) == 900


def test_sample_note_chunks_sensibly():
    doc = parse_text(SAMPLE.read_bytes())
    chunks = SectionChunker().chunk(doc)
    headings = [c.section_heading for c in chunks]
    assert len(chunks) >= 4
    assert any(h and "Dual-Mode Operation" in h for h in headings)
    assert all(c.word_count >= 40 for c in chunks)
