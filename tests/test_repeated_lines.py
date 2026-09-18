from vtu_rag.ingestion.parsers import ParsedDocument, ParsedPage, strip_repeated_lines

# The real footer from a BCS303 note, which was being read as a section heading
FOOTER = "Karthikeyan S M, Asst. Professor, Dept. of. CSE, SVIT, Bengaluru"

BODIES = [
    "An operating system acts as an intermediary between user and hardware.",
    "The kernel is the one program running at all times on the computer.",
    "A process is a program in execution, with a stack and a data section.",
    "Semaphores are accessed through the wait and signal operations only.",
    "Deadlock requires mutual exclusion, hold and wait, and circular wait.",
    "Paging avoids external fragmentation by splitting memory into frames.",
    "A page table maps virtual page numbers to physical frame numbers.",
    "Thrashing happens when a process spends more time paging than running.",
]


def note(count: int = 8, footer: bool = True, extra_on: tuple[int, ...] = ()) -> ParsedDocument:
    """A note whose pages each carry their own heading and body, plus a numbered footer.

    The footer sits mid-page on purpose: pypdf returns text in content-stream
    order, so a visual footer routinely lands in the middle of extracted text.
    """
    pages = []
    for i in range(1, count + 1):
        lines = [f"{i}.1 Section heading {BODIES[i - 1][:12]}"]
        if i in extra_on:
            lines.append("Advantages:")
        if footer:
            lines.append(f"{i} {FOOTER}")
        lines.append(BODIES[i - 1])
        pages.append(ParsedPage(number=i, text="\n".join(lines)))
    return ParsedDocument(pages=pages)


def test_running_footer_is_removed_from_every_page():
    cleaned = strip_repeated_lines(note())
    assert all("Karthikeyan" not in p.text for p in cleaned.pages)
    # the actual content survives
    assert all(BODIES[p.number - 1] in p.text for p in cleaned.pages)


def test_page_numbers_do_not_defeat_the_match():
    # "Page 1 of 20", "Page 2 of 20", ... differ only by digits
    doc = ParsedDocument(
        pages=[ParsedPage(i, f"{BODIES[i % 8]}\nPage {i} of 20") for i in range(1, 11)]
    )
    cleaned = strip_repeated_lines(doc)
    assert all("Page" not in p.text for p in cleaned.pages)


def test_repeated_mid_page_text_is_kept():
    """ "Advantages:" recurs on half the pages — that is content, not a footer."""
    cleaned = strip_repeated_lines(note(extra_on=(1, 2, 3, 4)))
    assert all("Advantages:" in cleaned.pages[i - 1].text for i in (1, 2, 3, 4))
    assert all("Karthikeyan" not in p.text for p in cleaned.pages)


def test_long_repeated_lines_are_kept():
    long_line = (
        "This definition is repeated verbatim in the notes across several pages, "
        "and is far too long to be a running footer."
    )
    doc = ParsedDocument(pages=[ParsedPage(i, f"Heading {i}\n{long_line}") for i in range(1, 11)])
    cleaned = strip_repeated_lines(doc)
    assert all(long_line in p.text for p in cleaned.pages)


def test_short_documents_are_left_alone():
    doc = ParsedDocument(pages=[ParsedPage(1, f"Body\n{FOOTER}"), ParsedPage(2, f"More\n{FOOTER}")])
    assert strip_repeated_lines(doc) is doc


def test_section_headings_survive():
    cleaned = strip_repeated_lines(note())
    assert all("Section heading" in p.text for p in cleaned.pages)


def test_ocr_flag_survives_cleaning():
    doc = note()
    doc.ocr_applied = True
    assert strip_repeated_lines(doc).ocr_applied is True
