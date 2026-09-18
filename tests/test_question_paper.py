from vtu_rag.ingestion.question_paper import parse_questions

# Shaped like a real VTU paper, including the header noise that must be ignored
PAPER = """
                                                            USN: 1AB21CS045

        Visvesvaraya Technological University, Belagavi
              Third Semester B.E. Degree Examination
                    Operating Systems (BCS303)

Time: 3 hrs.                                         Max. Marks: 100
Note: Answer any FIVE full questions, choosing ONE full question from each module.

                              Module-1
Q.1  a. Define operating system. Explain the two main roles of an OS.   (08 Marks)
     b. With a neat diagram, explain dual-mode operation in an
        operating system.                                              (07 Marks)
     c. List and explain any five services provided by an OS.          (05 Marks)

Q.2  a. Explain the different types of system calls with examples.     (10 Marks)
     b. Differentiate between monolithic and microkernel structures.   (10 Marks)
"""


def test_finds_every_lettered_part():
    questions = parse_questions(PAPER)
    assert [q.number for q in questions] == ["1a", "1b", "1c", "2a", "2b"]


def test_keeps_the_marks():
    marks = {q.number: q.marks for q in parse_questions(PAPER)}
    assert marks == {"1a": 8, "1b": 7, "1c": 5, "2a": 10, "2b": 10}


def test_joins_questions_that_wrap_across_lines():
    wrapped = next(q for q in parse_questions(PAPER) if q.number == "1b")
    assert wrapped.text == "With a neat diagram, explain dual-mode operation in an operating system"


def test_marks_are_stripped_from_the_question_text():
    assert all("Marks" not in q.text for q in parse_questions(PAPER))


def test_university_header_is_not_a_question():
    text = " ".join(q.text.lower() for q in parse_questions(PAPER))
    assert "visvesvaraya" not in text
    assert "usn" not in text
    assert "answer any five" not in text


def test_plain_numbered_questions_without_parts():
    simple = """
    1. Explain the structure of an operating system.   (10 Marks)
    2. What is a process control block? Describe its fields.  (10 Marks)
    """
    questions = parse_questions(simple)
    assert [q.number for q in questions] == ["1", "2"]
    assert questions[1].text.startswith("What is a process control block")


def test_fragments_are_ignored():
    assert parse_questions("a. OK\nb. no\n") == []


def test_empty_input():
    assert parse_questions("") == []


def test_ocr_noise_between_questions_is_tolerated():
    noisy = """
    Q.1 a. Define an operating system and list its goals.  (08 Marks)
    ~~~~ scanned artefact ~~~~
       b. Explain the storage hierarchy with a diagram.    (08 Marks)
    """
    questions = parse_questions(noisy)
    assert [q.number for q in questions] == ["1a", "1b"]
    # the artefact line is swept into the first question, but the question is intact
    assert questions[0].text.startswith("Define an operating system")
