from pathlib import Path

import pytest

from vtu_rag.ingestion.path_parser import NotePathError, canonical_note_path, parse_note_path


@pytest.mark.parametrize(
    ("relative", "semester", "module"),
    [
        ("cse/2022/sem3/BCS303/module1.pdf", 3, 1),
        ("CSE/2022/semester4/bcs401/Module_4 dynamic programming.pdf", 4, 4),
        ("cse/scheme2022/5/BCS502/mod2.md", 5, 2),
        ("cse/2022/sem3/BCS303/module-5.txt", 3, 5),
    ],
)
def test_parses_valid_paths(tmp_path: Path, relative: str, semester: int, module: int):
    loc = parse_note_path(tmp_path / relative, tmp_path)
    assert loc.branch == "cse"
    assert loc.scheme == "2022"
    assert loc.semester == semester
    assert loc.module_number == module
    assert loc.subject_code == loc.subject_code.upper()
    assert loc.relative_path == relative


@pytest.mark.parametrize(
    "relative",
    [
        "cse/2022/BCS303/module1.pdf",  # missing semester level
        "cse/2022/sem9/BCS303/module1.pdf",  # semester out of range
        "cse/latest/sem3/BCS303/module1.pdf",  # scheme not a year
        "cse/2022/sem3/BCS303/notes.pdf",  # no module number
        "cse/2022/sem3/BCS303/module1.docx",  # unsupported type
        "cse/2022/sem3/BCS303/module12.pdf",  # module numbers are single-digit
    ],
)
def test_rejects_invalid_paths(tmp_path: Path, relative: str):
    with pytest.raises(NotePathError):
        parse_note_path(tmp_path / relative, tmp_path)


def test_rejects_paths_outside_data_dir(tmp_path: Path):
    with pytest.raises(NotePathError):
        parse_note_path(tmp_path.parent / "elsewhere.pdf", tmp_path)


def test_canonical_path_round_trips(tmp_path: Path):
    path = canonical_note_path(tmp_path, "CSE", "2022", 3, "bcs303", "module2.pdf")
    loc = parse_note_path(path, tmp_path)
    assert (loc.subject_code, loc.module_number, loc.semester) == ("BCS303", 2, 3)
