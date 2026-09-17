from pathlib import Path

import pytest

from vtu_rag.ingestion.path_parser import parse_note_path
from vtu_rag.routers.notes import _safe_stem


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Module 4 Paging Notes.pdf", "paging-notes"),
        ("mod2_process_management.pdf", "process-management"),
        ("OS Notes (final) v2.md", "os-notes-final-v2"),
        ("module1.pdf", ""),  # nothing left once the module prefix is stripped
        ("../../etc/passwd.txt", "passwd"),
    ],
)
def test_safe_stem(filename: str, expected: str):
    assert _safe_stem(filename) == expected


def test_uploaded_names_stay_inside_the_data_dir_and_parse_back(tmp_path: Path):
    stem = _safe_stem("../../secret/Module 3 notes.pdf")
    path = tmp_path / "cse/2022/sem3/BCS303" / f"module3-{stem}.pdf"
    path.parent.mkdir(parents=True)
    path.touch()

    location = parse_note_path(path, tmp_path)
    assert location.module_number == 3
    assert location.subject_code == "BCS303"
    assert ".." not in location.relative_path
