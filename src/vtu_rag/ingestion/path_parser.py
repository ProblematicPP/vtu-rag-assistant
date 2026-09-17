"""Derives syllabus metadata from a note's location in the notes tree.

<data_dir>/<branch>/<scheme>/<semester>/<SUBJECT_CODE>/module<N>[anything].<ext>
"""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt"})

_SEMESTER_RE = re.compile(r"^(?:sem(?:ester)?[\s_-]*)?([1-8])$", re.IGNORECASE)
_SCHEME_RE = re.compile(r"^(?:scheme[\s_-]*)?(20\d{2})$", re.IGNORECASE)
_MODULE_RE = re.compile(r"^mod(?:ule)?[\s_-]*([1-9])(?!\d)", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"^[A-Za-z0-9]{4,12}$")


class NotePathError(ValueError):
    pass


@dataclass(frozen=True)
class NoteLocation:
    branch: str
    scheme: str
    semester: int
    subject_code: str
    module_number: int
    relative_path: str  # POSIX-style, relative to data_dir — used as the note's source_uri

    @property
    def title(self) -> str:
        return PurePosixPath(self.relative_path).stem


def parse_note_path(path: Path, data_dir: Path) -> NoteLocation:
    try:
        relative = path.resolve().relative_to(data_dir.resolve())
    except ValueError as exc:
        raise NotePathError(f"{path} is not inside {data_dir}") from exc

    parts = relative.parts
    if len(parts) != 5:
        raise NotePathError(
            f"{relative.as_posix()}: expected <branch>/<scheme>/<semester>/<subject>/<file>"
        )
    branch, scheme, semester, subject, filename = parts

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise NotePathError(f"{relative.as_posix()}: unsupported file type {path.suffix!r}")

    scheme_match = _SCHEME_RE.match(scheme)
    if not scheme_match:
        raise NotePathError(f"{relative.as_posix()}: scheme folder {scheme!r} is not a year")

    semester_match = _SEMESTER_RE.match(semester)
    if not semester_match:
        raise NotePathError(f"{relative.as_posix()}: bad semester folder {semester!r}")

    if not _SUBJECT_RE.match(subject):
        raise NotePathError(f"{relative.as_posix()}: bad subject code {subject!r}")

    module_match = _MODULE_RE.match(Path(filename).stem)
    if not module_match:
        raise NotePathError(f"{relative.as_posix()}: file name must start with 'module<N>'")

    return NoteLocation(
        branch=branch.lower(),
        scheme=scheme_match.group(1),
        semester=int(semester_match.group(1)),
        subject_code=subject.upper(),
        module_number=int(module_match.group(1)),
        relative_path=relative.as_posix(),
    )


def canonical_note_path(
    data_dir: Path, branch: str, scheme: str, semester: int, subject_code: str, filename: str
) -> Path:
    return data_dir / branch.lower() / scheme / f"sem{semester}" / subject_code.upper() / filename
