"""Where notes come from.

A `NoteSource` discovers notes and yields them with their syllabus location.
`LocalFolderSource` walks the data directory; `ScraperSource` is the
extension point for pulling notes from public VTU resource sites.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from vtu_rag.ingestion.path_parser import (
    SUPPORTED_EXTENSIONS,
    NoteLocation,
    NotePathError,
    parse_note_path,
)
from vtu_rag.models import SourceType

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredNote:
    location: NoteLocation
    source_uri: str
    source_type: SourceType
    extension: str
    content: bytes
    title: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


class NoteSource(ABC):
    @abstractmethod
    def discover(self) -> AsyncIterator[DiscoveredNote]: ...


# Folders the pipeline writes into; never scanned for notes
DERIVED_DIRS = frozenset({"derived"})


class LocalFolderSource(NoteSource):
    """Yields every supported file under `data_dir` that matches the folder convention."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.skipped: list[str] = []

    async def discover(self) -> AsyncIterator[DiscoveredNote]:
        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.name.lower() == "readme.md":
                continue
            # Skip what the pipeline itself writes: OCR cache, extracted figures
            relative = path.relative_to(self.data_dir)
            if relative.parts[0] in DERIVED_DIRS or any(p.startswith(".") for p in relative.parts):
                continue
            try:
                location = parse_note_path(path, self.data_dir)
            except NotePathError as exc:
                logger.warning("Skipping %s", exc)
                self.skipped.append(str(exc))
                continue
            yield DiscoveredNote(
                location=location,
                source_uri=location.relative_path,
                source_type=SourceType.UPLOAD,
                extension=path.suffix.lower(),
                content=path.read_bytes(),
                title=location.title,
            )


class ScraperSource(NoteSource):
    """Extension point for scraping public VTU resource sites.

    To implement a scraper:
      1. Subclass this and implement `list_documents()` to return
         (url, NoteLocation-like metadata) pairs for a site.
      2. Download each document in `discover()` and yield a `DiscoveredNote`
         with `source_type=SourceType.SCRAPED` and `source_uri=<url>`.
      3. Respect robots.txt, rate-limit requests, and only ingest material
         you are allowed to redistribute/use.

    The ingestion pipeline treats scraped notes exactly like local ones.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def discover(self) -> AsyncIterator[DiscoveredNote]:
        raise NotImplementedError("No scraper is implemented yet")
        yield  # pragma: no cover - makes this an async generator
