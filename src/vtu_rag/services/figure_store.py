"""Where extracted diagrams live on disk, and how they are served back."""

import logging
import shutil
from pathlib import Path

from vtu_rag.ingestion.figures import ExtractedFigure
from vtu_rag.services.data_files import MEDIA_TYPES, resolve_in_data_dir

logger = logging.getLogger(__name__)

FIGURES_DIRNAME = "derived/figures"


class FigureStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.root = data_dir / FIGURES_DIRNAME

    def note_dir(self, note_id: int) -> Path:
        return self.root / f"note{note_id}"

    def save(self, note_id: int, figure: ExtractedFigure) -> tuple[str, str]:
        """Writes the image and returns (path relative to DATA_DIR, media type)."""
        directory = self.note_dir(note_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"p{figure.page:04d}-{figure.index}{figure.extension}"
        path.write_bytes(figure.data)
        relative = path.relative_to(self.data_dir).as_posix()
        return relative, MEDIA_TYPES.get(figure.extension, "application/octet-stream")

    def clear(self, note_id: int) -> None:
        shutil.rmtree(self.note_dir(note_id), ignore_errors=True)

    def resolve(self, relative_path: str) -> Path | None:
        return resolve_in_data_dir(self.data_dir, relative_path)
