"""Where extracted diagrams live on disk, and how they are served back."""

import logging
import shutil
from pathlib import Path

from vtu_rag.ingestion.figures import ExtractedFigure

logger = logging.getLogger(__name__)

FIGURES_DIRNAME = "derived/figures"

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}


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
        """Maps a stored path back to a file, refusing anything outside the data folder."""
        base = self.data_dir.resolve()
        try:
            candidate = (base / relative_path).resolve()
            candidate.relative_to(base)
        except (ValueError, OSError):
            logger.warning("Refusing figure path outside the data folder: %s", relative_path)
            return None
        return candidate if candidate.is_file() else None
