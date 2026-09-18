"""Safe resolution of paths stored relative to the data directory."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}


def resolve_in_data_dir(data_dir: Path, relative_path: str) -> Path | None:
    """Maps a stored relative path to a file, refusing anything outside the data folder."""
    base = data_dir.resolve()
    try:
        candidate = (base / relative_path).resolve()
        candidate.relative_to(base)
    except (ValueError, OSError):
        logger.warning("Refusing path outside the data folder: %s", relative_path)
        return None
    return candidate if candidate.is_file() else None


def media_type_for(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
