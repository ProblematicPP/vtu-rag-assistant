"""Pulls diagrams out of note PDFs.

Exam answers in VTU papers usually carry a diagram, so the figures in a module
note matter as much as its prose. Every embedded image is extracted, the
decorative ones (college logos, watermarks, rules) are filtered out, and what
remains is stored next to the note and attached to answers that cite the page
the figure sits on.

Scanned notes are a special case: the whole page is one image, so the "figure"
is the page scan itself — still useful, since that is where the diagram is.
"""

import hashlib
import io
import logging
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

from vtu_rag.ingestion.parsers import ParsedDocument

logger = logging.getLogger(__name__)

# "Fig. 1.2 Layered operating system", "Figure 3 - Process states"
_CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|diagram)\s*\.?\s*(\d+(?:[.\-]\d+)*)?\s*[:.\-–—]?\s*(.{0,120})$",
    re.IGNORECASE,
)


class FigureKind(StrEnum):
    FIGURE = "figure"  # a diagram embedded in a text PDF
    PAGE = "page"  # a scanned page; the diagram is somewhere on it


@dataclass(frozen=True)
class FigureConfig:
    enabled: bool = True
    min_width: int = 150
    min_height: int = 90
    min_bytes: int = 3_000
    # Wider/taller than this is a rule or a decorative band, not a diagram
    max_aspect_ratio: float = 20.0
    max_per_page: int = 6
    # An identical image on this share of pages is a logo/watermark
    repeat_ratio: float = 0.25
    include_page_scans: bool = True
    # Read the labels inside a diagram (Tesseract) so answers can tell two
    # diagrams on the same page apart
    read_labels: bool = True
    label_timeout_seconds: float = 30.0

    @classmethod
    def from_settings(cls, settings) -> "FigureConfig":  # noqa: ANN001 - avoids a circular import
        f = settings.figures
        return cls(
            enabled=f.enabled,
            min_width=f.min_width,
            min_height=f.min_height,
            min_bytes=f.min_bytes,
            max_aspect_ratio=f.max_aspect_ratio,
            max_per_page=f.max_per_page,
            repeat_ratio=f.repeat_ratio,
            include_page_scans=f.include_page_scans,
            read_labels=f.read_labels,
            label_timeout_seconds=f.label_timeout_seconds,
        )


@dataclass
class ExtractedFigure:
    page: int
    index: int  # position within the page
    data: bytes
    extension: str
    width: int
    height: int
    sha256: str
    kind: FigureKind = FigureKind.FIGURE
    caption: str | None = None
    # Words read out of the image itself ("CPU GPU memory", "user mode kernel mode")
    label_text: str | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def _extension(name: str, image: Image.Image) -> str:
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix in {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}:
        return ".jpg" if suffix == "jpeg" else f".{suffix}"
    return f".{(image.format or 'PNG').lower()}"


def _page_captions(text: str) -> list[str]:
    captions = []
    for line in text.splitlines():
        match = _CAPTION_RE.match(line.strip())
        if match and (match.group(2) or match.group(3).strip()):
            caption = " ".join(line.split())
            if 4 <= len(caption) <= 160:
                captions.append(caption)
    return captions


def read_labels(data: bytes, config: FigureConfig) -> str | None:
    """Reads the words printed inside a diagram, via Tesseract.

    Two diagrams on the same page can belong to completely different topics, and
    the page text can't separate them. What the diagram itself says can: "CPU GPU
    memory" versus "user mode kernel mode". Sparse-text mode (--psm 11) suits
    scattered box labels, and greyscale PNG reads better than the source JPEG.
    """
    if shutil.which("tesseract") is None:
        return None

    try:
        with Image.open(io.BytesIO(data)) as image:
            grey = image.convert("L")
            with tempfile.TemporaryDirectory(prefix="vtu-fig-") as tmp:
                path = Path(tmp) / "figure.png"
                grey.save(path)
                result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    ["tesseract", str(path), "stdout", "--psm", "11"],
                    capture_output=True,
                    text=True,
                    timeout=config.label_timeout_seconds,
                    check=False,
                )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.debug("Could not read labels from figure: %s", exc)
        return None

    if result.returncode != 0:
        return None
    # Tesseract on line art emits plenty of junk; keep word-like tokens only
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", result.stdout) if len(w) > 1]
    seen: list[str] = []
    for word in words:
        if word.lower() not in {w.lower() for w in seen}:
            seen.append(word)
    return " ".join(seen)[:300] or None


def _looks_decorative(figure: ExtractedFigure, config: FigureConfig) -> bool:
    if figure.width < config.min_width or figure.height < config.min_height:
        return True
    if figure.size_bytes < config.min_bytes:
        return True
    long_side, short_side = max(figure.width, figure.height), min(figure.width, figure.height)
    return short_side == 0 or long_side / short_side > config.max_aspect_ratio


def extract_figures(
    data: bytes, parsed: ParsedDocument, config: FigureConfig
) -> list[ExtractedFigure]:
    """Returns the diagrams worth keeping, in page order."""
    if not config.enabled:
        return []

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        logger.warning("Could not open PDF for figure extraction: %s", exc)
        return []

    page_text = {p.number: p.text for p in parsed.pages}
    scanned = parsed.ocr_applied
    figures: list[ExtractedFigure] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)[: config.max_per_page]
        except Exception as exc:  # pypdf trips over unusual image filters
            logger.debug("No images read from page %d: %s", page_number, exc)
            continue

        captions = _page_captions(page_text.get(page_number, ""))
        for index, image_file in enumerate(images):
            raw = image_file.data
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    extension = _extension(image_file.name or "", image)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                logger.debug("Skipping unreadable image on page %d: %s", page_number, exc)
                continue

            figure = ExtractedFigure(
                page=page_number,
                index=index,
                data=raw,
                extension=extension,
                width=width,
                height=height,
                sha256=hashlib.sha256(raw).hexdigest(),
                kind=FigureKind.PAGE if scanned else FigureKind.FIGURE,
                caption=captions[index] if index < len(captions) else None,
            )
            if figure.kind is FigureKind.PAGE and not config.include_page_scans:
                continue
            if _looks_decorative(figure, config):
                continue
            figures.append(figure)

    kept = _drop_repeated(figures, page_count=parsed.page_count or len(reader.pages), config=config)
    if config.read_labels:
        for figure in kept:
            figure.label_text = read_labels(figure.data, config)
    return kept


def _drop_repeated(
    figures: list[ExtractedFigure], page_count: int, config: FigureConfig
) -> list[ExtractedFigure]:
    """Removes images that recur across pages — logos, watermarks, header bands."""
    if page_count < 3:
        return figures
    pages_per_hash = Counter(
        {h: len({f.page for f in figures if f.sha256 == h}) for h in {f.sha256 for f in figures}}
    )
    threshold = max(2, int(page_count * config.repeat_ratio))
    repeated = {h for h, pages in pages_per_hash.items() if pages >= threshold}
    if repeated:
        logger.info("Dropping %d repeated image(s) (logos/watermarks)", len(repeated))
    return [f for f in figures if f.sha256 not in repeated]
