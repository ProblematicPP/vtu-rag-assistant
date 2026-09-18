"""OCR for scanned notes, delegated to the `ocrmypdf` CLI (Tesseract under the hood).

Most VTU notes in circulation are photocopies scanned to PDF: the pages are
images, so `pypdf` extracts nothing. When a PDF looks like that, we run it
through ocrmypdf, which adds a text layer, and parse the result instead.

OCR is slow (seconds per page), so results are cached on disk under the data
directory, keyed by the content of the original file. Re-indexing an unchanged
note therefore never pays for OCR twice.
"""

import hashlib
import io
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

OCRMYPDF = "ocrmypdf"
CACHE_DIRNAME = ".ocr-cache"


class OcrError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrConfig:
    """Plain config so parsers and tests don't need the full Settings object."""

    enabled: bool = True
    language: str = "eng"
    # A page with fewer words than this is treated as un-extracted (i.e. an image)
    min_words_per_page: int = 20
    # OCR the document when at least this fraction of pages look un-extracted
    min_low_text_ratio: float = 0.3
    timeout_seconds: float = 1800.0
    # Retry with --force-ocr when the first pass didn't help (junk text layers)
    force_retry: bool = True
    deskew: bool = False
    cache_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings) -> "OcrConfig":  # noqa: ANN001 - avoids a circular import
        return cls(
            enabled=settings.ocr.enabled,
            language=settings.ocr.language,
            min_words_per_page=settings.ocr.min_words_per_page,
            min_low_text_ratio=settings.ocr.min_low_text_ratio,
            timeout_seconds=settings.ocr.timeout_seconds,
            force_retry=settings.ocr.force_retry,
            deskew=settings.ocr.deskew,
            cache_dir=settings.data_dir / CACHE_DIRNAME,
        )


def ocr_available() -> bool:
    return shutil.which(OCRMYPDF) is not None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def image_to_text(
    data: bytes, *, language: str = "eng", psm: str = "3", timeout: float = 60.0
) -> str:
    """Runs Tesseract over an image and returns its raw text.

    psm 3 reads a page of prose (a photographed question paper); psm 11 reads
    scattered labels (the text inside a diagram).
    """
    if not tesseract_available():
        return ""

    try:
        with Image.open(io.BytesIO(data)) as image:
            grey = image.convert("L")
            with tempfile.TemporaryDirectory(prefix="vtu-ocr-img-") as tmp:
                path = Path(tmp) / "page.png"
                grey.save(path)
                result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    ["tesseract", str(path), "stdout", "--psm", psm, "-l", language],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.warning("Tesseract failed on image: %s", exc)
        return ""
    return result.stdout if result.returncode == 0 else ""


def _cache_key(data: bytes, config: OcrConfig, force: bool) -> str:
    digest = hashlib.sha256(data)
    digest.update(f"|{config.language}|{force}|{config.deskew}".encode())
    return digest.hexdigest()


def _cache_path(config: OcrConfig, key: str) -> Path | None:
    return config.cache_dir / f"{key}.pdf" if config.cache_dir else None


def run_ocr(data: bytes, config: OcrConfig, *, force: bool = False) -> bytes:
    """Returns the OCR'd PDF bytes. Raises OcrError if ocrmypdf can't produce one."""
    key = _cache_key(data, config, force)
    cached = _cache_path(config, key)
    if cached is not None and cached.exists():
        logger.info("Using cached OCR result %s", cached.name)
        return cached.read_bytes()

    if not ocr_available():
        raise OcrError(
            "ocrmypdf is not installed. It ships in the Docker image; "
            "for local runs install ocrmypdf and tesseract-ocr."
        )

    with tempfile.TemporaryDirectory(prefix="vtu-ocr-") as tmp:
        src = Path(tmp) / "in.pdf"
        dst = Path(tmp) / "out.pdf"
        src.write_bytes(data)

        cmd = [
            OCRMYPDF,
            "--quiet",
            "--language",
            config.language,
            "--output-type",
            "pdf",
            # Skip image optimisation: we only want the text layer, not a smaller file
            "--optimize",
            "0",
            "--invalidate-digital-signatures",
            "--force-ocr" if force else "--skip-text",
        ]
        if config.deskew:
            cmd.append("--deskew")
        cmd += [str(src), str(dst)]

        logger.info(
            "Running OCR (%s, lang=%s) on %.1f MB PDF — this can take a while",
            "force" if force else "skip-text",
            config.language,
            len(data) / 1e6,
        )
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                cmd, capture_output=True, timeout=config.timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise OcrError(f"OCR timed out after {config.timeout_seconds:.0f}s") from exc
        except OSError as exc:
            raise OcrError(f"Could not run ocrmypdf: {exc}") from exc

        if result.returncode != 0 or not dst.exists():
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()
            raise OcrError(f"ocrmypdf failed (exit {result.returncode}): {detail[:500]}")

        ocred = dst.read_bytes()

    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(ocred)
        except OSError as exc:  # a full or read-only data volume must not fail ingestion
            logger.warning("Could not cache OCR result: %s", exc)
    return ocred
