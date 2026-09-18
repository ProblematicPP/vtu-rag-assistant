from pathlib import Path

import pytest

from vtu_rag.ingestion import parsers
from vtu_rag.ingestion.ocr import OcrConfig, OcrError, run_ocr
from vtu_rag.ingestion.parsers import ParsedDocument, ParsedPage, ParseError, needs_ocr, parse_pdf

CONFIG = OcrConfig(min_words_per_page=20, min_low_text_ratio=0.3)


def doc(*word_counts: int) -> ParsedDocument:
    return ParsedDocument(
        pages=[
            ParsedPage(number=i, text=" ".join(["word"] * n)) for i, n in enumerate(word_counts, 1)
        ]
    )


class TestNeedsOcr:
    def test_scanned_pdf_has_no_text_at_all(self):
        assert needs_ocr(doc(0, 0, 0), CONFIG) is True

    def test_digital_pdf_is_left_alone(self):
        assert needs_ocr(doc(400, 380, 420), CONFIG) is False

    def test_a_single_sparse_page_does_not_trigger_ocr(self):
        # A title page in an otherwise fine 10-page PDF: OCR would be wasted time
        assert needs_ocr(doc(5, *[400] * 9), CONFIG) is False

    def test_mostly_image_pages_do_trigger_ocr(self):
        assert needs_ocr(doc(0, 0, 400, 400), CONFIG) is True

    def test_empty_document(self):
        assert needs_ocr(ParsedDocument(), CONFIG) is False


class TestParsePdfWithOcr:
    def test_text_pdf_never_calls_ocr(self, monkeypatch):
        monkeypatch.setattr(parsers, "_extract", lambda data: doc(300, 300))
        monkeypatch.setattr(
            parsers, "run_ocr", lambda *a, **k: pytest.fail("OCR must not run on a text PDF")
        )
        result = parse_pdf(b"%PDF-fake", CONFIG)
        assert result.ocr_applied is False

    def test_scanned_pdf_is_rescued_by_the_first_pass(self, monkeypatch):
        calls: list[bool] = []
        results = iter([doc(0, 0), doc(250, 240)])
        monkeypatch.setattr(parsers, "_extract", lambda data: next(results))

        def fake_ocr(data, config, *, force=False):
            calls.append(force)
            return b"%PDF-ocred"

        monkeypatch.setattr(parsers, "run_ocr", fake_ocr)
        result = parse_pdf(b"%PDF-scan", CONFIG)

        assert calls == [False]  # no --force-ocr retry needed
        assert result.ocr_applied is True
        assert result.word_count == 490

    def test_junk_text_layer_falls_back_to_force_ocr(self, monkeypatch):
        calls: list[bool] = []
        # original → skip-text pass (no better) → force pass (real text)
        results = iter([doc(2, 3), doc(2, 3), doc(300, 310)])
        monkeypatch.setattr(parsers, "_extract", lambda data: next(results))

        def fake_ocr(data, config, *, force=False):
            calls.append(force)
            return b"%PDF-ocred"

        monkeypatch.setattr(parsers, "run_ocr", fake_ocr)
        result = parse_pdf(b"%PDF-scan", CONFIG)

        assert calls == [False, True]
        assert result.ocr_applied is True

    def test_ocr_failure_surfaces_as_a_parse_error(self, monkeypatch):
        monkeypatch.setattr(parsers, "_extract", lambda data: doc(0, 0))

        def broken_ocr(data, config, *, force=False):
            raise OcrError("ocrmypdf is not installed")

        monkeypatch.setattr(parsers, "run_ocr", broken_ocr)
        with pytest.raises(ParseError, match="ocrmypdf is not installed"):
            parse_pdf(b"%PDF-scan", CONFIG)

    def test_disabled_ocr_keeps_the_old_behaviour(self, monkeypatch):
        monkeypatch.setattr(parsers, "_extract", lambda data: doc(0))
        with pytest.raises(ParseError, match="scanned PDF needs OCR"):
            parse_pdf(b"%PDF-scan", OcrConfig(enabled=False))


class TestOcrCache:
    def test_cached_result_is_reused_without_running_ocrmypdf(self, tmp_path: Path, monkeypatch):
        config = OcrConfig(cache_dir=tmp_path)
        monkeypatch.setattr(
            "vtu_rag.ingestion.ocr.ocr_available", lambda: pytest.fail("should not shell out")
        )
        # Prime the cache the way a previous run would have
        from vtu_rag.ingestion.ocr import _cache_key

        (tmp_path / f"{_cache_key(b'%PDF-scan', config, False)}.pdf").write_bytes(b"%PDF-cached")

        assert run_ocr(b"%PDF-scan", config) == b"%PDF-cached"

    def test_missing_binary_is_reported_clearly(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("vtu_rag.ingestion.ocr.ocr_available", lambda: False)
        with pytest.raises(OcrError, match="not installed"):
            run_ocr(b"%PDF-scan", OcrConfig(cache_dir=tmp_path))
