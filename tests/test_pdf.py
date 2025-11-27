"""Tests for PDF processing utilities."""

import tempfile
from pathlib import Path

import pytest

from ocr_project.utils.pdf import is_pdf, pdf_to_images


class TestPDFUtils:
    """Tests for PDF utility functions."""

    def test_is_pdf_true(self):
        """Test is_pdf returns True for .pdf extension."""
        assert is_pdf(Path("document.pdf"))
        assert is_pdf(Path("DOCUMENT.PDF"))  # Case insensitive
        assert is_pdf(Path("/path/to/file.pdf"))

    def test_is_pdf_false(self):
        """Test is_pdf returns False for non-PDF files."""
        assert not is_pdf(Path("image.png"))
        assert not is_pdf(Path("document.txt"))
        assert not is_pdf(Path("file"))

    def test_pdf_to_images_nonexistent(self):
        """Test pdf_to_images raises FileNotFoundError for nonexistent PDF."""
        with pytest.raises(FileNotFoundError):
            pdf_to_images(Path("/nonexistent/file.pdf"))

    def test_pdf_to_images_invalid(self):
        """Test pdf_to_images raises ValueError for invalid PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"not a valid pdf")
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Failed to convert PDF"):
                pdf_to_images(temp_path)
        finally:
            temp_path.unlink()

    # Note: Testing actual PDF conversion would require a real PDF file
    # or mocking pdf2image.convert_from_path. For now, we test error cases.
    # In a production environment, you'd add a test PDF file to a fixtures folder.
