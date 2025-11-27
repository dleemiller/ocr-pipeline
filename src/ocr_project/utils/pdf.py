"""PDF processing utilities for converting PDFs to images."""

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[tuple[int, Image.Image]]:
    """Convert PDF to list of (page_number, image) tuples.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 200, higher = better quality but slower)

    Returns:
        List of (page_num, PIL.Image) tuples, 1-indexed
        Example: [(1, img1), (2, img2), (3, img3)] for a 3-page PDF

    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF is invalid or corrupted
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Convert PDF to list of PIL Images
        # pdf2image uses poppler under the hood
        images = convert_from_path(pdf_path, dpi=dpi)

        # Return as (page_number, image) tuples, 1-indexed
        return [(i + 1, img) for i, img in enumerate(images)]

    except Exception as e:
        raise ValueError(f"Failed to convert PDF {pdf_path}: {e}") from e


def is_pdf(file_path: Path) -> bool:
    """Check if file is a PDF by extension.

    Args:
        file_path: Path to file

    Returns:
        True if file has .pdf extension (case-insensitive)
    """
    return file_path.suffix.lower() == ".pdf"
