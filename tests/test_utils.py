"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pytest

from ocr_project.utils.file_io import read_markdown, save_markdown
from ocr_project.utils.image import get_image_files, load_image


class TestImageUtils:
    """Tests for image utility functions."""

    def test_load_image_nonexistent(self):
        """Test loading a nonexistent image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image(Path("/nonexistent/image.png"))

    def test_load_image_invalid(self):
        """Test loading an invalid image raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not an image")
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                load_image(temp_path)
        finally:
            temp_path.unlink()

    def test_get_image_files_empty_dir(self):
        """Test getting image files from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_image_files(Path(tmpdir))
            assert result == []

    def test_get_image_files_with_images(self):
        """Test getting image files from directory with images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create test files
            (tmppath / "test.txt").touch()
            (tmppath / "image1.png").touch()
            (tmppath / "image2.jpg").touch()

            result = get_image_files(tmppath)
            assert len(result) == 2
            assert all(f.suffix.lower() in {".png", ".jpg"} for f in result)


class TestFileIO:
    """Tests for file I/O utilities."""

    def test_save_and_read_markdown(self):
        """Test saving and reading markdown content."""
        content = "# Test\n\nThis is a test."

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"
            save_markdown(content, output_path)

            assert output_path.exists()
            read_content = read_markdown(output_path)
            assert read_content == content

    def test_save_markdown_creates_parent_dirs(self):
        """Test that save_markdown creates parent directories."""
        content = "# Test"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test.md"
            save_markdown(content, output_path)

            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == content
