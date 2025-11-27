"""Tests for dataset processor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_project.dataset.config import DatasetConfig, SubsetConfig
from ocr_project.dataset.processor import DatasetProcessor


class TestDatasetProcessor:
    """Tests for DatasetProcessor class."""

    @patch("httpx.get")
    def test_init_success(self, mock_get):
        """Test DatasetProcessor initialization with running server."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        subset = SubsetConfig(name="test", splits=["train"], content_columns=["content"])
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        processor = DatasetProcessor(config)

        assert processor.config == config
        assert processor.resolution == "base"

    @patch("httpx.get")
    def test_init_server_not_running(self, mock_get):
        """Test DatasetProcessor initialization with server not running."""
        mock_get.side_effect = Exception("Connection refused")

        subset = SubsetConfig(name="test", splits=["train"], content_columns=["content"])
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        with pytest.raises(RuntimeError, match="vLLM server is not running"):
            DatasetProcessor(config)

    def test_extract_content_from_bytes(self):
        """Test extracting image from bytes."""
        # Create a test image
        img = Image.new("RGB", (10, 10), color="red")
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        subset = SubsetConfig(name="test", splits=["train"], content_columns=["content"])
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        # Mock health check
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            processor = DatasetProcessor(config)

            # Test extraction
            result = processor._extract_content_from_bytes(img_bytes, ".png")

            assert result is not None
            assert isinstance(result, Image.Image)

    def test_extract_content_from_bytes_invalid(self):
        """Test extracting from invalid bytes."""
        invalid_bytes = b"not an image"

        subset = SubsetConfig(name="test", splits=["train"], content_columns=["content"])
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            processor = DatasetProcessor(config)
            result = processor._extract_content_from_bytes(invalid_bytes, ".png")

            assert result is None

    def test_should_process_row_no_filter(self):
        """Test row processing with no filter."""
        subset = SubsetConfig(name="test", splits=["train"], content_columns=["content"])
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            processor = DatasetProcessor(config)

            row = {"content": b"data", "file_type": "image"}
            assert processor._should_process_row(row, subset) is True

    def test_should_process_row_with_filter(self):
        """Test row processing with filter."""
        subset = SubsetConfig(
            name="test",
            splits=["train"],
            content_columns=["content"],
            filter_column="file_type",
            filter_values=["image", "document"],
        )
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            processor = DatasetProcessor(config)

            # Should process
            row1 = {"content": b"data", "file_type": "image"}
            assert processor._should_process_row(row1, subset) is True

            row2 = {"content": b"data", "file_type": "document"}
            assert processor._should_process_row(row2, subset) is True

            # Should not process
            row3 = {"content": b"data", "file_type": "video"}
            assert processor._should_process_row(row3, subset) is False

    def test_process_row_image_column(self):
        """Test processing row with image column."""
        subset = SubsetConfig(
            name="test",
            splits=["train"],
            content_columns=[],
            image_columns=["image"],
        )
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            processor = DatasetProcessor(config)

            # Create test image
            img = Image.new("RGB", (10, 10), color="blue")
            row = {"image": img}

            images = processor._process_row(row, subset)

            assert len(images) == 1
            assert images[0][0] == "image"
            assert images[0][1] == img

    def test_process_row_bytes_column(self):
        """Test processing row with bytes column."""
        subset = SubsetConfig(name="test", splits=["train"], content_columns=["content"])
        config = DatasetConfig(name="test/dataset", subsets=[subset], output_dir=Path("./output"))

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            processor = DatasetProcessor(config)

            # Create test image bytes
            img = Image.new("RGB", (10, 10), color="green")
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            row = {"content": img_bytes, "extension": ".png"}

            images = processor._process_row(row, subset)

            assert len(images) == 1
            assert images[0][0] == "content"
            assert isinstance(images[0][1], Image.Image)
