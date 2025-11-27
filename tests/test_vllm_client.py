"""Tests for vLLM API client."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_project.models.vllm_client import ServerNotAvailableError, VLLMClient


class TestVLLMClient:
    """Tests for VLLMClient class."""

    def test_init(self):
        """Test VLLMClient initialization."""
        client = VLLMClient()
        assert client.base_url == "http://127.0.0.1:8000/v1"
        assert client.timeout == 300
        assert client.max_retries == 3

    def test_init_custom_params(self):
        """Test VLLMClient with custom parameters."""
        client = VLLMClient(base_url="http://localhost:9000/v1", timeout=600, max_retries=5)
        assert client.base_url == "http://localhost:9000/v1"
        assert client.timeout == 600
        assert client.max_retries == 5

    @patch("httpx.get")
    def test_health_check_success(self, mock_get):
        """Test health check when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = VLLMClient()
        assert client.health_check() is True

    @patch("httpx.get")
    def test_health_check_failure(self, mock_get):
        """Test health check when server is not running."""
        mock_get.side_effect = Exception("Connection refused")

        client = VLLMClient()
        assert client.health_check() is False

    def test_image_to_base64(self):
        """Test image to base64 conversion."""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="red")

        client = VLLMClient()
        result = client.image_to_base64(img)

        # Check that result is a data URI
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 50  # Should contain base64 data

    @patch("httpx.get")
    def test_process_image_server_not_available(self, mock_get):
        """Test process_image raises error when server is not available."""
        mock_get.side_effect = Exception("Connection refused")

        client = VLLMClient()
        img = Image.new("RGB", (10, 10), color="red")

        with pytest.raises(ServerNotAvailableError):
            client.process_image(img)

    @patch("httpx.get")
    @patch("ocr_project.models.vllm_client.OpenAI")
    def test_process_image_success(self, mock_openai_class, mock_get):
        """Test successful image processing."""
        # Mock health check
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_get.return_value = mock_health_response

        # Mock OpenAI client response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "# Test Output\n\nOCR result"
        mock_client.chat.completions.create.return_value = mock_response

        client = VLLMClient()
        img = Image.new("RGB", (10, 10), color="red")
        result = client.process_image(img)

        assert result == "# Test Output\n\nOCR result"

    @patch("httpx.get")
    @patch("ocr_project.models.vllm_client.OpenAI")
    def test_process_batch(self, mock_openai_class, mock_get):
        """Test batch processing of images."""
        # Mock health check
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_get.return_value = mock_health_response

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OCR result"
        mock_client.chat.completions.create.return_value = mock_response

        client = VLLMClient()
        images = [
            Image.new("RGB", (10, 10), color="red"),
            Image.new("RGB", (10, 10), color="blue"),
        ]

        results = client.process_batch(images)

        assert len(results) == 2
        assert all(r == "OCR result" for r in results)
