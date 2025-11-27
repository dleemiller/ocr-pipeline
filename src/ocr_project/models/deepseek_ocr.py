"""DeepSeek-OCR model wrapper for vLLM integration."""

from collections.abc import Generator
from pathlib import Path

from ocr_project.models.vllm_client import VLLMClient
from ocr_project.utils.image import load_image


class DeepSeekOCR:
    """Wrapper for DeepSeek-OCR model using vLLM backend.

    This class provides a high-level interface for OCR processing using
    the DeepSeek-OCR model with vLLM for efficient batched inference.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000/v1",
        resolution: str = "base",
    ):
        """Initialize the DeepSeek-OCR model with vLLM client.

        Args:
            server_url: URL of the running vLLM server
            resolution: Resolution mode (tiny, small, base, large, gundam)

        Raises:
            RuntimeError: If vLLM server is not running
        """
        self.resolution = resolution
        self.client = VLLMClient(base_url=server_url)

        # Health check on initialization
        if not self.client.health_check():
            raise RuntimeError("vLLM server is not running. Start it with: uv run ocr server")

    def process_image(self, image_path: Path) -> str:
        """Process a single image file and return markdown output.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text in markdown format
        """
        # Load image from path
        image = load_image(image_path)

        # Send to vLLM for processing
        return self.client.process_image(image, self.resolution)

    def process_batch(
        self, image_paths: list[Path]
    ) -> Generator[tuple[Path, str | None], None, None]:
        """Process multiple images, yielding results as they complete.

        This uses a generator pattern to provide incremental results
        and allow progress tracking.

        Args:
            image_paths: List of paths to image files

        Yields:
            (image_path, markdown_text) tuples
            markdown_text will be None if processing failed for that image
        """
        for path in image_paths:
            try:
                result = self.process_image(path)
                yield (path, result)
            except Exception:
                # Log error but continue processing
                # Caller is responsible for handling None results
                yield (path, None)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        # VLLMClient doesn't hold resources that need cleanup
        # Server runs independently
        pass
