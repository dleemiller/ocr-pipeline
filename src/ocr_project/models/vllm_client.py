"""vLLM API client for communicating with OpenAI-compatible server."""

import base64
import io
import time

import httpx
from openai import OpenAI
from PIL import Image


class ServerNotAvailableError(Exception):
    """Raised when vLLM server is not running or not responsive."""

    pass


class APIError(Exception):
    """Raised when API request fails after retries."""

    pass


class VLLMClient:
    """Client for communicating with vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """Initialize OpenAI client for vLLM server.

        Args:
            base_url: Base URL for vLLM OpenAI-compatible API
            timeout: Request timeout in seconds (default 5 minutes)
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Extract base URL without /v1 for health check
        self.health_url = base_url.replace("/v1", "/health")

        # Initialize OpenAI client
        # vLLM OpenAI-compatible server doesn't require a real API key
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't validate API keys
            timeout=timeout,
        )

    def health_check(self) -> bool:
        """Check if vLLM server is running and responsive.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = httpx.get(self.health_url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64-encoded data URI.

        Args:
            image: PIL Image object
            format: Image format (PNG or JPEG)

        Returns:
            Data URI string (e.g., "data:image/png;base64,...")
        """
        # Convert image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Create data URI
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{img_base64}"

    def process_image(
        self, image: Image.Image, resolution: str = "base", prompt: str | None = None
    ) -> str:
        """Send image to vLLM for OCR processing.

        Args:
            image: PIL Image object
            resolution: Resolution mode (tiny/small/base/large/gundam)
            prompt: Optional custom prompt (default: simple OCR instruction)

        Returns:
            Markdown-formatted OCR text

        Raises:
            ServerNotAvailableError: If server is not running
            APIError: If API request fails after retries
        """
        # Check server health first
        if not self.health_check():
            raise ServerNotAvailableError(
                "vLLM server is not running. Start it with: uv run ocr server"
            )

        # Convert image to base64
        image_data_uri = self.image_to_base64(image)

        # Default prompt for OCR
        if prompt is None:
            prompt = "Extract all text from this image and return it in markdown format."

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Create chat completion request with image
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-OCR",  # Model name expected by vLLM
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_data_uri},
                                },
                            ],
                        }
                    ],
                    max_tokens=4096,  # Allow long OCR outputs
                )

                # Extract markdown response
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content or ""
                else:
                    raise APIError("No response content from API")

            except Exception as e:
                last_exception = e

                # If this is the last attempt, raise the error
                if attempt == self.max_retries - 1:
                    raise APIError(
                        f"API request failed after {self.max_retries} retries: {e}"
                    ) from e

                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2**attempt
                time.sleep(wait_time)

        # This should never be reached, but just in case
        raise APIError(f"Unexpected error: {last_exception}")

    def process_batch(self, images: list[Image.Image], resolution: str = "base") -> list[str]:
        """Process multiple images in batch.

        Note: Current implementation processes sequentially.
        Future enhancement: Could use asyncio for concurrent requests.

        Args:
            images: List of PIL Image objects
            resolution: Resolution mode

        Returns:
            List of markdown text results (same order as input)
        """
        results = []
        for image in images:
            try:
                result = self.process_image(image, resolution)
                results.append(result)
            except Exception as e:
                # Continue processing on errors, append empty string
                results.append(f"Error processing image: {e}")
        return results
