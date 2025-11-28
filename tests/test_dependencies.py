"""Integration tests for external dependencies.

These tests verify that critical external tools are installed and working:
- Playwright (browser automation)
- yt-dlp (YouTube downloading)
- Whisper (audio transcription)
- HTTP downloads (requests/httpx)

Run with: pytest tests/test_dependencies.py -v
"""

import tempfile
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_playwright_installed():
    """Test that Playwright is installed and browsers are available."""
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            # Try to launch chromium
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to a simple page
            await page.goto("https://example.com")
            title = await page.title()

            await browser.close()

            assert title == "Example Domain"
            print("✓ Playwright working - browser automation ready")

    except ImportError as e:
        pytest.fail(f"Playwright not installed: {e}")
    except Exception as e:
        # If browsers not installed, provide helpful message
        if "executable doesn't exist" in str(e).lower():
            pytest.fail("Playwright browsers not installed. Run: playwright install chromium")
        raise


def test_ytdlp_available():
    """Test that yt-dlp is available and can extract info."""
    try:
        import yt_dlp

        # Test with a known stable video (YouTube's own example)
        # This is a very short test video
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # Don't download, just get metadata
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(test_url, download=False)

            assert info is not None
            assert "title" in info
            print(f"✓ yt-dlp working - extracted: {info.get('title', 'N/A')}")

    except ImportError as e:
        pytest.fail(f"yt-dlp not installed: {e}")
    except Exception as e:
        # Network issues are acceptable for this test
        if "unable to extract" in str(e).lower() or "network" in str(e).lower():
            pytest.skip(f"Network issue during yt-dlp test: {e}")
        raise


def test_whisper_available():
    """Test that Whisper is available and model can load."""
    try:
        import whisper

        # Try to load the smallest model (base)
        print("Loading Whisper 'tiny' model (this may take a moment)...")
        model = whisper.load_model("tiny")

        assert model is not None
        print("✓ Whisper working - 'tiny' model loaded successfully")

        # Optional: Test with a very short audio clip
        # (Skipped for speed, but could be added if needed)

    except ImportError as e:
        pytest.fail(f"Whisper not installed: {e}")
    except Exception as e:
        if "out of memory" in str(e).lower():
            pytest.skip("Not enough memory to load Whisper model")
        raise


def test_http_download():
    """Test that HTTP downloads work (for PDFs)."""
    try:
        # Download a small test file
        test_url = "https://httpbin.org/bytes/1024"  # 1KB of random data

        response = httpx.get(test_url, timeout=10.0)
        response.raise_for_status()

        assert len(response.content) == 1024
        print("✓ HTTP downloads working")

    except httpx.RequestError as e:
        pytest.skip(f"Network issue during HTTP test: {e}")
    except Exception:
        raise


def test_pdf_download_and_convert():
    """Test downloading a PDF and converting to images."""
    try:
        import pdf2image
        from PIL import Image

        # Use a very small public domain PDF
        # This is a single-page test PDF from PDF Association
        pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download PDF
            response = httpx.get(pdf_url, timeout=10.0)
            response.raise_for_status()

            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(response.content)

            # Convert to images
            images = pdf2image.convert_from_path(pdf_path)

            assert len(images) > 0
            assert isinstance(images[0], Image.Image)
            print(f"✓ PDF download and conversion working ({len(images)} page(s))")

    except ImportError as e:
        pytest.fail(f"pdf2image not installed: {e}")
    except httpx.RequestError as e:
        pytest.skip(f"Network issue during PDF download: {e}")
    except Exception as e:
        if "poppler" in str(e).lower():
            pytest.fail("poppler not installed. Install with: sudo apt-get install poppler-utils")
        raise


def test_beautifulsoup_parsing():
    """Test that BeautifulSoup HTML parsing works."""
    try:
        from bs4 import BeautifulSoup

        html = """
        <html>
            <body>
                <h1>Test Heading</h1>
                <a href="https://example.com/doc.pdf">Document Link</a>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        heading = soup.find("h1")
        link = soup.find("a")

        assert heading.text == "Test Heading"
        assert link["href"] == "https://example.com/doc.pdf"
        print("✓ BeautifulSoup HTML parsing working")

    except ImportError as e:
        pytest.fail(f"BeautifulSoup not installed: {e}")


def test_dspy_import():
    """Test that DSPy can be imported."""
    try:
        import dspy

        # Just verify import works
        assert hasattr(dspy, "Module")
        assert hasattr(dspy, "Signature")
        print("✓ DSPy available and importable")

    except ImportError as e:
        pytest.fail(f"DSPy not installed: {e}")


def test_pydantic_validation():
    """Test that Pydantic validation works as expected."""
    try:
        from pydantic import BaseModel, HttpUrl, ValidationError

        class TestModel(BaseModel):
            url: HttpUrl
            count: int

        # Valid data
        valid = TestModel(url="https://example.com", count=5)
        assert str(valid.url) == "https://example.com/"

        # Invalid URL should raise
        with pytest.raises(ValidationError):
            TestModel(url="not-a-url", count=5)

        print("✓ Pydantic validation working")

    except ImportError as e:
        pytest.fail(f"Pydantic not installed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
