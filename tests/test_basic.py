"""Basic tests to verify project setup."""


def test_imports():
    """Test that package can be imported."""
    import ocr_project

    assert ocr_project.__version__ == "0.1.0"


def test_cli_imports():
    """Test that CLI module can be imported."""
    from ocr_project.cli import main

    assert callable(main.main)


def test_models_imports():
    """Test that models module can be imported."""
    from ocr_project.models.deepseek_ocr import DeepSeekOCR

    assert DeepSeekOCR is not None


def test_utils_imports():
    """Test that utils modules can be imported."""
    from ocr_project.utils import file_io, image, pdf

    assert callable(image.load_image)
    assert callable(file_io.save_markdown)
    assert callable(pdf.pdf_to_images)
    assert callable(pdf.is_pdf)


def test_vllm_client_import():
    """Test that vLLM client can be imported."""
    from ocr_project.models.vllm_client import VLLMClient

    assert VLLMClient is not None
