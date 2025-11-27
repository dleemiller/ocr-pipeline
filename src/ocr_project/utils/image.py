"""Image loading and preprocessing utilities."""

from pathlib import Path

from PIL import Image


def load_image(image_path: Path) -> Image.Image:
    """Load an image from a file path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)
        img.load()  # Verify image is valid
        return img
    except Exception as e:
        raise ValueError(f"Invalid image file {image_path}: {e}") from e


def get_image_files(directory: Path) -> list[Path]:
    """Get all image files from a directory.

    Args:
        directory: Directory to search for images

    Returns:
        List of image file paths
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
