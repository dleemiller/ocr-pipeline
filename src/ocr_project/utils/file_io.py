"""File I/O utilities for reading and writing outputs."""

from pathlib import Path


def save_markdown(content: str, output_path: Path) -> None:
    """Save markdown content to a file.

    Args:
        content: Markdown content to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def read_markdown(input_path: Path) -> str:
    """Read markdown content from a file.

    Args:
        input_path: Path to markdown file

    Returns:
        Markdown content as string
    """
    return input_path.read_text(encoding="utf-8")
