"""HuggingFace dataset processor for OCR."""

import io
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn

from ocr_project.dataset.config import DatasetConfig, SubsetConfig
from ocr_project.models.vllm_client import VLLMClient
from ocr_project.utils.file_io import save_markdown
from ocr_project.utils.pdf import pdf_to_images


class DatasetProcessor:
    """Process HuggingFace datasets for OCR extraction."""

    def __init__(
        self,
        config: DatasetConfig,
        server_url: str = "http://127.0.0.1:8000/v1",
        resolution: str = "base",
    ):
        """Initialize dataset processor.

        Args:
            config: Dataset configuration
            server_url: URL of vLLM server
            resolution: OCR resolution mode
        """
        self.config = config
        self.client = VLLMClient(base_url=server_url)
        self.resolution = resolution

        # Check server health
        if not self.client.health_check():
            raise RuntimeError("vLLM server is not running. Start with: uv run ocr-server")

    def _extract_content_from_bytes(
        self, content_bytes: bytes, file_extension: str | None = None
    ) -> list[tuple[int, Image.Image]]:
        """Extract images from bytes column.

        Args:
            content_bytes: Raw file bytes
            file_extension: File extension to determine type

        Returns:
            List of (page_number, Image) tuples. For images, returns [(1, image)].
            For PDFs, returns [(page_num, image), ...] for each page.
        """
        # Try to open as image directly
        try:
            img = Image.open(io.BytesIO(content_bytes))
            return [(1, img)]
        except Exception:
            pass

        # If extension suggests PDF, try to convert
        if file_extension and file_extension.lower() == ".pdf":
            try:
                # Write bytes to temporary file for pdf2image
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(content_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    # Convert PDF to images
                    pages = pdf_to_images(tmp_path)
                    return pages
                finally:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)

            except Exception:
                pass

        # Unable to extract
        return []

    def _process_row(
        self, row: dict[str, Any], subset_config: SubsetConfig
    ) -> list[tuple[str, int, Image.Image]]:
        """Extract images from a dataset row.

        Args:
            row: Dataset row
            subset_config: Subset configuration

        Returns:
            List of (column_name, page_number, image) tuples
        """
        images = []

        # Check if content is available (if column specified)
        if subset_config.content_available_column:
            if not row.get(subset_config.content_available_column, True):
                # Content not available, skip this row
                return images

        # Process image columns
        for col in subset_config.image_columns:
            if col in row and row[col] is not None:
                if isinstance(row[col], Image.Image):
                    images.append((col, 1, row[col]))

        # Process content columns (bytes)
        for col in subset_config.content_columns:
            if col in row and row[col] is not None:
                # Check if it's bytes
                if isinstance(row[col], bytes):
                    # Get file extension from configured column or fallback to 'extension'
                    file_ext = None
                    if subset_config.extension_column and subset_config.extension_column in row:
                        file_ext = row[subset_config.extension_column]
                    elif "extension" in row:
                        file_ext = row["extension"]

                    pages = self._extract_content_from_bytes(row[col], file_ext)
                    # Add all pages from the content
                    for page_num, img in pages:
                        images.append((col, page_num, img))

        return images

    def _should_process_row(self, row: dict[str, Any], subset_config: SubsetConfig) -> bool:
        """Check if row should be processed based on filters.

        Args:
            row: Dataset row
            subset_config: Subset configuration

        Returns:
            True if row should be processed
        """
        if subset_config.filter_column is None:
            return True

        if subset_config.filter_column not in row:
            return False

        value = row[subset_config.filter_column]
        if subset_config.filter_values is None:
            return True

        return value in subset_config.filter_values

    def process_subset(
        self, subset_config: SubsetConfig
    ) -> Generator[tuple[str, str, str | None], None, None]:
        """Process a single dataset subset.

        Args:
            subset_config: Subset configuration

        Yields:
            Tuple of (identifier, output_path, result or error)
        """
        for split in subset_config.splits:
            # Load dataset
            dataset = load_dataset(
                self.config.name,
                name=subset_config.name,
                split=split,
                streaming=self.config.streaming,
            )

            # Apply max samples limit
            if self.config.max_samples:
                if self.config.streaming:
                    dataset = dataset.take(self.config.max_samples)
                else:
                    dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            # Process each row
            for idx, row in enumerate(dataset):
                # Check filters
                if not self._should_process_row(row, subset_config):
                    continue

                # Extract images from row
                images = self._process_row(row, subset_config)

                # Process each image (including PDF pages)
                for col_name, page_num, image in images:
                    # Generate identifier
                    row_id = row.get("path", f"row_{idx}")
                    if page_num > 1:
                        identifier = (
                            f"{subset_config.name}/{split}/{row_id}/{col_name}/page{page_num}"
                        )
                    else:
                        identifier = f"{subset_config.name}/{split}/{row_id}/{col_name}"

                    try:
                        # Process with OCR
                        result = self.client.process_image(image, self.resolution)

                        # Generate output path
                        safe_id = row_id.replace("/", "_").replace("\\", "_").replace(".pdf", "")
                        if page_num > 1:
                            out_path = (
                                self.config.output_dir
                                / subset_config.name
                                / split
                                / f"{safe_id}_{col_name}_page{page_num:03d}.md"
                            )
                        else:
                            out_path = (
                                self.config.output_dir
                                / subset_config.name
                                / split
                                / f"{safe_id}_{col_name}.md"
                            )

                        # Save result
                        save_markdown(result, out_path)

                        yield (identifier, str(out_path), None)

                    except Exception as e:
                        yield (identifier, "", str(e))

    def process_all(self) -> dict[str, dict[str, int]]:
        """Process all configured subsets.

        Returns:
            Statistics dictionary with counts per subset
        """
        stats = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            for subset_config in self.config.subsets:
                task = progress.add_task(f"Processing {subset_config.name}...", total=None)

                subset_stats = {"success": 0, "error": 0, "total": 0}

                for identifier, output_path, error in self.process_subset(subset_config):
                    subset_stats["total"] += 1

                    if error:
                        subset_stats["error"] += 1
                        progress.console.print(f"[red]Error processing {identifier}: {error}")
                    else:
                        subset_stats["success"] += 1
                        progress.console.print(f"[green]Processed {identifier} -> {output_path}")

                    progress.update(task, advance=1)

                progress.update(
                    task,
                    description=f"[green]Completed {subset_config.name} "
                    f"({subset_stats['success']}/{subset_stats['total']} successful)",
                )
                stats[subset_config.name] = subset_stats

        return stats
