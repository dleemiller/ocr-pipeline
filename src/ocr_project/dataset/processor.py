"""HuggingFace dataset processor for OCR."""

import io
from collections.abc import Generator
from typing import Any

from datasets import load_dataset
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn

from ocr_project.dataset.config import DatasetConfig, SubsetConfig
from ocr_project.models.vllm_client import VLLMClient
from ocr_project.utils.file_io import save_markdown


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
    ) -> Image.Image | None:
        """Extract image from bytes column.

        Args:
            content_bytes: Raw file bytes
            file_extension: File extension to determine type

        Returns:
            PIL Image if successfully decoded, None otherwise
        """
        try:
            # Try to open as image directly
            return Image.open(io.BytesIO(content_bytes))
        except Exception:
            # If not an image, might need conversion (e.g., PDF)
            # For now, skip non-image content
            return None

    def _process_row(
        self, row: dict[str, Any], subset_config: SubsetConfig
    ) -> list[tuple[str, Image.Image]]:
        """Extract images from a dataset row.

        Args:
            row: Dataset row
            subset_config: Subset configuration

        Returns:
            List of (column_name, image) tuples
        """
        images = []

        # Process image columns
        for col in subset_config.image_columns:
            if col in row and row[col] is not None:
                if isinstance(row[col], Image.Image):
                    images.append((col, row[col]))

        # Process content columns (bytes)
        for col in subset_config.content_columns:
            if col in row and row[col] is not None:
                # Check if it's bytes
                if isinstance(row[col], bytes):
                    file_ext = row.get("extension") if "extension" in row else None
                    img = self._extract_content_from_bytes(row[col], file_ext)
                    if img is not None:
                        images.append((col, img))

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

                # Process each image
                for col_name, image in images:
                    # Generate identifier
                    row_id = row.get("path", f"row_{idx}")
                    identifier = f"{subset_config.name}/{split}/{row_id}/{col_name}"

                    try:
                        # Process with OCR
                        result = self.client.process_image(image, self.resolution)

                        # Generate output path
                        safe_id = row_id.replace("/", "_").replace("\\", "_")
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
