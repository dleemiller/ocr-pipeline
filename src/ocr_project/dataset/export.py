"""Export OCR results to HuggingFace dataset format."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import Progress


class DatasetExporter:
    """Export OCR markdown files to HuggingFace dataset format."""

    def __init__(self, output_dir: Path, dataset_name: str):
        """Initialize dataset exporter.

        Args:
            output_dir: Directory containing OCR markdown output files
            dataset_name: Name for the dataset
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

    def collect_files(self, subset_dir: Path) -> list[dict[str, Any]]:
        """Collect all markdown files from a subset directory.

        Args:
            subset_dir: Directory containing markdown files for a subset

        Returns:
            List of file records with metadata
        """
        records = []

        for md_file in subset_dir.rglob("*.md"):
            # Read markdown content
            with open(md_file, encoding="utf-8") as f:
                text = f.read()

            # Extract metadata from path
            relative_path = md_file.relative_to(subset_dir)
            parts = relative_path.parts

            # Determine split from path structure (typically train/test/validation)
            split = parts[0] if len(parts) > 1 else "train"

            # Get filename without extension
            filename = md_file.stem

            # Check if it's a multi-page document
            page_num = None
            if "_page" in filename:
                # Extract page number from filename like "doc_page001"
                base_name = filename.rsplit("_page", 1)[0]
                page_str = filename.rsplit("_page", 1)[1]
                try:
                    page_num = int(page_str)
                except ValueError:
                    base_name = filename
            else:
                base_name = filename

            record = {
                "id": str(relative_path),
                "source_file": base_name,
                "split": split,
                "page_number": page_num,
                "text": text,
                "text_length": len(text),
                "file_path": str(relative_path),
            }

            records.append(record)

        return records

    def create_parquet(self, records: list[dict[str, Any]], output_path: Path) -> None:
        """Create parquet file from records.

        Args:
            records: List of file records
            output_path: Path to save parquet file
        """
        # Convert to pandas DataFrame
        df = pd.DataFrame(records)

        # Define schema with proper types
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("source_file", pa.string()),
                ("split", pa.string()),
                ("page_number", pa.int32()),
                ("text", pa.string()),
                ("text_length", pa.int64()),
                ("file_path", pa.string()),
            ]
        )

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df, schema=schema)

        # Write parquet file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")

    def export_subset(self, subset_name: str, split: str | None = None) -> Path:
        """Export a single subset to parquet.

        Args:
            subset_name: Name of the subset to export
            split: Optional split name (if None, processes all splits)

        Returns:
            Path to created parquet file
        """
        subset_dir = self.output_dir / subset_name
        if not subset_dir.exists():
            raise ValueError(f"Subset directory not found: {subset_dir}")

        # Collect all files
        records = self.collect_files(subset_dir)

        # Filter by split if specified
        if split:
            records = [r for r in records if r["split"] == split]

        if not records:
            raise ValueError(f"No records found for subset {subset_name}")

        # Create output filename
        if split:
            output_filename = f"{subset_name}-{split}.parquet"
        else:
            output_filename = f"{subset_name}.parquet"

        output_path = self.output_dir / "dataset" / output_filename

        # Create parquet file
        self.create_parquet(records, output_path)

        return output_path

    def export_all(self, max_shard_size_mb: int = 500) -> dict[str, list[Path]]:
        """Export all subsets to HuggingFace dataset format.

        Args:
            max_shard_size_mb: Maximum size per parquet shard in MB

        Returns:
            Dictionary mapping subset names to list of parquet file paths
        """
        results = {}

        # Find all subset directories
        subset_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name != "dataset"]

        with Progress() as progress:
            task = progress.add_task("[cyan]Exporting datasets...", total=len(subset_dirs))

            for subset_dir in subset_dirs:
                subset_name = subset_dir.name
                progress.console.print(f"\n[bold]Processing subset: {subset_name}")

                # Collect all files
                records = self.collect_files(subset_dir)

                if not records:
                    progress.console.print(f"[yellow]No files found in {subset_name}")
                    progress.update(task, advance=1)
                    continue

                # Group by split
                splits = {}
                for record in records:
                    split = record["split"]
                    if split not in splits:
                        splits[split] = []
                    splits[split].append(record)

                subset_files = []

                # Export each split
                for split, split_records in splits.items():
                    # Check if we need to shard
                    df = pd.DataFrame(split_records)
                    size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

                    if size_mb > max_shard_size_mb:
                        # Calculate number of shards needed
                        num_shards = int(size_mb / max_shard_size_mb) + 1
                        shard_size = len(split_records) // num_shards

                        progress.console.print(
                            f"  Sharding {split} into {num_shards} files ({size_mb:.1f}MB total)"
                        )

                        # Create shards
                        for i in range(num_shards):
                            start_idx = i * shard_size
                            end_idx = (
                                start_idx + shard_size if i < num_shards - 1 else len(split_records)
                            )
                            shard_records = split_records[start_idx:end_idx]

                            output_path = (
                                self.output_dir
                                / "dataset"
                                / f"{subset_name}-{split}-{i:05d}-of-{num_shards:05d}.parquet"
                            )
                            self.create_parquet(shard_records, output_path)
                            subset_files.append(output_path)

                    else:
                        # Single file for this split
                        output_path = self.output_dir / "dataset" / f"{subset_name}-{split}.parquet"
                        self.create_parquet(split_records, output_path)
                        subset_files.append(output_path)

                results[subset_name] = subset_files
                progress.console.print(
                    f"[green]✓ {subset_name}: {len(records)} records -> {len(subset_files)} file(s)"
                )

                progress.update(task, advance=1)

        return results

    def create_dataset_card(self, output_path: Path | None = None) -> str:
        """Create a dataset card (README.md) for HuggingFace.

        Args:
            output_path: Optional path to save the card

        Returns:
            Dataset card content as string
        """
        # Count files and records
        total_files = 0
        total_records = 0
        subset_stats = {}

        for subset_dir in self.output_dir.iterdir():
            if subset_dir.is_dir() and subset_dir.name != "dataset":
                records = self.collect_files(subset_dir)
                subset_stats[subset_dir.name] = {
                    "files": len(list(subset_dir.rglob("*.md"))),
                    "records": len(records),
                }
                total_files += subset_stats[subset_dir.name]["files"]
                total_records += len(records)

        card = f"""---
license: mit
task_categories:
- text-generation
- text-retrieval
language:
- en
size_categories:
- {self._get_size_category(total_records)}
---

# {self.dataset_name}

This dataset contains OCR-extracted text from documents processed using DeepSeek-OCR.

## Dataset Summary

- **Total Records**: {total_records:,}
- **Total Source Files**: {total_files:,}
- **Format**: Parquet (Snappy compression)

## Dataset Structure

### Data Fields

- `id`: Unique identifier for the record
- `source_file`: Original source filename (without page numbers)
- `split`: Dataset split (train/test/validation)
- `page_number`: Page number for multi-page documents (null for single-page)
- `text`: OCR-extracted text in markdown format
- `text_length`: Length of extracted text in characters
- `file_path`: Relative path to the source markdown file

### Subsets

"""

        for subset_name, stats in subset_stats.items():
            card += (
                f"- **{subset_name}**: {stats['records']:,} records from {stats['files']:,} files\n"
            )

        card += """
## Usage

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset("path/to/dataset")

# Load specific subset
dataset = load_dataset("path/to/dataset", name="subset_name")

# Load with streaming for large datasets
dataset = load_dataset("path/to/dataset", streaming=True)
```

## Source

This dataset was created using the OCR Pipeline with DeepSeek-OCR and vLLM.

## License

MIT License
"""

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(card)

        return card

    def _get_size_category(self, num_records: int) -> str:
        """Get HuggingFace size category based on number of records."""
        if num_records < 1000:
            return "n<1K"
        elif num_records < 10000:
            return "1K<n<10K"
        elif num_records < 100000:
            return "10K<n<100K"
        elif num_records < 1000000:
            return "100K<n<1M"
        else:
            return "1M<n<10M"
