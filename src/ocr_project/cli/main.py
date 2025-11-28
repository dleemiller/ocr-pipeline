"""CLI entry point for OCR processing."""

import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """OCR processing using DeepSeek-OCR and vLLM."""
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file or directory path (default: same dir as input)",
)
@click.option(
    "--resolution",
    "-r",
    type=click.Choice(["tiny", "small", "base", "large", "gundam"]),
    default="base",
    help="Resolution mode for processing",
)
def process(input_path: Path, output: Path | None, resolution: str) -> None:
    """Process a single image or PDF file and extract text as markdown.

    For images: Creates one markdown file
    For PDFs: Creates one markdown file per page

    Args:
        input_path: Path to the image or PDF file
        output: Optional output file/directory path
        resolution: Resolution mode for processing
    """
    from ocr_project.models.deepseek_ocr import DeepSeekOCR
    from ocr_project.utils.file_io import save_markdown
    from ocr_project.utils.pdf import is_pdf, pdf_to_images

    # Initialize model
    try:
        model = DeepSeekOCR(resolution=resolution)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("\nMake sure to start the server first in another terminal:", err=True)
        click.echo("  uv run ocr server", err=True)
        return

    try:
        if is_pdf(input_path):
            # Convert PDF to images
            click.echo(f"Converting PDF: {input_path}")
            pages = pdf_to_images(input_path)
            click.echo(f"Found {len(pages)} page(s)")

            # Process each page
            for page_num, image in pages:
                click.echo(f"Processing page {page_num}...")
                result = model.client.process_image(image, resolution)

                # Determine output path
                if output:
                    # If output is a directory, use it; otherwise use as base path
                    if output.is_dir() or not output.suffix:
                        out_dir = output
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{input_path.stem}_page{page_num:03d}.md"
                    else:
                        # Output specified as file - add page number before extension
                        out_path = (
                            output.parent / f"{output.stem}_page{page_num:03d}{output.suffix}"
                        )
                else:
                    # Save in same directory as input
                    out_path = input_path.parent / f"{input_path.stem}_page{page_num:03d}.md"

                save_markdown(result, out_path)
                click.echo(f"  Saved: {out_path}")

            click.echo(f"\nProcessed {len(pages)} page(s) successfully")

        else:
            # Process single image
            click.echo(f"Processing image: {input_path}")
            result = model.process_image(input_path)

            if output:
                save_markdown(result, output)
                click.echo(f"Saved: {output}")
            else:
                # Output to stdout if no output path specified
                click.echo("\n--- OCR Result ---")
                click.echo(result)

    except Exception as e:
        click.echo(f"Error processing file: {e}", err=True)
        return


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for processed files",
)
@click.option(
    "--resolution",
    "-r",
    type=click.Choice(["tiny", "small", "base", "large", "gundam"]),
    default="base",
    help="Resolution mode for processing",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=4,
    help="(Not currently used - processing is sequential)",
)
def batch(input_dir: Path, output: Path, resolution: str, batch_size: int) -> None:
    """Batch process images and PDFs from a directory.

    Recursively scans input_dir, processes all images and PDFs,
    and mirrors directory structure in output directory.

    Args:
        input_dir: Directory containing images and/or PDFs to process
        output: Output directory for markdown files
        resolution: Resolution mode for processing
        batch_size: Batch size (currently not used - sequential processing)
    """
    from rich.progress import Progress

    from ocr_project.models.deepseek_ocr import DeepSeekOCR
    from ocr_project.utils.file_io import save_markdown
    from ocr_project.utils.pdf import is_pdf, pdf_to_images

    # Initialize model
    try:
        model = DeepSeekOCR(resolution=resolution)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("\nMake sure to start the server first in another terminal:", err=True)
        click.echo("  uv run ocr server", err=True)
        return

    # Collect all files to process
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".pdf"}
    all_files = []

    click.echo(f"Scanning {input_dir}...")
    for file in input_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() in supported_extensions:
            all_files.append(file)

    if not all_files:
        click.echo("No images or PDFs found in directory")
        return

    click.echo(f"Found {len(all_files)} file(s) to process\n")

    # Track errors and statistics
    errors = []
    processed_count = 0
    total_pages = 0

    # Process with progress bar
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(all_files))

        for file_path in all_files:
            try:
                # Determine relative path for mirroring structure
                rel_path = file_path.relative_to(input_dir)

                if is_pdf(file_path):
                    # Process PDF pages
                    pages = pdf_to_images(file_path)

                    for page_num, image in pages:
                        result = model.client.process_image(image, resolution)

                        # Mirror directory structure
                        out_dir = output / rel_path.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{file_path.stem}_page{page_num:03d}.md"

                        save_markdown(result, out_path)
                        processed_count += 1
                        total_pages += 1
                else:
                    # Process single image
                    result = model.process_image(file_path)

                    # Mirror directory structure
                    out_dir = output / rel_path.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{file_path.stem}.md"

                    save_markdown(result, out_path)
                    processed_count += 1

            except Exception as e:
                errors.append((file_path, str(e)))

            progress.update(task, advance=1)

    # Report results
    click.echo(f"\n✓ Successfully processed {processed_count} file(s)")
    if total_pages > 0:
        click.echo(f"  ({total_pages} PDF page(s) total)")

    if errors:
        click.echo(f"\n✗ {len(errors)} file(s) failed:")
        error_report_path = output / "errors.txt"

        with open(error_report_path, "w") as f:
            for file_path, error in errors:
                error_line = f"{file_path}: {error}"
                click.echo(f"  - {error_line}")
                f.write(error_line + "\n")

        click.echo(f"\nError details saved to: {error_report_path}")
    else:
        click.echo("\nAll files processed successfully!")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--server-url",
    type=str,
    default="http://127.0.0.1:8000/v1",
    help="URL of vLLM server",
)
@click.option(
    "--resolution",
    "-r",
    type=click.Choice(["tiny", "small", "base", "large", "gundam"]),
    default="base",
    help="Resolution mode for processing",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output files (default: skip existing)",
)
def dataset(config_path: Path, server_url: str, resolution: str, overwrite: bool) -> None:
    """Process HuggingFace datasets using a YAML configuration file.

    The config file should specify dataset name, subsets, splits,
    and columns to process. See configs/epstractor-raw.yaml for an example.

    Args:
        config_path: Path to YAML configuration file
        server_url: URL of vLLM server
        resolution: Resolution mode for processing
    """
    from ocr_project.dataset.config import DatasetConfig
    from ocr_project.dataset.processor import DatasetProcessor

    # Load configuration
    try:
        config = DatasetConfig.from_yaml(config_path)
        # Override overwrite setting from CLI flag
        if overwrite:
            config.overwrite = True
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        return

    click.echo(f"Processing dataset: {config.name}")
    click.echo(f"Output directory: {config.output_dir}")
    click.echo(f"Subsets: {len(config.subsets)}")
    if config.max_samples:
        click.echo(f"Max samples per subset: {config.max_samples}")
    click.echo(f"Overwrite existing files: {config.overwrite}")
    click.echo()

    # Initialize processor
    try:
        processor = DatasetProcessor(config, server_url, resolution)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("\nMake sure to start the server first in another terminal:", err=True)
        click.echo("  uv run ocr-server", err=True)
        return

    # Process all subsets
    try:
        stats = processor.process_all()

        # Report final statistics
        click.echo("\n" + "=" * 60)
        click.echo("PROCESSING COMPLETE")
        click.echo("=" * 60)
        for subset_name, subset_stats in stats.items():
            click.echo(f"\n{subset_name}:")
            click.echo(f"  Total: {subset_stats['total']}")
            click.echo(f"  Success: {subset_stats['success']}")
            click.echo(f"  Skipped: {subset_stats['skipped']}")
            click.echo(f"  Errors: {subset_stats['error']}")

    except Exception as e:
        click.echo(f"Error processing dataset: {e}", err=True)
        return


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--dataset-name",
    "-n",
    type=str,
    required=True,
    help="Name for the dataset",
)
@click.option(
    "--subset",
    "-s",
    type=str,
    help="Export only a specific subset (default: export all)",
)
@click.option(
    "--split",
    type=str,
    help="Export only a specific split within subset",
)
@click.option(
    "--max-shard-size",
    type=int,
    default=500,
    help="Maximum shard size in MB (default: 500)",
)
@click.option(
    "--ocr-model",
    type=str,
    default="deepseek-ai/DeepSeek-OCR",
    help="OCR model used for processing (for metadata)",
)
@click.option(
    "--resolution",
    type=str,
    default="base",
    help="Resolution mode used (tiny/small/base/large/gundam)",
)
def export(
    output_dir: Path,
    dataset_name: str,
    subset: str | None,
    split: str | None,
    max_shard_size: int,
    ocr_model: str,
    resolution: str,
) -> None:
    """Export OCR markdown files to HuggingFace dataset format.

    Converts a directory of markdown files into parquet format suitable
    for uploading to HuggingFace Hub.

    Args:
        output_dir: Directory containing OCR markdown output files
        dataset_name: Name for the dataset
        subset: Optional specific subset to export
        split: Optional specific split to export
        max_shard_size: Maximum shard size in MB
    """
    from ocr_project.dataset.export import DatasetExporter

    click.echo(f"Exporting dataset: {dataset_name}")
    click.echo(f"Source directory: {output_dir}")
    click.echo(f"OCR model: {ocr_model}")
    click.echo(f"Resolution: {resolution}")
    click.echo(f"Max shard size: {max_shard_size}MB")
    click.echo()

    exporter = DatasetExporter(output_dir, dataset_name, ocr_model, resolution)

    try:
        if subset:
            # Export single subset
            click.echo(f"Exporting subset: {subset}")
            if split:
                click.echo(f"  Split: {split}")

            output_path = exporter.export_subset(subset, split)
            click.echo(f"\n[green]✓ Created: {output_path}")

        else:
            # Export all subsets
            results = exporter.export_all(max_shard_size_mb=max_shard_size)

            click.echo("\n" + "=" * 60)
            click.echo("EXPORT COMPLETE")
            click.echo("=" * 60)

            total_files = sum(len(files) for files in results.values())
            click.echo(f"\nCreated {total_files} parquet file(s)")

            for subset_name, files in results.items():
                click.echo(f"\n{subset_name}:")
                for file_path in files:
                    click.echo(f"  - {file_path.name}")

            # Create dataset card
            card_path = output_dir / "dataset" / "README.md"
            exporter.create_dataset_card(card_path)
            click.echo(f"\n✓ Created dataset card: {card_path}")

            dataset_dir = output_dir / "dataset"
            click.echo("\n[bold green]Dataset ready for upload:")
            click.echo(f"  Directory: {dataset_dir}")
            click.echo("\nTo upload to HuggingFace:")
            click.echo("  huggingface-cli login")
            click.echo(f"  huggingface-cli upload <username>/{dataset_name} {dataset_dir}")

    except Exception as e:
        click.echo(f"Error exporting dataset: {e}", err=True)
        return


@cli.command()
@click.option(
    "--model",
    type=str,
    default="deepseek-ai/DeepSeek-OCR",
    help="Model name or path",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to bind the server to",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization (0.0-1.0)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Number of GPUs for tensor parallelism",
)
def server(
    model: str,
    host: str,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> None:
    """Start vLLM OpenAI-compatible server with DeepSeek-OCR.

    This starts a server that provides an OpenAI-compatible API for OCR processing.
    The server will continue running until stopped with Ctrl+C.
    """
    # Pass arguments via sys.argv for the server module
    import sys

    from ocr_project.server import main as server_main

    original_argv = sys.argv
    sys.argv = [
        "server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]

    try:
        server_main()
    finally:
        sys.argv = original_argv


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        cli()
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
