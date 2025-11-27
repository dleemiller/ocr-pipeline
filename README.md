# OCR Project

A Python-based OCR (Optical Character Recognition) tool using DeepSeek-OCR and vLLM for efficient document-to-markdown conversion.

## Features

- **DeepSeek-OCR Integration**: Utilizes the 3B parameter DeepSeek-OCR model for high-quality text extraction
- **vLLM Backend**: Leverages vLLM for fast, batched inference with prefix caching support
- **Document-to-Markdown**: Converts images and documents to structured markdown format
- **Multiple Resolution Modes**: Support for Tiny, Small, Base, Large, and Gundam resolution variants
- **Grounding & Free OCR**: Flexible OCR modes for different use cases
- **CLI Interface**: Easy-to-use command-line interface for batch processing

## Quick Start

```bash
# Clone and setup
cd ocr-project
uv sync

# Run tests
uv run pytest

# Run the CLI
uv run ocr --help
```

## Installation

```bash
# Install dependencies with CUDA 12.8 support
uv sync --extra cu128

# For development (includes testing tools)
uv sync --extra dev --extra cu128
```

**Requirements:**
- Python 3.12+
- NVIDIA GPU with CUDA 12.8
- ~6GB VRAM for the 3B parameter model

## Usage

### Starting the vLLM Server

```bash
# Start server with default settings (recommended)
uv run ocr server

# Or use the standalone command
uv run ocr-server

# Start on all interfaces with custom port
uv run ocr server --host 0.0.0.0 --port 8080

# Adjust GPU memory utilization
uv run ocr server --gpu-memory-utilization 0.95

# Use custom model
uv run ocr server --model /path/to/model
```

The server provides an OpenAI-compatible API at:
- **API Endpoint**: `http://127.0.0.1:8000/v1`
- **Health Check**: `http://127.0.0.1:8000/health`
- **API Docs**: `http://127.0.0.1:8000/docs`

### Processing Files with the CLI

**Important**: Make sure the vLLM server is running first (see above).

```bash
# Process a single image
uv run ocr process image.png

# Process a single image and save to file
uv run ocr process image.png --output result.md

# Process a PDF (creates one .md file per page)
uv run ocr process document.pdf --output output_dir/

# Process with specific resolution mode
uv run ocr process scan.jpg --resolution large

# Batch process a directory (images and PDFs)
uv run ocr batch input_docs/ --output processed/

# Batch process with custom resolution
uv run ocr batch scans/ --output results/ --resolution base
```

### Processing HuggingFace Datasets

The CLI supports processing HuggingFace datasets using YAML configuration files. This is useful for processing large-scale document collections.

```bash
# Process a dataset using a configuration file
uv run ocr dataset configs/epstractor-raw.yaml

# Use custom server URL
uv run ocr dataset configs/my-dataset.yaml --server-url http://localhost:8080/v1

# Use specific resolution mode
uv run ocr dataset configs/my-dataset.yaml --resolution large
```

#### Dataset Configuration Format

Create a YAML configuration file (see `configs/epstractor-raw.yaml` for a complete example):

```yaml
name: user/dataset-name              # HuggingFace dataset identifier
output_dir: ./output/dataset-name    # Where to save processed files
streaming: true                      # Use streaming mode (recommended for large datasets)
max_samples: null                    # Limit samples per subset (null = process all)

subsets:
  - name: subset1                    # Subset/config name
    splits:                          # List of splits to process
      - train
      - test
    content_columns:                 # Columns containing bytes/binary data
      - content
      - data
    image_columns:                   # Columns containing PIL Images
      - image
    filter_column: file_type         # Optional: column to filter by
    filter_values:                   # Optional: values to include
      - image
      - document
```

**Key Features:**
- **Multi-subset support**: Process multiple configurations in one run
- **Flexible column mapping**: Handle bytes columns, image columns, or both
- **Filtering**: Process only specific file types or categories
- **Streaming**: Efficient processing of large datasets without loading all data into memory
- **Progress tracking**: Real-time progress bars and statistics

**Example with epstractor-raw dataset:**

This dataset contains public records with document images. The provided configuration processes three subsets:
- `epstein_estate_2025_09`: 5 files (0.09 GB)
- `epstein_estate_2025_11`: 26,035 files (36.56 GB)
- `house_doj_2025_09`: 33,380 files (78.58 GB)

For testing, set `max_samples: 10` in the config to process only a small subset.

### Output Structure

The batch command mirrors your input directory structure:

```
input_docs/
├── report.pdf (3 pages)
├── photo.jpg
└── subfolder/
    └── scan.png

processed/
├── report_page001.md
├── report_page002.md
├── report_page003.md
├── photo.md
└── subfolder/
    └── scan.md
```

If any files fail to process, an `errors.txt` file will be created in the output directory with details.

## Development

```bash
# Run linting
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Testing

Run tests with:

```bash
uv run pytest                              # All tests
uv run pytest tests/test_file.py           # Single file
uv run pytest tests/test_file.py::test_fn  # Single test
uv run pytest -v                           # Verbose output
```

## Project Structure

```
ocr-project/
├── src/ocr_project/
│   ├── cli/           # Command-line interface
│   ├── models/        # DeepSeek-OCR model wrapper and vLLM client
│   ├── dataset/       # HuggingFace dataset processing
│   └── utils/         # Utility functions (image, PDF, file I/O)
├── tests/             # Test suite (43 tests, 41 passing)
├── configs/           # Dataset configuration files (YAML)
└── output/            # Default output directory
```

## Model Information

This project uses **DeepSeek-OCR** (deepseek-ai/DeepSeek-OCR):
- 3B parameters in bfloat16 precision
- MIT licensed and open-source
- Officially supported in vLLM (as of Oct 2025)
- Supports multiple resolution modes for quality/speed tradeoffs

## License

MIT License
