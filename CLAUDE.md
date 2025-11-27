# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Purpose:**
This project provides a high-performance OCR (Optical Character Recognition) system using DeepSeek-OCR and vLLM. It focuses on converting images and documents to structured markdown format with support for batched inference and multiple resolution modes.

**Key Components:**
- **CLI Module** (`src/ocr_project/cli/`): Command-line interface for user interaction
- **Models Module** (`src/ocr_project/models/`): DeepSeek-OCR model wrapper and vLLM integration
- **Utils Module** (`src/ocr_project/utils/`): Helper functions for image processing, file I/O, and formatting

## Development Environment

**Python:** 3.12 (managed with uv)
**Package Manager:** uv
**Linting/Formatting:** ruff
**Testing:** pytest (target 70% coverage)
**Pre-commit:** Used for code quality checks
**GPU Requirements:** NVIDIA GPU with CUDA 11.8+ (~6GB VRAM minimum)

## Common Commands

**Development:**
```bash
# Install dependencies
uv sync

# Run linting/formatting
ruff check .
ruff format .

# Run tests
uv run pytest
uv run pytest tests/test_file.py              # Single file
uv run pytest tests/test_file.py::test_fn     # Single test

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

**Project Commands:**
```bash
# Basic OCR
uv run ocr process image.png

# With resolution mode
uv run ocr process image.png --resolution base

# Batch processing
uv run ocr batch input_dir/ --output output_dir/

# Help
uv run ocr --help
```

## Architecture Guidelines

**Code Quality:**
- Follow DRY principles rigorously
- Target 70% test coverage for critical components
- Use type hints throughout (especially for vLLM and model interfaces)
- Maintain clean separation between CLI, models, and utilities
- Keep functions focused and small
- Handle GPU memory efficiently (use context managers for model loading)

**Dependencies:**
- **vLLM**: High-performance inference engine with prefix caching and batching
- **DeepSeek-OCR**: The core OCR model (deepseek-ai/DeepSeek-OCR, 3B params)
- **torch**: PyTorch backend for model operations
- **transformers**: Model loading and tokenization
- **Pillow**: Image loading and preprocessing
- **click**: CLI framework for user-friendly command-line interface

## Project Structure

```
src/ocr_project/
├── __init__.py           # Package version and exports
├── cli/
│   ├── __init__.py
│   └── main.py          # CLI entry point with click commands
├── models/
│   ├── __init__.py
│   ├── deepseek_ocr.py  # DeepSeek-OCR wrapper class
│   └── vllm_engine.py   # vLLM engine initialization and batching
└── utils/
    ├── __init__.py
    ├── image.py         # Image loading and preprocessing
    ├── file_io.py       # File I/O operations
    └── markdown.py      # Markdown formatting utilities
```

## Testing Strategy

- Target 70%+ coverage on critical paths (model loading, inference, CLI commands)
- Use pytest fixtures for model mocking (avoid loading actual models in tests)
- Mock GPU operations and vLLM calls (use CPU-only test mode)
- Test both happy paths and error cases (missing files, invalid images, OOM errors)
- Keep tests fast and independent
- Test CLI commands using click.testing.CliRunner
- Verify markdown output formatting

## Model Integration Notes

**DeepSeek-OCR via vLLM:**
- Initialize with `LLM()` from vLLM
- Use prefix caching for repeated prompts
- Support batched inference for multiple images
- Resolution modes: tiny, small, base, large, gundam
- Expect outputs in markdown format
- Handle model download and caching automatically

**GPU Memory Management:**
- Model requires ~6GB VRAM minimum
- Use `torch.cuda.empty_cache()` after batch processing
- Implement batch size limits based on available memory
- Provide CPU fallback warnings (not officially supported but graceful degradation)

## Common Pitfalls to Avoid

- Always use `uv run` for executing Python commands (dependencies are in uv-managed venv)
- Don't commit model weights or cache files
- Don't commit test images or output files (except small examples in examples/)
- Don't skip pre-commit hooks
- Follow existing code style and patterns
- Update tests when changing functionality
- Handle GPU out-of-memory errors gracefully
- Don't assume GPU availability - check and provide helpful error messages
- Use type hints for vLLM types (can be complex generics)
- Cache model instances appropriately (don't reload for every image)

## Environment Variables

Optional configuration via environment variables:
- `OCR_MODEL_PATH`: Override default model path (default: deepseek-ai/DeepSeek-OCR)
- `OCR_CACHE_DIR`: Model cache directory (default: ~/.cache/ocr-project)
- `OCR_BATCH_SIZE`: Default batch size for processing (default: auto-detect based on VRAM)
- `CUDA_VISIBLE_DEVICES`: GPU selection for multi-GPU systems

## Performance Considerations

- Use batching whenever processing multiple images
- Enable prefix caching in vLLM for repeated operations
- Monitor GPU memory usage during batch processing
- Consider resolution mode vs quality tradeoffs
- Profile critical paths if performance issues arise
