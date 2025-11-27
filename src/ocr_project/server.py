"""vLLM OpenAI-compatible server for DeepSeek-OCR."""

import argparse
import subprocess
import sys


def main() -> int:
    """Start vLLM OpenAI-compatible server with DeepSeek-OCR.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Start vLLM server for DeepSeek-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  uv run ocr-server

  # Start with custom host/port
  uv run ocr-server --host 0.0.0.0 --port 8080

  # Start with custom model path
  uv run ocr-server --model /path/to/model

  # Adjust GPU memory and tensor parallelism
  uv run ocr-server --gpu-memory-utilization 0.95 --tensor-parallel-size 1
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Model name or path (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0, default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (default: model's max)",
    )

    args = parser.parse_args()

    print("Starting vLLM server for DeepSeek-OCR...")
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print()
    print("Once started, access the API at:")
    print(f"  OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1")
    print(f"  Health check: http://{args.host}:{args.port}/health")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print()

    # Build vllm serve command
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--trust-remote-code",
    ]

    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    try:
        # Run the vLLM server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
