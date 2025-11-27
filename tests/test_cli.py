"""Tests for CLI commands."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from ocr_project.cli.main import cli


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "OCR processing" in result.output

    def test_process_command_help(self):
        """Test process command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process a single image" in result.output

    def test_batch_command_help(self):
        """Test batch command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Batch process" in result.output

    def test_process_command_nonexistent_file(self):
        """Test process command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "/nonexistent/file.png"])
        assert result.exit_code != 0

    def test_process_command_with_temp_file_no_server(self):
        """Test process command fails gracefully when server is not running."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["process", str(temp_path)])
            # Command should handle server error gracefully (not crash)
            assert "vLLM server is not running" in result.output
        finally:
            temp_path.unlink()

    def test_batch_command_with_temp_dir_no_server(self):
        """Test batch command fails gracefully when server is not running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            runner = CliRunner()
            result = runner.invoke(cli, ["batch", tmpdir, "--output", str(output_dir)])
            # Command should handle server error gracefully (not crash)
            assert "vLLM server is not running" in result.output
