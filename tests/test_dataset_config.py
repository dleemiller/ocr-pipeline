"""Tests for dataset configuration."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from ocr_project.dataset.config import DatasetConfig, SubsetConfig


class TestSubsetConfig:
    """Tests for SubsetConfig class."""

    def test_init(self):
        """Test SubsetConfig initialization."""
        config = SubsetConfig(
            name="test_subset",
            splits=["train", "test"],
            content_columns=["content"],
            image_columns=["image"],
            filter_column="file_type",
            filter_values=["image", "document"],
        )

        assert config.name == "test_subset"
        assert config.splits == ["train", "test"]
        assert config.content_columns == ["content"]
        assert config.image_columns == ["image"]
        assert config.filter_column == "file_type"
        assert config.filter_values == ["image", "document"]

    def test_init_minimal(self):
        """Test SubsetConfig with minimal parameters."""
        config = SubsetConfig(
            name="minimal",
            splits=["train"],
            content_columns=["content"],
        )

        assert config.name == "minimal"
        assert config.splits == ["train"]
        assert config.content_columns == ["content"]
        assert config.image_columns == []
        assert config.filter_column is None
        assert config.filter_values is None


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_init(self):
        """Test DatasetConfig initialization."""
        subset = SubsetConfig(
            name="test_subset",
            splits=["train"],
            content_columns=["content"],
        )

        config = DatasetConfig(
            name="test/dataset",
            subsets=[subset],
            output_dir=Path("./output"),
            streaming=True,
            max_samples=100,
        )

        assert config.name == "test/dataset"
        assert len(config.subsets) == 1
        assert config.output_dir == Path("./output")
        assert config.streaming is True
        assert config.max_samples == 100

    def test_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
name: test/dataset
output_dir: ./test_output
streaming: true
max_samples: 50

subsets:
  - name: subset1
    splits:
      - train
      - test
    content_columns:
      - content
    image_columns:
      - image
    filter_column: file_type
    filter_values:
      - image
      - document
"""

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = Path(f.name)

        try:
            config = DatasetConfig.from_yaml(temp_path)

            assert config.name == "test/dataset"
            assert config.output_dir == Path("./test_output")
            assert config.streaming is True
            assert config.max_samples == 50
            assert len(config.subsets) == 1

            subset = config.subsets[0]
            assert subset.name == "subset1"
            assert subset.splits == ["train", "test"]
            assert subset.content_columns == ["content"]
            assert subset.image_columns == ["image"]
            assert subset.filter_column == "file_type"
            assert subset.filter_values == ["image", "document"]

        finally:
            temp_path.unlink()

    def test_to_yaml(self):
        """Test saving configuration to YAML."""
        subset = SubsetConfig(
            name="test_subset",
            splits=["train"],
            content_columns=["content"],
            image_columns=["image"],
        )

        config = DatasetConfig(
            name="test/dataset",
            subsets=[subset],
            output_dir=Path("./output"),
            streaming=False,
            max_samples=None,
        )

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.to_yaml(temp_path)

            # Load it back and verify
            loaded_config = DatasetConfig.from_yaml(temp_path)

            assert loaded_config.name == config.name
            assert loaded_config.output_dir == config.output_dir
            assert loaded_config.streaming == config.streaming
            assert loaded_config.max_samples == config.max_samples
            assert len(loaded_config.subsets) == 1

        finally:
            temp_path.unlink()

    def test_roundtrip_yaml(self):
        """Test YAML save and load roundtrip."""
        subset1 = SubsetConfig(
            name="subset1",
            splits=["train", "test"],
            content_columns=["content", "data"],
            image_columns=["image"],
            filter_column="type",
            filter_values=["image"],
        )

        subset2 = SubsetConfig(
            name="subset2",
            splits=["validation"],
            content_columns=["bytes"],
        )

        config = DatasetConfig(
            name="test/roundtrip",
            subsets=[subset1, subset2],
            output_dir=Path("./test_output"),
            streaming=True,
            max_samples=1000,
        )

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save and load
            config.to_yaml(temp_path)
            loaded = DatasetConfig.from_yaml(temp_path)

            # Verify all fields match
            assert loaded.name == config.name
            assert loaded.output_dir == config.output_dir
            assert loaded.streaming == config.streaming
            assert loaded.max_samples == config.max_samples
            assert len(loaded.subsets) == 2

            # Verify first subset
            assert loaded.subsets[0].name == subset1.name
            assert loaded.subsets[0].splits == subset1.splits
            assert loaded.subsets[0].content_columns == subset1.content_columns
            assert loaded.subsets[0].image_columns == subset1.image_columns
            assert loaded.subsets[0].filter_column == subset1.filter_column
            assert loaded.subsets[0].filter_values == subset1.filter_values

            # Verify second subset
            assert loaded.subsets[1].name == subset2.name
            assert loaded.subsets[1].splits == subset2.splits
            assert loaded.subsets[1].content_columns == subset2.content_columns

        finally:
            temp_path.unlink()
