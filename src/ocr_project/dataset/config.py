"""Configuration schema for HuggingFace dataset processing."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SubsetConfig:
    """Configuration for a dataset subset/config."""

    name: str
    splits: list[str]
    content_columns: list[str]
    image_columns: list[str] = field(default_factory=list)
    filter_column: str | None = None
    filter_values: list[str] | None = None
    extension_column: str | None = None  # Column containing file extensions
    content_available_column: str | None = None  # Column indicating if content is available


@dataclass
class DatasetConfig:
    """Configuration for processing a HuggingFace dataset."""

    name: str
    subsets: list[SubsetConfig]
    output_dir: Path
    streaming: bool = True
    max_samples: int | None = None
    overwrite: bool = False  # Whether to overwrite existing output files

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            DatasetConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        subsets = [
            SubsetConfig(
                name=s["name"],
                splits=s["splits"],
                content_columns=s["content_columns"],
                image_columns=s.get("image_columns", []),
                filter_column=s.get("filter_column"),
                filter_values=s.get("filter_values"),
                extension_column=s.get("extension_column"),
                content_available_column=s.get("content_available_column"),
            )
            for s in data["subsets"]
        ]

        return cls(
            name=data["name"],
            subsets=subsets,
            output_dir=Path(data["output_dir"]),
            streaming=data.get("streaming", True),
            max_samples=data.get("max_samples"),
            overwrite=data.get("overwrite", False),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration file
        """
        data = {
            "name": self.name,
            "output_dir": str(self.output_dir),
            "streaming": self.streaming,
            "max_samples": self.max_samples,
            "overwrite": self.overwrite,
            "subsets": [
                {
                    "name": s.name,
                    "splits": s.splits,
                    "content_columns": s.content_columns,
                    "image_columns": s.image_columns,
                    "filter_column": s.filter_column,
                    "filter_values": s.filter_values,
                    "extension_column": s.extension_column,
                    "content_available_column": s.content_available_column,
                }
                for s in self.subsets
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
