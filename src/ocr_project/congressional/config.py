"""Configuration schemas for Congressional proceedings pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CloudflareConfig:
    """Cloudflare bypass configuration."""

    max_wait_seconds: int = 30
    stealth_mode: bool = True


@dataclass
class ScrapingConfig:
    """Web scraping configuration."""

    headless: bool = True
    max_concurrent: int = 3
    timeout_seconds: int = 60
    cloudflare: CloudflareConfig = field(default_factory=CloudflareConfig)
    include_extensions: list[str] = field(default_factory=lambda: [".pdf", ".mp4", ".mp3"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["*/share/*", "*/print/*"])


@dataclass
class GovInfoAPIConfig:
    """GovInfo API configuration."""

    enabled: bool = True
    api_key: str | None = None
    collections: list[str] = field(default_factory=lambda: ["CREC", "CRECB"])


@dataclass
class CongressGovConfig:
    """Congress.gov scraping configuration."""

    enabled: bool = True
    max_depth: int = 2
    timeout_seconds: int = 60


@dataclass
class YouTubeConfig:
    """YouTube download configuration."""

    enabled: bool = True
    extract_audio: bool = True
    quality: str = "best"


@dataclass
class SourcesConfig:
    """Data sources configuration."""

    govinfo_api: GovInfoAPIConfig = field(default_factory=GovInfoAPIConfig)
    congress_gov: CongressGovConfig = field(default_factory=CongressGovConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)


@dataclass
class OCRConfig:
    """OCR processing configuration."""

    resolution: str = "base"
    batch_size: int = 4
    skip_existing: bool = True


@dataclass
class TranscriptionConfig:
    """Audio transcription configuration."""

    model: str = "base"
    language: str = "en"
    device: str = "cuda"
    batch_size: int = 1


@dataclass
class AgentConfig:
    """DSPy agent configuration."""

    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    enable_few_shot: bool = True
    max_retries: int = 3


@dataclass
class CongressionalConfig:
    """Main configuration for Congressional proceedings pipeline."""

    sources: SourcesConfig = field(default_factory=SourcesConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "CongressionalConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            CongressionalConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CongressionalConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            CongressionalConfig instance
        """
        # Parse nested configurations
        sources_data = data.get("sources", {})
        sources = SourcesConfig(
            govinfo_api=GovInfoAPIConfig(**sources_data.get("govinfo_api", {})),
            congress_gov=CongressGovConfig(**sources_data.get("congress_gov", {})),
            youtube=YouTubeConfig(**sources_data.get("youtube", {})),
        )

        scraping_data = data.get("scraping", {})
        cloudflare_data = scraping_data.pop("cloudflare", {})
        scraping = ScrapingConfig(**scraping_data, cloudflare=CloudflareConfig(**cloudflare_data))

        ocr = OCRConfig(**data.get("ocr", {}))
        transcription = TranscriptionConfig(**data.get("transcription", {}))
        agent = AgentConfig(**data.get("agent", {}))

        return cls(
            sources=sources,
            scraping=scraping,
            ocr=ocr,
            transcription=transcription,
            agent=agent,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "sources": {
                "govinfo_api": {
                    "enabled": self.sources.govinfo_api.enabled,
                    "api_key": self.sources.govinfo_api.api_key,
                    "collections": self.sources.govinfo_api.collections,
                },
                "congress_gov": {
                    "enabled": self.sources.congress_gov.enabled,
                    "max_depth": self.sources.congress_gov.max_depth,
                    "timeout_seconds": self.sources.congress_gov.timeout_seconds,
                },
                "youtube": {
                    "enabled": self.sources.youtube.enabled,
                    "extract_audio": self.sources.youtube.extract_audio,
                    "quality": self.sources.youtube.quality,
                },
            },
            "scraping": {
                "headless": self.scraping.headless,
                "max_concurrent": self.scraping.max_concurrent,
                "timeout_seconds": self.scraping.timeout_seconds,
                "cloudflare": {
                    "max_wait_seconds": self.scraping.cloudflare.max_wait_seconds,
                    "stealth_mode": self.scraping.cloudflare.stealth_mode,
                },
                "include_extensions": self.scraping.include_extensions,
                "exclude_patterns": self.scraping.exclude_patterns,
            },
            "ocr": {
                "resolution": self.ocr.resolution,
                "batch_size": self.ocr.batch_size,
                "skip_existing": self.ocr.skip_existing,
            },
            "transcription": {
                "model": self.transcription.model,
                "language": self.transcription.language,
                "device": self.transcription.device,
                "batch_size": self.transcription.batch_size,
            },
            "agent": {
                "llm_model": self.agent.llm_model,
                "temperature": self.agent.temperature,
                "enable_few_shot": self.agent.enable_few_shot,
                "max_retries": self.agent.max_retries,
            },
        }
