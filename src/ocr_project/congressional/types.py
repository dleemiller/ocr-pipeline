"""Type definitions for Congressional proceedings data collection."""

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, HttpUrl


class SourceType(str, Enum):
    """Data source type."""

    GOVINFO_API = "govinfo_api"
    CONGRESS_GOV = "congress_gov"
    YOUTUBE = "youtube"


class ResourceType(str, Enum):
    """Resource type."""

    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    HTML = "html"


class ProcessingStatus(str, Enum):
    """Processing status."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResourceRecord(BaseModel):
    """Record of a discovered resource."""

    url: HttpUrl
    resource_type: ResourceType
    source_type: SourceType
    filename: str
    local_path: Path | None = None

    # Metadata
    discovered_at: datetime
    priority: int

    # Processing status
    download_status: ProcessingStatus = ProcessingStatus.PENDING
    ocr_status: ProcessingStatus | None = None
    transcription_status: ProcessingStatus | None = None

    # Output paths
    ocr_output_path: Path | None = None
    transcription_output_path: Path | None = None

    # Errors
    error_message: str | None = None
    retry_count: int = 0


class ProceedingMetadata(BaseModel):
    """Metadata about a Congressional proceeding."""

    url: HttpUrl
    proceeding_id: str
    title: str
    date: datetime | None = None
    chamber: str | None = None  # House, Senate, Joint
    committee: str | None = None
    congress_number: int | None = None
    session_type: str | None = None  # hearing, markup, business meeting

    # Participants
    witnesses: list[str] = []
    members: list[str] = []

    # Content
    description: str | None = None
    topics: list[str] = []


class CollectionManifest(BaseModel):
    """Collection manifest tracking all resources and processing status."""

    metadata: ProceedingMetadata

    # Timing
    started_at: datetime
    completed_at: datetime | None = None
    last_updated: datetime

    # Resources
    resources: dict[str, ResourceRecord] = {}  # URL -> ResourceRecord

    # Statistics
    total_resources: int = 0
    downloaded: int = 0
    ocr_completed: int = 0
    transcription_completed: int = 0
    failed: int = 0

    # Config snapshot
    config_snapshot: dict = {}
