"""Tests for Congressional types."""

from datetime import datetime

from pydantic import HttpUrl

from ocr_project.congressional.types import (
    CollectionManifest,
    ProceedingMetadata,
    ProcessingStatus,
    ResourceRecord,
    ResourceType,
    SourceType,
)


def test_resource_record_creation():
    """Test ResourceRecord creation."""
    record = ResourceRecord(
        url=HttpUrl("https://example.com/doc.pdf"),
        resource_type=ResourceType.PDF,
        source_type=SourceType.CONGRESS_GOV,
        filename="doc.pdf",
        discovered_at=datetime.now(),
        priority=5,
    )

    assert record.resource_type == ResourceType.PDF
    assert record.download_status == ProcessingStatus.PENDING
    assert record.retry_count == 0


def test_proceeding_metadata_creation():
    """Test ProceedingMetadata creation."""
    metadata = ProceedingMetadata(
        url=HttpUrl("https://www.congress.gov/event/119th-congress/house-event/118671"),
        proceeding_id="house-event-118671",
        title="DOJ Oversight Hearing",
        chamber="House",
        congress_number=119,
    )

    assert metadata.chamber == "House"
    assert metadata.congress_number == 119
    assert len(metadata.witnesses) == 0


def test_collection_manifest_creation():
    """Test CollectionManifest creation."""
    metadata = ProceedingMetadata(
        url=HttpUrl("https://www.congress.gov/event/119th-congress/house-event/118671"),
        proceeding_id="house-event-118671",
        title="DOJ Oversight Hearing",
    )

    now = datetime.now()
    manifest = CollectionManifest(metadata=metadata, started_at=now, last_updated=now)

    assert manifest.total_resources == 0
    assert manifest.downloaded == 0
    assert len(manifest.resources) == 0
