"""Tests for manifest management."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import HttpUrl

from ocr_project.congressional.manifest import ManifestManager
from ocr_project.congressional.types import (
    ProceedingMetadata,
    ProcessingStatus,
    ResourceRecord,
    ResourceType,
    SourceType,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metadata():
    """Create test proceeding metadata."""
    return ProceedingMetadata(
        url=HttpUrl("https://www.congress.gov/event/119th-congress/house-event/118671"),
        proceeding_id="house-event-118671",
        title="Test Hearing",
        chamber="House",
    )


def test_create_manifest(temp_dir, metadata):
    """Test creating new manifest."""
    manager = ManifestManager(temp_dir)
    manifest = manager.create_manifest(metadata)

    assert manifest.metadata.title == "Test Hearing"
    assert manifest.total_resources == 0
    assert (temp_dir / "manifest.json").exists()


def test_load_manifest(temp_dir, metadata):
    """Test loading existing manifest."""
    manager = ManifestManager(temp_dir)
    manager.create_manifest(metadata)

    # Create new manager and load
    manager2 = ManifestManager(temp_dir)
    loaded_manifest = manager2.load_manifest()

    assert loaded_manifest.metadata.title == "Test Hearing"


def test_add_resource(temp_dir, metadata):
    """Test adding resource to manifest."""
    manager = ManifestManager(temp_dir)
    manager.create_manifest(metadata)

    resource = ResourceRecord(
        url=HttpUrl("https://example.com/doc.pdf"),
        resource_type=ResourceType.PDF,
        source_type=SourceType.CONGRESS_GOV,
        filename="doc.pdf",
        discovered_at=datetime.now(),
        priority=5,
    )

    manager.add_resource(resource)

    assert manager.manifest.total_resources == 1
    assert "https://example.com/doc.pdf" in manager.manifest.resources


def test_update_resource_status(temp_dir, metadata):
    """Test updating resource status."""
    manager = ManifestManager(temp_dir)
    manager.create_manifest(metadata)

    resource = ResourceRecord(
        url=HttpUrl("https://example.com/doc.pdf"),
        resource_type=ResourceType.PDF,
        source_type=SourceType.CONGRESS_GOV,
        filename="doc.pdf",
        discovered_at=datetime.now(),
        priority=5,
    )

    manager.add_resource(resource)
    manager.update_resource_status(
        "https://example.com/doc.pdf", download_status=ProcessingStatus.DOWNLOADED
    )

    updated_resource = manager.manifest.resources["https://example.com/doc.pdf"]
    assert updated_resource.download_status == ProcessingStatus.DOWNLOADED
    assert manager.manifest.downloaded == 1


def test_get_pending_resources(temp_dir, metadata):
    """Test getting pending resources."""
    manager = ManifestManager(temp_dir)
    manager.create_manifest(metadata)

    # Add pending resource
    resource1 = ResourceRecord(
        url=HttpUrl("https://example.com/doc1.pdf"),
        resource_type=ResourceType.PDF,
        source_type=SourceType.CONGRESS_GOV,
        filename="doc1.pdf",
        discovered_at=datetime.now(),
        priority=5,
    )
    manager.add_resource(resource1)

    # Add downloaded resource
    resource2 = ResourceRecord(
        url=HttpUrl("https://example.com/doc2.pdf"),
        resource_type=ResourceType.PDF,
        source_type=SourceType.CONGRESS_GOV,
        filename="doc2.pdf",
        discovered_at=datetime.now(),
        priority=5,
        download_status=ProcessingStatus.DOWNLOADED,
    )
    manager.add_resource(resource2)

    pending = manager.get_pending_resources()
    assert len(pending) == 1
    assert pending[0].filename == "doc1.pdf"


def test_get_statistics(temp_dir, metadata):
    """Test getting collection statistics."""
    manager = ManifestManager(temp_dir)
    manager.create_manifest(metadata)

    resource = ResourceRecord(
        url=HttpUrl("https://example.com/doc.pdf"),
        resource_type=ResourceType.PDF,
        source_type=SourceType.CONGRESS_GOV,
        filename="doc.pdf",
        discovered_at=datetime.now(),
        priority=5,
    )
    manager.add_resource(resource)

    stats = manager.get_statistics()
    assert stats["total_resources"] == 1
    assert stats["pending"] == 1
    assert stats["downloaded"] == 0
