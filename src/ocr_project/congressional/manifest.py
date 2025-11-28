"""Manifest management for Congressional proceedings collection."""

import json
from datetime import datetime
from pathlib import Path

from ocr_project.congressional.types import (
    CollectionManifest,
    ProceedingMetadata,
    ProcessingStatus,
    ResourceRecord,
    ResourceType,
)


class ManifestManager:
    """Manage collection manifest for tracking resources and processing status."""

    def __init__(self, output_dir: Path):
        """Initialize manifest manager.

        Args:
            output_dir: Directory for collection output and manifest
        """
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest: CollectionManifest | None = None

    def create_manifest(self, metadata: ProceedingMetadata) -> CollectionManifest:
        """Create new collection manifest.

        Args:
            metadata: Proceeding metadata

        Returns:
            New CollectionManifest instance
        """
        now = datetime.now()
        self.manifest = CollectionManifest(
            metadata=metadata,
            started_at=now,
            last_updated=now,
        )
        self.save_manifest()
        return self.manifest

    def load_manifest(self) -> CollectionManifest:
        """Load existing manifest from disk.

        Returns:
            Loaded CollectionManifest

        Raises:
            FileNotFoundError: If manifest file doesn't exist
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path) as f:
            data = json.load(f)

        self.manifest = CollectionManifest(**data)
        return self.manifest

    def save_manifest(self) -> None:
        """Save manifest to disk."""
        if self.manifest is None:
            raise ValueError("No manifest to save")

        # Update timestamp
        self.manifest.last_updated = datetime.now()

        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest.model_dump(mode="json"), f, indent=2, default=str)

    def add_resource(self, resource: ResourceRecord) -> None:
        """Add resource to manifest.

        Args:
            resource: ResourceRecord to add
        """
        if self.manifest is None:
            raise ValueError("No manifest loaded")

        url_str = str(resource.url)
        self.manifest.resources[url_str] = resource
        self.manifest.total_resources = len(self.manifest.resources)
        self.save_manifest()

    def update_resource_status(
        self,
        url: str,
        download_status: ProcessingStatus | None = None,
        ocr_status: ProcessingStatus | None = None,
        transcription_status: ProcessingStatus | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update resource processing status.

        Args:
            url: Resource URL
            download_status: New download status
            ocr_status: New OCR status
            transcription_status: New transcription status
            error_message: Error message if failed
        """
        if self.manifest is None:
            raise ValueError("No manifest loaded")

        if url not in self.manifest.resources:
            raise KeyError(f"Resource not found in manifest: {url}")

        resource = self.manifest.resources[url]

        if download_status is not None:
            resource.download_status = download_status
        if ocr_status is not None:
            resource.ocr_status = ocr_status
        if transcription_status is not None:
            resource.transcription_status = transcription_status
        if error_message is not None:
            resource.error_message = error_message

        # Update statistics
        self._update_statistics()
        self.save_manifest()

    def _update_statistics(self) -> None:
        """Update manifest statistics based on resource statuses."""
        if self.manifest is None:
            return

        downloaded = 0
        ocr_completed = 0
        transcription_completed = 0
        failed = 0

        for resource in self.manifest.resources.values():
            if resource.download_status == ProcessingStatus.DOWNLOADED:
                downloaded += 1
            elif resource.download_status == ProcessingStatus.FAILED:
                failed += 1

            if resource.ocr_status == ProcessingStatus.COMPLETED:
                ocr_completed += 1
            elif resource.ocr_status == ProcessingStatus.FAILED:
                failed += 1

            if resource.transcription_status == ProcessingStatus.COMPLETED:
                transcription_completed += 1
            elif resource.transcription_status == ProcessingStatus.FAILED:
                failed += 1

        self.manifest.downloaded = downloaded
        self.manifest.ocr_completed = ocr_completed
        self.manifest.transcription_completed = transcription_completed
        self.manifest.failed = failed

    def get_pending_resources(
        self, resource_type: ResourceType | None = None
    ) -> list[ResourceRecord]:
        """Get resources pending download.

        Args:
            resource_type: Optional filter by resource type

        Returns:
            List of pending ResourceRecords
        """
        if self.manifest is None:
            raise ValueError("No manifest loaded")

        pending = []
        for resource in self.manifest.resources.values():
            if resource.download_status == ProcessingStatus.PENDING:
                if resource_type is None or resource.resource_type == resource_type:
                    pending.append(resource)

        return pending

    def get_downloaded_resources(
        self, resource_type: ResourceType | None = None
    ) -> list[ResourceRecord]:
        """Get downloaded resources pending processing.

        Args:
            resource_type: Optional filter by resource type

        Returns:
            List of downloaded ResourceRecords
        """
        if self.manifest is None:
            raise ValueError("No manifest loaded")

        downloaded = []
        for resource in self.manifest.resources.values():
            if resource.download_status == ProcessingStatus.DOWNLOADED:
                if resource_type is None or resource.resource_type == resource_type:
                    downloaded.append(resource)

        return downloaded

    def mark_completed(self) -> None:
        """Mark collection as completed."""
        if self.manifest is None:
            raise ValueError("No manifest loaded")

        self.manifest.completed_at = datetime.now()
        self.save_manifest()

    def get_statistics(self) -> dict:
        """Get collection statistics.

        Returns:
            Dictionary with statistics
        """
        if self.manifest is None:
            raise ValueError("No manifest loaded")

        return {
            "total_resources": self.manifest.total_resources,
            "downloaded": self.manifest.downloaded,
            "ocr_completed": self.manifest.ocr_completed,
            "transcription_completed": self.manifest.transcription_completed,
            "failed": self.manifest.failed,
            "pending": self.manifest.total_resources
            - self.manifest.downloaded
            - self.manifest.failed,
        }
