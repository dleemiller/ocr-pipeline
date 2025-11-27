"""Debug script to check dataset structure."""

from datasets import load_dataset

# Load a small sample from the first subset
print("Loading epstein_estate_2025_09...")
dataset = load_dataset(
    "public-records-research/epstractor-raw",
    name="epstein_estate_2025_09",
    split="train",
    streaming=True,
)

# Check first few rows
print("\nFirst 10 rows:")
for i, row in enumerate(dataset.take(10)):
    print(f"\nRow {i}:")
    print(f"  Keys: {row.keys()}")
    if "file_type" in row:
        print(f"  file_type: {row['file_type']}")
    if "path" in row:
        print(f"  path: {row['path']}")
    if "content_available" in row:
        print(f"  content_available: {row['content_available']}")
    if "extension" in row:
        print(f"  extension: {row['extension']}")
    if "content" in row:
        content_size = len(row["content"]) if row["content"] else 0
        print(f"  content size: {content_size} bytes")
