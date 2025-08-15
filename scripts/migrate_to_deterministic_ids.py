#!/usr/bin/env python3
"""
Migration script to convert existing random UUIDs to deterministic IDs.

This script helps migrate existing Crawlerr installations to use the new
content-based deduplication system with deterministic IDs.

Usage:
    python scripts/migrate_to_deterministic_ids.py [options]

Options:
    --dry-run       Preview changes without applying them
    --batch-size    Number of chunks to process per batch (default: 100)
    --confirm       Skip interactive confirmation prompt
    --source-url    Migrate only specific source URL
    --help          Show this help message

The script will:
1. Connect to the Qdrant vector database
2. Scan for chunks with random UUID format (32+ hex chars)
3. Generate deterministic IDs based on source URL + chunk index
4. Update chunk IDs while preserving all other data
5. Report migration statistics
"""

import argparse
import asyncio
import hashlib
import logging
import re
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlparse, urlunparse

from crawler_mcp.config import settings
from crawler_mcp.core.vectors import VectorService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# UUID pattern for random IDs (32+ hex characters)
UUID_PATTERN = re.compile(r"^[0-9a-f]{32,}$", re.IGNORECASE)


class MigrationError(Exception):
    """Exception raised during migration process."""

    pass


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent deterministic ID generation.

    This matches the logic in RagService._normalize_url().
    """
    try:
        parsed = urlparse(url)

        # Normalize protocol to https
        scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme

        # Remove trailing slash from path
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # Sort query parameters for consistency
        if parsed.query:
            from urllib.parse import parse_qs, urlencode

            params = parse_qs(parsed.query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            query = urlencode(sorted_params, doseq=True)
        else:
            query = ""

        # Reconstruct normalized URL (without fragment)
        normalized = urlunparse(
            (
                scheme,
                parsed.netloc.lower(),  # Lowercase domain
                path,
                parsed.params,
                query,
                "",  # Remove fragment
            )
        )

        return normalized
    except Exception as e:
        logger.warning(f"Failed to normalize URL {url}: {e}")
        return url


def generate_deterministic_id(url: str, chunk_index: int) -> str:
    """
    Generate deterministic ID from URL and chunk index.

    This matches the logic in RagService._generate_deterministic_id().
    """
    normalized_url = normalize_url(url)
    id_string = f"{normalized_url}:{chunk_index}"
    return hashlib.sha256(id_string.encode()).hexdigest()[:16]


def is_random_uuid(chunk_id: str) -> bool:
    """Check if a chunk ID appears to be a random UUID."""
    return bool(UUID_PATTERN.match(chunk_id.replace("-", "")))


async def scan_chunks_for_migration(
    vector_service: VectorService, source_url: str | None = None
) -> list[dict[str, Any]]:
    """
    Scan vector database for chunks that need migration.

    Args:
        vector_service: Vector service instance
        source_url: Optional source URL filter

    Returns:
        List of chunks needing migration
    """
    logger.info("Scanning vector database for chunks needing migration...")

    try:
        # Get all chunks from the collection
        # This is a simplified approach - in production, you'd want pagination
        scroll_result = await vector_service.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10000,  # Adjust based on your dataset size
            with_payload=True,
            with_vectors=False,  # We don't need vectors for migration
        )

        chunks_to_migrate = []
        total_chunks = len(scroll_result[0]) if scroll_result[0] else 0

        logger.info(f"Found {total_chunks} total chunks in collection")

        for point in scroll_result[0]:
            chunk_id = str(point.id)
            payload = point.payload or {}

            # Skip if not a random UUID
            if not is_random_uuid(chunk_id):
                continue

            # Filter by source URL if specified
            chunk_source_url = payload.get("source_url", "")
            if source_url and chunk_source_url != source_url:
                continue

            chunks_to_migrate.append(
                {
                    "id": chunk_id,
                    "payload": payload,
                    "source_url": chunk_source_url,
                    "chunk_index": payload.get("chunk_index", 0),
                }
            )

        logger.info(f"Found {len(chunks_to_migrate)} chunks needing migration")
        return chunks_to_migrate

    except Exception as e:
        logger.error(f"Failed to scan chunks: {e}")
        raise MigrationError(f"Chunk scanning failed: {e}") from e


async def migrate_chunk_batch(
    vector_service: VectorService, chunks: list[dict[str, Any]], dry_run: bool = False
) -> dict[str, int]:
    """
    Migrate a batch of chunks to deterministic IDs.

    Args:
        vector_service: Vector service instance
        chunks: List of chunks to migrate
        dry_run: If True, only preview changes

    Returns:
        Migration statistics
    """
    stats = {"processed": 0, "migrated": 0, "skipped": 0, "errors": 0}

    migrations = []  # List of (old_id, new_id, payload) tuples

    for chunk in chunks:
        stats["processed"] += 1

        try:
            old_id = chunk["id"]
            source_url = chunk["source_url"]
            chunk_index = chunk["chunk_index"]

            if not source_url:
                logger.warning(f"Chunk {old_id} has no source_url, skipping")
                stats["skipped"] += 1
                continue

            # Generate new deterministic ID
            new_id = generate_deterministic_id(source_url, chunk_index)

            if old_id == new_id:
                logger.debug(f"Chunk {old_id} already has correct ID, skipping")
                stats["skipped"] += 1
                continue

            # Check if new ID already exists
            try:
                existing = await vector_service.client.retrieve(
                    collection_name=settings.qdrant_collection, ids=[new_id]
                )
                if existing:
                    logger.warning(f"New ID {new_id} already exists, skipping {old_id}")
                    stats["skipped"] += 1
                    continue
            except Exception:
                # New ID doesn't exist, which is what we want
                pass

            migrations.append((old_id, new_id, chunk["payload"]))
            stats["migrated"] += 1

            logger.debug(f"Will migrate {old_id} -> {new_id}")

        except Exception as e:
            logger.error(f"Error processing chunk {chunk.get('id', 'unknown')}: {e}")
            stats["errors"] += 1

    if dry_run:
        logger.info(f"DRY RUN: Would migrate {len(migrations)} chunks")
        return stats

    # Perform actual migrations
    if not migrations:
        logger.info("No chunks to migrate in this batch")
        return stats

    try:
        # Get the original points with vectors
        old_ids = [migration[0] for migration in migrations]
        original_points = await vector_service.client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=old_ids,
            with_vectors=True,
            with_payload=True,
        )

        # Create new points with deterministic IDs
        points_to_upsert = []
        points_to_delete = []

        for i, (old_id, new_id, payload) in enumerate(migrations):
            if i < len(original_points):
                original_point = original_points[i]

                # Create new point with same vector and updated payload
                from qdrant_client.models import PointStruct

                new_point = PointStruct(
                    id=new_id,
                    vector=original_point.vector,
                    payload={
                        **payload,
                        "migrated_from": old_id,
                        "migration_timestamp": datetime.utcnow().isoformat(),
                    },
                )
                points_to_upsert.append(new_point)
                points_to_delete.append(old_id)

        # Upsert new points
        if points_to_upsert:
            await vector_service.client.upsert(
                collection_name=settings.qdrant_collection, points=points_to_upsert
            )
            logger.info(f"Upserted {len(points_to_upsert)} points with new IDs")

        # Delete old points
        if points_to_delete:
            await vector_service.client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=points_to_delete,
            )
            logger.info(f"Deleted {len(points_to_delete)} points with old IDs")

    except Exception as e:
        logger.error(f"Failed to migrate batch: {e}")
        stats["errors"] += len(migrations)
        stats["migrated"] = 0
        raise MigrationError(f"Batch migration failed: {e}") from e

    return stats


async def run_migration(
    source_url: str | None = None,
    batch_size: int = 100,
    dry_run: bool = False,
    confirm: bool = False,
) -> None:
    """
    Run the full migration process.

    Args:
        source_url: Optional source URL filter
        batch_size: Number of chunks to process per batch
        dry_run: If True, only preview changes
        confirm: If True, skip confirmation prompt
    """
    logger.info("Starting migration from random UUIDs to deterministic IDs")
    logger.info(f"Configuration: batch_size={batch_size}, dry_run={dry_run}")

    if source_url:
        logger.info(f"Filtering to source URL: {source_url}")

    # Initialize vector service
    vector_service = VectorService()

    try:
        async with vector_service:
            # Check connection
            if not await vector_service.health_check():
                raise MigrationError("Vector service health check failed")

            # Scan for chunks needing migration
            chunks_to_migrate = await scan_chunks_for_migration(
                vector_service, source_url
            )

            if not chunks_to_migrate:
                logger.info(
                    "No chunks need migration - all IDs are already deterministic!"
                )
                return

            # Show preview
            logger.info(f"Found {len(chunks_to_migrate)} chunks to migrate:")
            for i, chunk in enumerate(chunks_to_migrate[:5]):  # Show first 5
                old_id = chunk["id"]
                new_id = generate_deterministic_id(
                    chunk["source_url"], chunk["chunk_index"]
                )
                logger.info(f"  {old_id} -> {new_id}")

            if len(chunks_to_migrate) > 5:
                logger.info(f"  ... and {len(chunks_to_migrate) - 5} more")

            # Confirmation prompt
            if not dry_run and not confirm:
                response = input(
                    f"\nProceed with migration of {len(chunks_to_migrate)} chunks? (y/N): "
                )
                if response.lower() not in ("y", "yes"):
                    logger.info("Migration cancelled by user")
                    return

            # Process in batches
            total_stats = {"processed": 0, "migrated": 0, "skipped": 0, "errors": 0}

            for i in range(0, len(chunks_to_migrate), batch_size):
                batch = chunks_to_migrate[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(chunks_to_migrate) + batch_size - 1) // batch_size

                logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)..."
                )

                try:
                    batch_stats = await migrate_chunk_batch(
                        vector_service, batch, dry_run
                    )

                    # Aggregate statistics
                    for key in total_stats:
                        total_stats[key] += batch_stats[key]

                    logger.info(
                        f"Batch {batch_num} completed: "
                        f"{batch_stats['migrated']} migrated, "
                        f"{batch_stats['skipped']} skipped, "
                        f"{batch_stats['errors']} errors"
                    )

                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    total_stats["errors"] += len(batch)
                    continue

            # Final report
            logger.info("Migration completed!")
            logger.info("Final statistics:")
            logger.info(f"  Processed: {total_stats['processed']}")
            logger.info(f"  Migrated: {total_stats['migrated']}")
            logger.info(f"  Skipped: {total_stats['skipped']}")
            logger.info(f"  Errors: {total_stats['errors']}")

            if total_stats["errors"] > 0:
                logger.warning(
                    f"Migration completed with {total_stats['errors']} errors"
                )
                sys.exit(1)

            if dry_run:
                logger.info("This was a DRY RUN - no changes were made")
            else:
                logger.info("Migration successfully completed!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate random UUIDs to deterministic IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying them"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks to process per batch (default: 100)",
    )

    parser.add_argument(
        "--confirm", action="store_true", help="Skip interactive confirmation prompt"
    )

    parser.add_argument(
        "--source-url", type=str, help="Migrate only chunks from specific source URL"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run migration
    asyncio.run(
        run_migration(
            source_url=args.source_url,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            confirm=args.confirm,
        )
    )


if __name__ == "__main__":
    main()
