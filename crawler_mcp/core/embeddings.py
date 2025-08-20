"""
Service for generating embeddings using HF Text Embeddings Inference (TEI).
"""

import asyncio
import logging
import time
from typing import Any

import httpx
from fastmcp.exceptions import ToolError

from ..config import settings
from ..models.rag import EmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using HF TEI.
    """

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.tei_timeout),
            limits=httpx.Limits(
                max_keepalive_connections=settings.tei_max_concurrent_requests,
                max_connections=settings.tei_max_concurrent_requests * 2,
            ),
        )
        self.base_url = settings.tei_url.rstrip("/")
        self.model_name = settings.tei_model

    async def __aenter__(self) -> "EmbeddingService":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: type, exc_val: Exception, exc_tb: object
    ) -> None:
        """Async context manager exit."""
        await self.client.aclose()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def _ensure_client_open(self) -> None:
        """Ensure the HTTP client is open and recreate if closed."""
        if self.client.is_closed:
            logger.debug("HTTP client was closed, recreating...")
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.tei_timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=settings.tei_max_concurrent_requests,
                    max_connections=settings.tei_max_concurrent_requests * 2,
                ),
            )

    async def health_check(self) -> bool:
        """
        Check if TEI service is healthy and responsive.
        """
        try:
            await self._ensure_client_open()
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"TEI health check failed: {e}")
            return False

    async def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.
        """
        try:
            await self._ensure_client_open()
            response = await self.client.get(f"{self.base_url}/info")
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            else:
                logger.warning(f"Failed to get model info: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    async def generate_embedding(
        self, text: str, normalize: bool | None = None, truncate: bool = True
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed
            normalize: Whether to normalize the embedding (defaults to config setting)
            truncate: Whether to truncate text if it exceeds max length

        Returns:
            EmbeddingResult with the generated embedding
        """
        if not text.strip():
            raise ToolError("Cannot generate embedding for empty text")

        # Truncate text if necessary
        if truncate and len(text) > settings.embedding_max_length:
            text = text[: settings.embedding_max_length]
            logger.warning(
                f"Text truncated to {settings.embedding_max_length} characters"
            )

        # Use config default if not specified
        if normalize is None:
            normalize = settings.embedding_normalize

        start_time = time.time()

        try:
            # Ensure client is open before making request
            await self._ensure_client_open()

            # Prepare request payload
            payload = {"inputs": text, "normalize": normalize, "truncate": truncate}

            # Make request to TEI service
            response = await self.client.post(
                f"{self.base_url}/embed",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                error_msg = (
                    f"TEI service returned {response.status_code}: {response.text}"
                )
                logger.error(error_msg)
                raise ToolError(f"Embedding generation failed: {error_msg}")

            # Parse response
            result = response.json()
            embedding = result[0] if isinstance(result, list) else result

            processing_time = time.time() - start_time

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model_name,
                dimensions=len(embedding),
                processing_time=processing_time,
            )

        except httpx.TimeoutException as e:
            raise ToolError(
                f"Embedding request timed out after {settings.tei_timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise ToolError(f"Failed to connect to TEI service: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise ToolError(f"Embedding generation failed: {e!s}") from e

    async def generate_embeddings_true_batch(
        self,
        texts: list[str],
        normalize: bool | None = None,
        truncate: bool = True,
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in a single API call (true batch).

        This is much faster than individual calls for large batches.
        """
        if not texts:
            return []

        # Filter and truncate texts
        valid_texts = []
        for text in texts:
            if text.strip():
                if truncate and len(text) > settings.embedding_max_length:
                    text = text[: settings.embedding_max_length]
                valid_texts.append(text)

        if not valid_texts:
            return []

        if normalize is None:
            normalize = settings.embedding_normalize

        start_time = time.time()

        try:
            await self._ensure_client_open()

            # Send all texts in single request
            payload = {
                "inputs": valid_texts,
                "normalize": normalize,
                "truncate": truncate,
            }

            response = await self.client.post(
                f"{self.base_url}/embed",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                error_msg = (
                    f"TEI service returned {response.status_code}: {response.text}"
                )
                logger.error(error_msg)
                raise ToolError(f"Batch embedding failed: {error_msg}")

            result = response.json()
            processing_time = time.time() - start_time

            # Convert to EmbeddingResult objects
            results = []
            for _i, (text, embedding) in enumerate(
                zip(valid_texts, result, strict=False)
            ):
                results.append(
                    EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model=self.model_name,
                        dimensions=len(embedding),
                        processing_time=processing_time
                        / len(valid_texts),  # Average time per embedding
                    )
                )

            logger.info(
                f"Generated {len(results)} embeddings in {processing_time:.2f}s (true batch) - {len(results) / processing_time:.1f} embeddings/sec"
            )
            return results

        except httpx.TimeoutException as e:
            raise ToolError(
                f"Batch embedding request timed out after {settings.tei_timeout}s"
            ) from e
        except httpx.RequestError as e:
            raise ToolError(f"Failed to connect to TEI service: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in batch embedding: {e}")
            raise ToolError(f"Batch embedding failed: {e!s}") from e

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        normalize: bool | None = None,
        truncate: bool = True,
        batch_size: int | None = None,
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            truncate: Whether to truncate texts if they exceed max length
            batch_size: Size of batches (defaults to config setting)

        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []

        if batch_size is None:
            batch_size = settings.tei_batch_size

        # Filter out empty texts
        valid_texts = [
            (i, text.strip()) for i, text in enumerate(texts) if text.strip()
        ]
        if not valid_texts:
            raise ToolError("No valid texts to embed")

        results = []
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(valid_texts)} texts in {total_batches} batches")

        # Start timing the entire batch process
        batch_start_time = time.time()

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(valid_texts))
            batch = valid_texts[start_idx:end_idx]

            logger.debug(
                f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} texts)"
            )

            # Process batch concurrently
            batch_tasks = [
                self.generate_embedding(text, normalize=normalize, truncate=truncate)
                for _, text in batch
            ]

            try:
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Handle results and exceptions
                for (original_idx, _), result in zip(
                    batch, batch_results, strict=False
                ):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to embed text {original_idx}: {result}")
                        # Create error result
                        error_result = EmbeddingResult(
                            text=texts[original_idx][:100] + "..."
                            if len(texts[original_idx]) > 100
                            else texts[original_idx],
                            embedding=[0.0]
                            * settings.embedding_dimension,  # Zero embedding for failed texts
                            model=self.model_name,
                            dimensions=settings.embedding_dimension,
                            processing_time=0.0,
                        )
                        results.append((original_idx, error_result))
                    else:
                        # Type assertion: result is EmbeddingResult if not Exception
                        assert isinstance(result, EmbeddingResult)
                        results.append((original_idx, result))

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise ToolError(f"Batch embedding generation failed: {e!s}") from e

        # Log total batch processing time
        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time
        logger.info(
            f"Completed embedding generation for {len(valid_texts)} texts in {total_batch_time:.2f}s - {len(valid_texts) / total_batch_time:.1f} embeddings/sec"
        )

        # Sort results by original index and return embedding results
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    async def compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(embedding1) != len(embedding2):
            raise ToolError("Embeddings must have the same dimensions")

        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))

        # Compute magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        # Ensure result is in [0, 1] range (convert from [-1, 1])
        normalized_similarity: float = max(0.0, min(1.0, (similarity + 1) / 2))
        return normalized_similarity
