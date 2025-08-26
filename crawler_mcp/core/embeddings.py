"""
Service for generating embeddings using HF Text Embeddings Inference (TEI).
"""

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

import httpx
from fastmcp.exceptions import ToolError

from ..config import settings
from ..models.rag import EmbeddingResult
from .resilience import exponential_backoff

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

    def _chunked(self, iterable: Sequence["T"], size: int) -> Iterator[list["T"]]:
        """Yield successive chunks from the iterable."""
        if size <= 0:
            raise ValueError("size must be > 0")
        for i in range(0, len(iterable), size):
            # always return list chunks for a consistent downstream type
            yield list(iterable[i : i + size])

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

    @exponential_backoff(exceptions=(httpx.TimeoutException, httpx.RequestError))
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

        start_time = time.perf_counter()

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

            processing_time = max(time.perf_counter() - start_time, 1e-6)

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

        start_time = time.perf_counter()

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
            if not isinstance(result, list):
                raise ToolError(f"Unexpected TEI response shape: {type(result)}")
            if len(result) != len(valid_texts):
                raise ToolError(
                    f"TEI returned {len(result)} embeddings for {len(valid_texts)} inputs"
                )
            processing_time = max(time.perf_counter() - start_time, 1e-6)

            # Convert to EmbeddingResult objects
            results = []
            for _i, (text, embedding) in enumerate(
                zip(valid_texts, result, strict=True)
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
        Generate embeddings for multiple texts using TRUE BATCH API.
        This method now delegates to generate_embeddings_true_batch for optimal performance.

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

        # For optimal performance, use true batch API
        # Split into chunks if needed based on batch_size
        results = []

        for batch in self._chunked(texts, batch_size):
            batch_results = await self.generate_embeddings_true_batch(
                batch, normalize=normalize, truncate=truncate
            )
            results.extend(batch_results)

        return results

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
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=True))

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
