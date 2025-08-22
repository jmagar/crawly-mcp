"""
Text chunking strategies and token counting for RAG operations.

This module provides various chunking strategies for breaking down text content
into optimal chunks for embedding generation and vector search.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ...config import settings

logger = logging.getLogger(__name__)

# Approximate word-to-token ratio for different tokenizers
WORD_TO_TOKEN_RATIO = 1.3  # General estimate for English text
QWEN3_WORD_TO_TOKEN_RATIO = (
    1.4  # More accurate for Qwen3 tokenizer based on empirical testing
)


def find_paragraph_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find paragraph break boundary."""
    paragraph_breaks = [
        i for i, char in enumerate(search_text) if search_text[i : i + 2] == "\n\n"
    ]
    suitable_breaks = [
        b for b in paragraph_breaks if ideal_end - 100 <= b <= ideal_end + 100
    ]
    return max(suitable_breaks) if suitable_breaks else None


def find_sentence_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find sentence ending boundary."""
    sentence_patterns = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
    sentence_breaks = []
    for pattern in sentence_patterns:
        sentence_breaks.extend(
            [
                i + len(pattern)
                for i in range(len(search_text) - len(pattern))
                if search_text[i : i + len(pattern)] == pattern
            ]
        )
    suitable_breaks = [
        b for b in sentence_breaks if ideal_end - 50 <= b <= ideal_end + 50
    ]
    return max(suitable_breaks) if suitable_breaks else None


def find_line_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find line break boundary."""
    line_breaks = [i + 1 for i, char in enumerate(search_text) if char == "\n"]
    suitable_breaks = [b for b in line_breaks if ideal_end - 30 <= b <= ideal_end + 30]
    return max(suitable_breaks) if suitable_breaks else None


def find_word_boundary(search_text: str, ideal_end: int) -> int | None:
    """Find word boundary."""
    word_breaks = [i + 1 for i, char in enumerate(search_text) if char == " "]
    suitable_breaks = [b for b in word_breaks if ideal_end - 20 <= b <= ideal_end + 20]
    return max(suitable_breaks) if suitable_breaks else None


class TokenCounter:
    """Accurate token counting with multiple tokenizer support."""

    def __init__(self):
        self.tokenizer = None
        self.tokenizer_type = "character"  # Default tokenizer type
        self.word_to_token_ratio = settings.word_to_token_ratio

        # Initialize tokenizer
        try:
            import tiktoken

            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.tokenizer_type = "tiktoken"
        except ImportError:
            logger.warning("tiktoken not available, using character-based chunking")
            self.tokenizer_type = "character"

    def count_tokens(self, text: str) -> int:
        """
        Calculate accurate token count using actual tokenizer when available.

        Args:
            text: Text to count tokens for

        Returns:
            Accurate token count
        """
        if self.tokenizer:
            try:
                # Use actual tokenizer for precise count
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"Tokenizer failed, falling back to estimation: {e}")

        # Fallback to word-based estimation with configurable ratio
        word_count = len(text.split())
        return int(word_count * self.word_to_token_ratio)

    def estimate_tokens_from_words(self, text: str) -> int:
        """Estimate token count from word count."""
        word_count = len(text.split())
        return int(word_count * self.word_to_token_ratio)

    def calculate_chunk_token_distribution(self, chunks: list[str]) -> dict[str, float]:
        """Calculate token distribution statistics for chunks."""
        if not chunks:
            return {}

        token_counts = [self.count_tokens(chunk) for chunk in chunks]
        return {
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "total_tokens": sum(token_counts),
        }


class ChunkingStrategy(ABC):
    """Base class for text chunking strategies."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_counter = TokenCounter()

    @abstractmethod
    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Chunk text using the specific strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include in chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        pass

    def validate_chunk_size(self, chunk: str) -> bool:
        """Validate if chunk size is within acceptable limits."""
        return len(chunk) > 50 and len(chunk) <= self.chunk_size * 2

    def assess_chunk_quality(
        self, chunk: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Assess chunk quality across multiple dimensions."""
        return {
            "completeness": self._assess_completeness(chunk),
            "coherence": self._assess_coherence(chunk),
            "information_density": self._assess_information_density(chunk),
            "context_preservation": self._assess_context_preservation(
                chunk, metadata or {}
            ),
        }

    def _assess_completeness(self, chunk: str) -> float:
        """Assess how complete the chunk is (sentences end properly, etc.)."""
        if not chunk.strip():
            return 0.0

        # Check if chunk ends with sentence punctuation
        last_char = chunk.strip()[-1]
        if last_char in ".!?":
            return 1.0
        elif last_char in ",;:":
            return 0.7
        else:
            return 0.5

    def _assess_coherence(self, chunk: str) -> float:
        """Assess how coherent the chunk is."""
        # Simple heuristic: longer chunks with proper sentence structure are more coherent
        sentences = chunk.split(".")
        if len(sentences) < 2:
            return 0.5

        avg_sentence_length = sum(len(s.strip().split()) for s in sentences) / len(
            sentences
        )
        if 5 <= avg_sentence_length <= 25:  # Reasonable sentence length
            return 1.0
        else:
            return 0.7

    def _assess_information_density(self, chunk: str) -> float:
        """Assess information density of the chunk."""
        words = chunk.split()
        if not words:
            return 0.0

        # Heuristic: ratio of unique words to total words
        unique_words = {word.lower() for word in words}
        density = len(unique_words) / len(words)
        return min(1.0, density * 2)  # Scale to 0-1 range

    def _assess_context_preservation(
        self, chunk: str, metadata: dict[str, Any]
    ) -> float:
        """Assess how well context is preserved in the chunk."""
        # If we have overlap, context is better preserved
        if self.overlap > 0:
            return 1.0
        else:
            return 0.8


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with overlap and smart boundary detection."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Optimized character-based chunking with semantic boundary detection.
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0

        while start < text_length:
            end = start + self.chunk_size

            # Find a good break point (end of sentence or paragraph)
            if end < text_length:
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_ends = []

                for i in range(search_start, min(end + 50, text_length)):
                    if (
                        text[i] in ".!?"
                        and i + 1 < text_length
                        and text[i + 1] in " \n\t"
                    ):
                        sentence_ends.append(i + 1)

                # Use the last sentence end if found, otherwise stick to character limit
                if sentence_ends:
                    end = sentence_ends[-1]

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "token_count_estimate": self.token_counter.count_tokens(chunk_text),
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap, end)

            # Prevent infinite loop
            if start <= end - self.chunk_size:
                start = end

        return chunks


class TokenBasedChunker(ChunkingStrategy):
    """Token-based chunking using actual tokenizer for optimal semantic boundaries."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Token-based chunking using tokenizer for optimal semantic boundaries.
        """
        chunks = []

        if self.token_counter.tokenizer:
            # Use actual tokenizer
            tokens = self.token_counter.tokenizer.encode(text)
            total_tokens = len(tokens)
            start_token = 0
            chunk_index = 0

            while start_token < total_tokens:
                end_token = min(start_token + self.chunk_size, total_tokens)

                # Extract token chunk
                chunk_tokens = tokens[start_token:end_token]

                # Ensure chunk_tokens is a flat list of integers
                if (
                    isinstance(chunk_tokens, list)
                    and len(chunk_tokens) > 0
                    and isinstance(chunk_tokens[0], list)
                ):
                    # Flatten nested list
                    chunk_tokens = [
                        token for sublist in chunk_tokens for token in sublist
                    ]

                chunk_text = self.token_counter.tokenizer.decode(chunk_tokens)

                if chunk_text.strip():
                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_pos": start_token,
                        "end_pos": end_token,
                        "start_token": start_token,
                        "end_token": end_token,
                        "token_count": len(chunk_tokens),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Move start position with overlap
                start_token = max(
                    start_token + self.chunk_size - self.overlap, end_token
                )

                # Prevent infinite loop
                if start_token <= end_token - self.chunk_size:
                    start_token = end_token
        else:
            # Fallback to approximate token-based chunking using word estimation
            words = text.split()
            total_words = len(words)
            # Use configurable word-to-token ratio for accuracy
            approx_tokens_per_word = self.token_counter.word_to_token_ratio
            chunk_size_words = int(self.chunk_size / approx_tokens_per_word)
            overlap_words = int(self.overlap / approx_tokens_per_word)

            start_word = 0
            chunk_index = 0

            while start_word < total_words:
                end_word = min(start_word + chunk_size_words, total_words)
                chunk_words = words[start_word:end_word]
                chunk_text = " ".join(chunk_words)

                if chunk_text.strip():
                    estimated_tokens = int(len(chunk_words) * approx_tokens_per_word)
                    # Calculate character positions for consistency
                    text_start_pos = len(" ".join(words[:start_word])) + (
                        1 if start_word > 0 else 0
                    )
                    text_end_pos = text_start_pos + len(chunk_text)

                    chunk = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_pos": text_start_pos,
                        "end_pos": text_end_pos,
                        "start_word": start_word,
                        "end_word": end_word,
                        "token_count_estimate": estimated_tokens,
                        "word_count": len(chunk_words),
                        "char_count": len(chunk_text),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Move start position with overlap
                start_word = max(
                    start_word + chunk_size_words - overlap_words, end_word
                )

                # Prevent infinite loop
                if start_word <= end_word - chunk_size_words:
                    start_word = end_word

        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic boundary-aware chunking."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk text based on semantic boundaries like paragraphs and sentences."""
        chunks = []
        chunk_index = 0

        # First try to split on paragraphs
        paragraphs = self.split_on_paragraphs(text)

        current_chunk = ""
        start_pos = 0

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk.strip():
                    chunk = {
                        "text": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "start_pos": start_pos,
                        "end_pos": start_pos + len(current_chunk),
                        "word_count": len(current_chunk.split()),
                        "char_count": len(current_chunk),
                        "token_count_estimate": self.token_counter.count_tokens(
                            current_chunk
                        ),
                        **(metadata or {}),
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_text = current_chunk[-self.overlap :]
                    current_chunk = overlap_text + paragraph
                    start_pos = (
                        start_pos
                        + len(current_chunk)
                        - len(overlap_text)
                        - len(paragraph)
                    )
                else:
                    current_chunk = paragraph
                    start_pos = start_pos + len(current_chunk)
            else:
                current_chunk += paragraph

        # Add final chunk
        if current_chunk.strip():
            chunk = {
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "start_pos": start_pos,
                "end_pos": start_pos + len(current_chunk),
                "word_count": len(current_chunk.split()),
                "char_count": len(current_chunk),
                "token_count_estimate": self.token_counter.count_tokens(current_chunk),
                **(metadata or {}),
            }
            chunks.append(chunk)

        return chunks

    def find_semantic_boundaries(self, text: str) -> list[int]:
        """Find semantic boundaries in text."""
        boundaries = []

        # Find paragraph boundaries
        for i, _char in enumerate(text):
            if i < len(text) - 1 and text[i : i + 2] == "\n\n":
                boundaries.append(i)

        # Find sentence boundaries if no paragraphs
        if not boundaries:
            sentence_patterns = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
            for pattern in sentence_patterns:
                for i in range(len(text) - len(pattern)):
                    if text[i : i + len(pattern)] == pattern:
                        boundaries.append(i + len(pattern))

        return sorted(set(boundaries))

    def split_on_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"[.!?]+\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def split_on_paragraphs(self, text: str) -> list[str]:
        """Split text on paragraph boundaries."""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


class AdaptiveChunker(ChunkingStrategy):
    """Adaptive chunking based on content type and structure."""

    def chunk_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk text using adaptive strategy based on content type."""
        content_type = self.detect_content_structure(text)

        if content_type == "code":
            return self.chunk_code_content(text, metadata)
        elif content_type == "markdown":
            return self.chunk_markdown_content(text, metadata)
        else:
            return self.chunk_plain_text(text, metadata)

    def detect_content_structure(self, text: str) -> str:
        """Detect the structure/type of content."""
        # Simple heuristics for content type detection
        lines = text.split("\n")

        # Check for code indicators
        code_indicators = 0
        markdown_indicators = 0

        for line in lines[:20]:  # Check first 20 lines
            if any(
                keyword in line
                for keyword in [
                    "def ",
                    "class ",
                    "function ",
                    "import ",
                    "const ",
                    "var ",
                ]
            ):
                code_indicators += 1
            if any(
                marker in line for marker in ["# ", "## ", "### ", "* ", "- ", "```"]
            ):
                markdown_indicators += 1

        if code_indicators > markdown_indicators:
            return "code"
        elif markdown_indicators > 0:
            return "markdown"
        else:
            return "plain_text"

    def chunk_code_content(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk code content preserving function/class boundaries."""
        # For code, try to keep functions/classes together
        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_line = 0

        for i, line in enumerate(lines):
            # Check if we're starting a new function/class and current chunk is getting large
            if (
                any(keyword in line for keyword in ["def ", "class ", "function "])
                and len(current_chunk) > self.chunk_size * 0.8
                and current_chunk.strip()
            ):
                chunk = {
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "start_pos": start_line,
                    "end_pos": i,
                    "word_count": len(current_chunk.split()),
                    "char_count": len(current_chunk),
                    "token_count_estimate": self.token_counter.count_tokens(
                        current_chunk
                    ),
                    "content_type": "code",
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = ""
                start_line = i

            current_chunk += line + "\n"

            # Force split if chunk gets too large
            if len(current_chunk) > self.chunk_size * 1.5 and current_chunk.strip():
                chunk = {
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "start_pos": start_line,
                    "end_pos": i + 1,
                    "word_count": len(current_chunk.split()),
                    "char_count": len(current_chunk),
                    "token_count_estimate": self.token_counter.count_tokens(
                        current_chunk
                    ),
                    "content_type": "code",
                    **(metadata or {}),
                }
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = ""
                start_line = i + 1

        # Add final chunk
        if current_chunk.strip():
            chunk = {
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "start_pos": start_line,
                "end_pos": len(lines),
                "word_count": len(current_chunk.split()),
                "char_count": len(current_chunk),
                "token_count_estimate": self.token_counter.count_tokens(current_chunk),
                "content_type": "code",
                **(metadata or {}),
            }
            chunks.append(chunk)

        return chunks

    def chunk_markdown_content(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk markdown content preserving section boundaries."""
        # Use semantic chunker for markdown but with section awareness
        semantic_chunker = SemanticChunker(self.chunk_size, self.overlap)
        chunks = semantic_chunker.chunk_text(text, metadata)

        # Add markdown-specific metadata
        for chunk in chunks:
            chunk["content_type"] = "markdown"

        return chunks

    def chunk_plain_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk plain text using fixed-size strategy."""
        fixed_chunker = FixedSizeChunker(self.chunk_size, self.overlap)
        chunks = fixed_chunker.chunk_text(text, metadata)

        # Add content type metadata
        for chunk in chunks:
            chunk["content_type"] = "plain_text"

        return chunks
